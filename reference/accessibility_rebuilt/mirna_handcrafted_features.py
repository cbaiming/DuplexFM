from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import supervised_poc as sp


HANDCRAFTED_FEATURE_PREFIX = "hf_"

_COMPLEMENT = {
    "A": "U",
    "U": "A",
    "G": "C",
    "C": "G",
}

_WOBBLE_TARGET = {
    "A": None,
    "U": "G",
    "G": "U",
    "C": None,
}

_SITE_RANK_VALUE = {
    "none": 0.0,
    "6mer": 0.25,
    "7mer_a1": 0.50,
    "7mer_m8": 0.75,
    "8mer": 1.00,
}

_SITE_RANK_ORDER = {
    "none": 0,
    "6mer": 1,
    "7mer_a1": 2,
    "7mer_m8": 3,
    "8mer": 4,
}

_CANONICAL_SITE_TYPES = ("6mer", "7mer_a1", "7mer_m8", "8mer")

_REFERENCE_PAPERS = [
    {
        "key": "lewis2005",
        "citation": (
            "Lewis BP, Burge CB, Bartel DP. Conserved seed pairing, often flanked by "
            "adenosines, indicates that thousands of human genes are microRNA targets. "
            "Cell. 2005;120(1):15-20."
        ),
        "url": "https://pubmed.ncbi.nlm.nih.gov/15652477/",
    },
    {
        "key": "grimson2007",
        "citation": (
            "Grimson A, Farh KKH, Johnston WK, Garrett-Engele P, Lim LP, Bartel DP. "
            "MicroRNA targeting specificity in mammals: determinants beyond seed pairing. "
            "Mol Cell. 2007;27(1):91-105."
        ),
        "url": "https://pubmed.ncbi.nlm.nih.gov/17612493/",
    },
    {
        "key": "agarwal2015",
        "citation": (
            "Agarwal V, Bell GW, Nam JW, Bartel DP. Predicting effective microRNA target "
            "sites in mammalian mRNAs. eLife. 2015;4:e05005."
        ),
        "url": "https://pubmed.ncbi.nlm.nih.gov/26267216/",
    },
    {
        "key": "enright2003",
        "citation": (
            "Enright AJ, John B, Gaul U, Tuschl T, Sander C, Marks DS. MicroRNA targets "
            "in Drosophila. Genome Biol. 2003;5(1):R1."
        ),
        "url": "https://pubmed.ncbi.nlm.nih.gov/14709173/",
    },
    {
        "key": "reynolds2004",
        "citation": (
            "Reynolds A, Leake D, Boese Q, Scaringe S, Marshall WS, Khvorova A. "
            "Rational siRNA design for RNA interference. Nat Biotechnol. 2004;22(3):326-330."
        ),
        "url": "https://pubmed.ncbi.nlm.nih.gov/14758366/",
    },
    {
        "key": "huesken2005",
        "citation": (
            "Huesken D, Lange J, Mickanin C, et al. Design of a genome-wide siRNA library "
            "using an artificial neural network. Nat Biotechnol. 2005;23(8):995-1001."
        ),
        "url": "https://pubmed.ncbi.nlm.nih.gov/16025137/",
    },
]


@dataclass(frozen=True)
class FeatureDescription:
    name: str
    definition: str
    references: Tuple[str, ...]


@dataclass(frozen=True)
class SeedCandidateSite:
    site_type: str
    start: int
    end: int
    window_start: int
    window_end: int
    mismatch_count: int
    wobble_count: int
    miranda_score: int
    fallback: bool = False


_CLASSIC_FEATURE_DESCRIPTIONS: Tuple[FeatureDescription, ...] = (
    FeatureDescription(
        "site_count_6mer_log1p",
        "log1p of the number of canonical 6mer seed matches in the 3'UTR.",
        ("lewis2005", "grimson2007", "agarwal2015"),
    ),
    FeatureDescription(
        "site_count_7mer_a1_log1p",
        "log1p of the number of canonical 7mer-A1 seed matches in the 3'UTR.",
        ("lewis2005", "grimson2007", "agarwal2015"),
    ),
    FeatureDescription(
        "site_count_7mer_m8_log1p",
        "log1p of the number of canonical 7mer-m8 seed matches in the 3'UTR.",
        ("lewis2005", "grimson2007", "agarwal2015"),
    ),
    FeatureDescription(
        "site_count_8mer_log1p",
        "log1p of the number of canonical 8mer seed matches in the 3'UTR.",
        ("lewis2005", "grimson2007", "agarwal2015"),
    ),
    FeatureDescription(
        "best_site_type_rank_norm",
        "Best canonical site type rank normalized to [0, 1].",
        ("lewis2005", "grimson2007", "agarwal2015"),
    ),
    FeatureDescription(
        "best_seed_mismatch_fraction",
        "Fraction of non-wobble mismatches in the best 7-nt seed-aligned window.",
        ("enright2003",),
    ),
    FeatureDescription(
        "best_seed_wobble_fraction",
        "Fraction of G:U wobble pairs in the best 7-nt seed-aligned window.",
        ("enright2003",),
    ),
    FeatureDescription(
        "best_seed_miranda_score_norm",
        "Best ungapped 7-nt seed-window score normalized by the perfect miRanda seed score.",
        ("enright2003",),
    ),
    FeatureDescription(
        "local_au_fraction_total",
        "Fraction of A/U bases in up to 30-nt upstream and 30-nt downstream flanks, "
        "excluding the matched site span.",
        ("grimson2007", "agarwal2015"),
    ),
    FeatureDescription(
        "local_au_fraction_upstream",
        "Fraction of A/U bases in the upstream flank of the selected site.",
        ("grimson2007", "agarwal2015"),
    ),
    FeatureDescription(
        "local_au_fraction_downstream",
        "Fraction of A/U bases in the downstream flank of the selected site.",
        ("grimson2007", "agarwal2015"),
    ),
)

_EXTENDED_ONLY_FEATURE_DESCRIPTIONS: Tuple[FeatureDescription, ...] = (
    FeatureDescription(
        "mirna_pos1_is_A",
        "One-hot indicator that miRNA position 1 is A.",
        ("reynolds2004", "huesken2005"),
    ),
    FeatureDescription(
        "mirna_pos1_is_U",
        "One-hot indicator that miRNA position 1 is U.",
        ("reynolds2004", "huesken2005"),
    ),
    FeatureDescription(
        "mirna_pos1_is_G",
        "One-hot indicator that miRNA position 1 is G.",
        ("reynolds2004", "huesken2005"),
    ),
    FeatureDescription(
        "mirna_pos1_is_C",
        "One-hot indicator that miRNA position 1 is C.",
        ("reynolds2004", "huesken2005"),
    ),
    FeatureDescription(
        "mirna_pos8_is_A",
        "One-hot indicator that miRNA position 8 is A.",
        ("reynolds2004", "huesken2005"),
    ),
    FeatureDescription(
        "mirna_pos8_is_U",
        "One-hot indicator that miRNA position 8 is U.",
        ("reynolds2004", "huesken2005"),
    ),
    FeatureDescription(
        "mirna_pos8_is_G",
        "One-hot indicator that miRNA position 8 is G.",
        ("reynolds2004", "huesken2005"),
    ),
    FeatureDescription(
        "mirna_pos8_is_C",
        "One-hot indicator that miRNA position 8 is C.",
        ("reynolds2004", "huesken2005"),
    ),
    FeatureDescription(
        "site_count_offset_6mer_log1p",
        "log1p of the number of offset-6mer sites in the 3'UTR, defined by complementarity to miRNA nts 3-8.",
        ("grimson2007", "agarwal2015"),
    ),
    FeatureDescription(
        "target_site_gc_fraction",
        "GC fraction of the selected reference target site span.",
        ("reynolds2004", "huesken2005"),
    ),
)

_FEATURE_SET_DESCRIPTIONS: Dict[str, Tuple[FeatureDescription, ...]] = {
    "classic_seed_context": _CLASSIC_FEATURE_DESCRIPTIONS,
    "classic_seed_context_extended": _CLASSIC_FEATURE_DESCRIPTIONS + _EXTENDED_ONLY_FEATURE_DESCRIPTIONS,
}

_CLASSIC_FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "canonical_site": (
        "site_count_6mer_log1p",
        "site_count_7mer_a1_log1p",
        "site_count_7mer_m8_log1p",
        "site_count_8mer_log1p",
        "best_site_type_rank_norm",
    ),
    "seed_pairing_quality": (
        "best_seed_mismatch_fraction",
        "best_seed_wobble_fraction",
        "best_seed_miranda_score_norm",
    ),
    "local_au_context": (
        "local_au_fraction_total",
        "local_au_fraction_upstream",
        "local_au_fraction_downstream",
    ),
}

_FEATURE_SET_GROUPS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "classic_seed_context": _CLASSIC_FEATURE_GROUPS,
    "classic_seed_context_extended": {
        **_CLASSIC_FEATURE_GROUPS,
        "extended_only": tuple(item.name for item in _EXTENDED_ONLY_FEATURE_DESCRIPTIONS),
    },
}


SEED_CANDIDATE_SITE_TYPES: Tuple[str, ...] = (
    "offset_6mer",
    "6mer",
    "7mer_a1",
    "7mer_m8",
    "8mer",
    "near_seed_like",
)

_SEED_CANDIDATE_SITE_TYPE_TO_ID: Dict[str, int] = {
    site_type: index for index, site_type in enumerate(SEED_CANDIDATE_SITE_TYPES)
}

_SEED_CANDIDATE_PRIORITY: Dict[str, int] = {
    "near_seed_like": 0,
    "offset_6mer": 1,
    "6mer": 2,
    "7mer_a1": 3,
    "7mer_m8": 4,
    "8mer": 5,
}


def _normalize_sequence(value: str) -> str:
    return sp.normalize_rna(value)


def _reverse_complement_rna(seq: str) -> str:
    return "".join(_COMPLEMENT.get(base, "N") for base in reversed(seq))


def _score_target_base(mirna_base: str, target_base: str) -> Tuple[int, int, int]:
    canonical_base = _COMPLEMENT.get(mirna_base)
    wobble_base = _WOBBLE_TARGET.get(mirna_base)
    if canonical_base is not None and target_base == canonical_base:
        return 5, 0, 0
    if wobble_base is not None and target_base == wobble_base:
        return 2, 0, 1
    return -3, 1, 0


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _au_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    au_count = sum(1 for base in seq if base in {"A", "U"})
    return float(au_count) / float(len(seq))


def _build_feature_row(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _normalize_feature_set(feature_set: Optional[str]) -> str:
    if feature_set is None:
        return "none"
    normalized = str(feature_set).strip().lower()
    if normalized in {"", "none"}:
        return "none"
    if normalized not in _FEATURE_SET_DESCRIPTIONS:
        raise ValueError(f"未知 handcrafted feature set: {feature_set!r}")
    return normalized


def get_feature_names(feature_set: str) -> List[str]:
    normalized = _normalize_feature_set(feature_set)
    if normalized == "none":
        return []
    return [item.name for item in _FEATURE_SET_DESCRIPTIONS[normalized]]


def _normalize_group_selection(value: Optional[Sequence[str] | str]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = value.split(",")
    else:
        candidates = list(value)
    normalized = []
    for item in candidates:
        text = str(item).strip()
        if not text:
            continue
        normalized.append(text)
    return normalized


def get_feature_groups(feature_set: str) -> Dict[str, List[str]]:
    normalized = _normalize_feature_set(feature_set)
    if normalized == "none":
        return {}
    group_map = _FEATURE_SET_GROUPS.get(normalized, {})
    return {group_name: list(feature_names) for group_name, feature_names in group_map.items()}


def resolve_selected_feature_names(
    feature_set: str,
    include_groups: Optional[Sequence[str] | str] = None,
    exclude_groups: Optional[Sequence[str] | str] = None,
) -> List[str]:
    normalized = _normalize_feature_set(feature_set)
    all_feature_names = get_feature_names(normalized)
    if normalized == "none":
        return []

    feature_groups = get_feature_groups(normalized)
    include_group_names = _normalize_group_selection(include_groups)
    exclude_group_names = _normalize_group_selection(exclude_groups)
    unknown_groups = sorted(
        (set(include_group_names) | set(exclude_group_names)) - set(feature_groups)
    )
    if unknown_groups:
        raise ValueError(
            f"{feature_set!r} 包含未知 feature groups: {unknown_groups}. "
            f"可用 groups: {sorted(feature_groups)}"
        )

    if include_group_names:
        included = {
            feature_name
            for group_name in include_group_names
            for feature_name in feature_groups[group_name]
        }
    else:
        included = set(all_feature_names)

    excluded = {
        feature_name
        for group_name in exclude_group_names
        for feature_name in feature_groups[group_name]
    }
    selected = included - excluded
    return [feature_name for feature_name in all_feature_names if feature_name in selected]


def get_selected_prefixed_feature_columns(
    feature_set: str,
    include_groups: Optional[Sequence[str] | str] = None,
    exclude_groups: Optional[Sequence[str] | str] = None,
) -> List[str]:
    return [
        HANDCRAFTED_FEATURE_PREFIX + feature_name
        for feature_name in resolve_selected_feature_names(
            feature_set=feature_set,
            include_groups=include_groups,
            exclude_groups=exclude_groups,
        )
    ]


def get_prefixed_feature_columns(feature_set: str) -> List[str]:
    return [HANDCRAFTED_FEATURE_PREFIX + name for name in get_feature_names(feature_set)]


def build_feature_reference_manifest(
    feature_set: str,
    include_groups: Optional[Sequence[str] | str] = None,
    exclude_groups: Optional[Sequence[str] | str] = None,
) -> Dict[str, Any]:
    normalized = _normalize_feature_set(feature_set)
    if normalized == "none":
        return {
            "feature_set": "none",
            "enabled": False,
            "feature_columns": [],
            "features": [],
            "references": [],
            "available_feature_groups": {},
            "selected_feature_groups": {},
            "selected_feature_names": [],
        }
    reference_map = {item["key"]: item for item in _REFERENCE_PAPERS}
    selected_feature_names = resolve_selected_feature_names(
        feature_set=normalized,
        include_groups=include_groups,
        exclude_groups=exclude_groups,
    )
    selected_feature_name_set = set(selected_feature_names)
    available_feature_groups = get_feature_groups(normalized)
    selected_feature_groups = {
        group_name: [
            feature_name
            for feature_name in feature_names
            if feature_name in selected_feature_name_set
        ]
        for group_name, feature_names in available_feature_groups.items()
        if any(feature_name in selected_feature_name_set for feature_name in feature_names)
    }
    features = []
    inherited_names = {item.name for item in _CLASSIC_FEATURE_DESCRIPTIONS}
    active_descriptions = _FEATURE_SET_DESCRIPTIONS[normalized]
    for description in active_descriptions:
        if description.name not in selected_feature_name_set:
            continue
        features.append(
            {
                "name": description.name,
                "column": HANDCRAFTED_FEATURE_PREFIX + description.name,
                "definition": description.definition,
                "reference_keys": list(description.references),
                "citations": [reference_map[key]["citation"] for key in description.references],
                "urls": [reference_map[key]["url"] for key in description.references],
                "source_group": (
                    "classic_seed_context"
                    if description.name in inherited_names
                    else "classic_seed_context_extended_only"
                ),
            }
        )
    return {
        "feature_set": normalized,
        "enabled": True,
        "feature_columns": [
            HANDCRAFTED_FEATURE_PREFIX + feature_name
            for feature_name in selected_feature_names
        ],
        "available_feature_groups": available_feature_groups,
        "selected_feature_groups": selected_feature_groups,
        "selected_feature_names": selected_feature_names,
        "features": features,
        "references": list(_REFERENCE_PAPERS),
        "rules": {
            "canonical_site_priority": ["8mer", "7mer_m8", "7mer_a1", "6mer"],
            "seed_window": "miRNA nts 2-8",
            "miranda_scoring": {"+5": "Watson-Crick", "+2": "G:U wobble", "-3": "mismatch"},
            "local_au_window_nt": 30,
            "offset_6mer_window": "miRNA nts 3-8",
        },
    }


def get_seed_candidate_site_types() -> List[str]:
    return list(SEED_CANDIDATE_SITE_TYPES)


def get_seed_candidate_site_type_to_id() -> Dict[str, int]:
    return dict(_SEED_CANDIDATE_SITE_TYPE_TO_ID)


def _score_seed_span(mirna_segment: str, target_span: str) -> Tuple[int, int, int]:
    mirrored_seed = _normalize_sequence(mirna_segment)[::-1]
    target_norm = _normalize_sequence(target_span)
    if len(mirrored_seed) != len(target_norm):
        raise ValueError(
            "mirna_segment 与 target_span 长度必须一致，"
            f"got {len(mirrored_seed)} vs {len(target_norm)}"
        )

    score = 0
    mismatch_count = 0
    wobble_count = 0
    for mirna_base, target_base in zip(mirrored_seed, target_norm):
        delta, mismatch_inc, wobble_inc = _score_target_base(mirna_base, target_base)
        score += delta
        mismatch_count += mismatch_inc
        wobble_count += wobble_inc
    return score, mismatch_count, wobble_count


def _candidate_sort_key(site: SeedCandidateSite) -> Tuple[int, int, int, int, int, int]:
    return (
        site.start,
        -_SEED_CANDIDATE_PRIORITY.get(site.site_type, -1),
        -site.miranda_score,
        site.mismatch_count,
        site.wobble_count,
        site.end,
    )


def _register_seed_candidate(
    registry: Dict[Tuple[int, int], SeedCandidateSite],
    site: SeedCandidateSite,
) -> None:
    key = (site.start, site.end)
    existing = registry.get(key)
    if existing is None:
        registry[key] = site
        return
    current_rank = _SEED_CANDIDATE_PRIORITY.get(existing.site_type, -1)
    new_rank = _SEED_CANDIDATE_PRIORITY.get(site.site_type, -1)
    if new_rank > current_rank:
        registry[key] = site
        return
    if new_rank < current_rank:
        return
    if (
        site.miranda_score > existing.miranda_score
        or (
            site.miranda_score == existing.miranda_score
            and (
                site.mismatch_count < existing.mismatch_count
                or (
                    site.mismatch_count == existing.mismatch_count
                    and (
                        site.wobble_count < existing.wobble_count
                        or (
                            site.wobble_count == existing.wobble_count
                            and site.start < existing.start
                        )
                    )
                )
            )
        )
    ):
        registry[key] = site


def enumerate_seed_candidate_sites(
    gene: str,
    mirna: str,
    *,
    include_offset_6mer: bool = True,
    include_near_seed_like: bool = True,
    max_non_wobble_mismatch: int = 1,
    max_wobble: int = 1,
) -> List[Dict[str, Any]]:
    gene_norm = _normalize_sequence(gene)
    mirna_norm = _normalize_sequence(mirna)
    if not gene_norm or not mirna_norm:
        return []

    registry: Dict[Tuple[int, int], SeedCandidateSite] = {}

    if include_offset_6mer and len(gene_norm) >= 6 and len(mirna_norm) >= 8:
        offset_seed = _reverse_complement_rna(mirna_norm[2:8])
        for start in range(len(gene_norm) - 6 + 1):
            span = gene_norm[start : start + 6]
            if span != offset_seed:
                continue
            score, mismatch_count, wobble_count = _score_seed_span(mirna_norm[2:8], span)
            _register_seed_candidate(
                registry,
                SeedCandidateSite(
                    site_type="offset_6mer",
                    start=start,
                    end=start + 6,
                    window_start=start,
                    window_end=start + 6,
                    mismatch_count=mismatch_count,
                    wobble_count=wobble_count,
                    miranda_score=score,
                ),
            )

    if len(gene_norm) >= 7 and len(mirna_norm) >= 8:
        for window_start in range(len(gene_norm) - 7 + 1):
            candidate = _classify_seed_window(
                gene=gene_norm,
                mirna=mirna_norm,
                window_start=window_start,
            )
            if candidate is None:
                continue
            site_type = str(candidate["site_type"])
            if site_type != "none":
                _register_seed_candidate(
                    registry,
                    SeedCandidateSite(
                        site_type=site_type,
                        start=int(candidate["site_span_start"]),
                        end=int(candidate["site_span_end"]),
                        window_start=int(window_start),
                        window_end=int(window_start + 7),
                        mismatch_count=int(candidate["mismatch_count"]),
                        wobble_count=int(candidate["wobble_count"]),
                        miranda_score=int(candidate["miranda_score"]),
                    ),
                )
                continue
            if not include_near_seed_like:
                continue
            mismatch_count = int(candidate["mismatch_count"])
            wobble_count = int(candidate["wobble_count"])
            if mismatch_count <= max_non_wobble_mismatch and wobble_count <= max_wobble:
                _register_seed_candidate(
                    registry,
                    SeedCandidateSite(
                        site_type="near_seed_like",
                        start=window_start,
                        end=window_start + 7,
                        window_start=window_start,
                        window_end=window_start + 7,
                        mismatch_count=mismatch_count,
                        wobble_count=wobble_count,
                        miranda_score=int(candidate["miranda_score"]),
                    ),
                )

    ordered = sorted(registry.values(), key=_candidate_sort_key)
    return [
        {
            "site_type": site.site_type,
            "start": int(site.start),
            "end": int(site.end),
            "window_start": int(site.window_start),
            "window_end": int(site.window_end),
            "mismatch_count": int(site.mismatch_count),
            "wobble_count": int(site.wobble_count),
            "miranda_score": int(site.miranda_score),
            "fallback": bool(site.fallback),
        }
        for site in ordered
    ]


def select_seed_fallback_site(gene: str, mirna: str) -> Optional[Dict[str, Any]]:
    gene_norm = _normalize_sequence(gene)
    mirna_norm = _normalize_sequence(mirna)
    if len(gene_norm) < 7 or len(mirna_norm) < 8:
        return None

    candidates: List[SeedCandidateSite] = []
    for window_start in range(len(gene_norm) - 7 + 1):
        candidate = _classify_seed_window(
            gene=gene_norm,
            mirna=mirna_norm,
            window_start=window_start,
        )
        if candidate is None:
            continue
        candidates.append(
            SeedCandidateSite(
                site_type="near_seed_like",
                start=window_start,
                end=window_start + 7,
                window_start=window_start,
                window_end=window_start + 7,
                mismatch_count=int(candidate["mismatch_count"]),
                wobble_count=int(candidate["wobble_count"]),
                miranda_score=int(candidate["miranda_score"]),
                fallback=True,
            )
        )
    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            -item.miranda_score,
            item.mismatch_count,
            item.wobble_count,
            item.start,
        )
    )
    best = candidates[0]
    return {
        "site_type": best.site_type,
        "start": int(best.start),
        "end": int(best.end),
        "window_start": int(best.window_start),
        "window_end": int(best.window_end),
        "mismatch_count": int(best.mismatch_count),
        "wobble_count": int(best.wobble_count),
        "miranda_score": int(best.miranda_score),
        "fallback": True,
    }


def _collect_canonical_site_counts(gene: str, mirna: str) -> Dict[str, int]:
    counts = {site_type: 0 for site_type in _CANONICAL_SITE_TYPES}
    if len(gene) < 6 or len(mirna) < 7:
        return counts

    seed_6mer = _reverse_complement_rna(mirna[1:7])
    seed_7mer_m8 = _reverse_complement_rna(mirna[1:8]) if len(mirna) >= 8 else ""
    for start in range(len(gene)):
        site_type = "none"
        if len(mirna) >= 8 and start + 8 <= len(gene):
            if gene[start : start + 7] == seed_7mer_m8 and gene[start + 7] == "A":
                site_type = "8mer"
        if site_type == "none" and len(mirna) >= 8 and start + 7 <= len(gene):
            if gene[start : start + 7] == seed_7mer_m8:
                site_type = "7mer_m8"
        if site_type == "none" and start + 7 <= len(gene):
            if gene[start : start + 6] == seed_6mer and gene[start + 6] == "A":
                site_type = "7mer_a1"
        if site_type == "none" and start + 6 <= len(gene):
            if gene[start : start + 6] == seed_6mer:
                site_type = "6mer"
        if site_type != "none":
            counts[site_type] += 1
    return counts


def _classify_seed_window(
    gene: str,
    mirna: str,
    window_start: int,
) -> Optional[Dict[str, Any]]:
    if len(mirna) < 8 or window_start < 0 or window_start + 7 > len(gene):
        return None

    window = gene[window_start : window_start + 7]
    seed_site = _reverse_complement_rna(mirna[1:8])
    seed_6mer = _reverse_complement_rna(mirna[1:7])
    downstream_base = gene[window_start + 7] if window_start + 7 < len(gene) else None

    site_type = "none"
    site_span_start = window_start
    site_span_end = window_start + 7
    if window == seed_site:
        if downstream_base == "A":
            site_type = "8mer"
            site_span_start = window_start
            site_span_end = window_start + 8
        else:
            site_type = "7mer_m8"
            site_span_start = window_start
            site_span_end = window_start + 7
    elif window[1:] == seed_6mer:
        if downstream_base == "A":
            site_type = "7mer_a1"
            site_span_start = window_start + 1
            site_span_end = window_start + 8
        else:
            site_type = "6mer"
            site_span_start = window_start + 1
            site_span_end = window_start + 7

    mirrored_seed = mirna[1:8][::-1]
    score = 0
    mismatch_count = 0
    wobble_count = 0
    for idx, mirna_base in enumerate(mirrored_seed):
        delta, mismatch_inc, wobble_inc = _score_target_base(mirna_base, window[idx])
        score += delta
        mismatch_count += mismatch_inc
        wobble_count += wobble_inc

    upstream = gene[max(0, site_span_start - 30) : site_span_start]
    downstream = gene[site_span_end : min(len(gene), site_span_end + 30)]
    local_au_up = _au_fraction(upstream)
    local_au_down = _au_fraction(downstream)
    total_context = upstream + downstream
    local_au_total = _au_fraction(total_context)

    return {
        "window_start": window_start,
        "window": window,
        "site_type": site_type,
        "site_rank": _SITE_RANK_ORDER[site_type],
        "site_rank_norm": _SITE_RANK_VALUE[site_type],
        "site_span_start": site_span_start,
        "site_span_end": site_span_end,
        "miranda_score": score,
        "miranda_score_norm": score / 35.0,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": _safe_fraction(mismatch_count, 7),
        "wobble_count": wobble_count,
        "wobble_fraction": _safe_fraction(wobble_count, 7),
        "local_au_total": local_au_total,
        "local_au_upstream": local_au_up,
        "local_au_downstream": local_au_down,
    }


def _pick_reference_window(gene: str, mirna: str) -> Optional[Dict[str, Any]]:
    if len(mirna) < 8 or len(gene) < 7:
        return None
    candidates = []
    for start in range(len(gene) - 7 + 1):
        candidate = _classify_seed_window(gene=gene, mirna=mirna, window_start=start)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item["site_rank"],
            item["miranda_score"],
            -item["mismatch_fraction"],
            -item["wobble_fraction"],
            item["local_au_total"],
            -item["window_start"],
        ),
        reverse=True,
    )
    return candidates[0]


def _count_offset_6mer_sites(gene: str, mirna: str) -> int:
    if len(gene) < 6 or len(mirna) < 8:
        return 0
    offset_seed = _reverse_complement_rna(mirna[2:8])
    return sum(
        1
        for start in range(len(gene) - 6 + 1)
        if gene[start : start + 6] == offset_seed
    )


def _one_hot_base_at_position(seq: str, zero_based_position: int) -> List[float]:
    if zero_based_position < 0 or zero_based_position >= len(seq):
        return [0.0, 0.0, 0.0, 0.0]
    base = seq[zero_based_position]
    return [
        1.0 if base == "A" else 0.0,
        1.0 if base == "U" else 0.0,
        1.0 if base == "G" else 0.0,
        1.0 if base == "C" else 0.0,
    ]


def compute_classic_seed_context_features(gene: str, mirna: str) -> np.ndarray:
    gene_norm = _normalize_sequence(gene)
    mirna_norm = _normalize_sequence(mirna)
    if len(gene_norm) == 0 or len(mirna_norm) < 8:
        return _build_feature_row([0.0] * len(_CLASSIC_FEATURE_DESCRIPTIONS))

    site_counts = _collect_canonical_site_counts(gene=gene_norm, mirna=mirna_norm)
    reference = _pick_reference_window(gene=gene_norm, mirna=mirna_norm)
    if reference is None:
        reference = {
            "site_rank_norm": 0.0,
            "mismatch_fraction": 0.0,
            "wobble_fraction": 0.0,
            "miranda_score_norm": 0.0,
            "local_au_total": 0.0,
            "local_au_upstream": 0.0,
            "local_au_downstream": 0.0,
        }

    return _build_feature_row(
        [
            math.log1p(site_counts["6mer"]),
            math.log1p(site_counts["7mer_a1"]),
            math.log1p(site_counts["7mer_m8"]),
            math.log1p(site_counts["8mer"]),
            float(reference["site_rank_norm"]),
            float(reference["mismatch_fraction"]),
            float(reference["wobble_fraction"]),
            float(reference["miranda_score_norm"]),
            float(reference["local_au_total"]),
            float(reference["local_au_upstream"]),
            float(reference["local_au_downstream"]),
        ]
    )


def compute_classic_seed_context_extended_features(gene: str, mirna: str) -> np.ndarray:
    gene_norm = _normalize_sequence(gene)
    mirna_norm = _normalize_sequence(mirna)
    classic = compute_classic_seed_context_features(gene=gene_norm, mirna=mirna_norm)
    reference = _pick_reference_window(gene=gene_norm, mirna=mirna_norm)
    site_gc_fraction = 0.0
    if reference is not None:
        site_seq = gene_norm[reference["site_span_start"] : reference["site_span_end"]]
        if site_seq:
            site_gc_fraction = float(sum(1 for base in site_seq if base in {"G", "C"})) / float(len(site_seq))

    extended = _build_feature_row(
        [
            *_one_hot_base_at_position(mirna_norm, 0),
            *_one_hot_base_at_position(mirna_norm, 7),
            math.log1p(_count_offset_6mer_sites(gene=gene_norm, mirna=mirna_norm)),
            site_gc_fraction,
        ]
    )
    return np.concatenate([classic, extended]).astype(np.float32, copy=False)


def compute_feature_matrix(
    feature_set: str,
    genes: Sequence[str],
    mirnas: Sequence[str],
    workers: int = 1,
    chunk_size: int = 50_000,
) -> np.ndarray:
    normalized = _normalize_feature_set(feature_set)
    if normalized == "none":
        return np.zeros((len(genes), 0), dtype=np.float32)
    if int(workers) > 1 and len(genes) > int(chunk_size):
        payloads = [
            (
                feature_set,
                list(genes[start : start + int(chunk_size)]),
                list(mirnas[start : start + int(chunk_size)]),
            )
            for start in range(0, len(genes), int(chunk_size))
        ]
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            chunks = list(executor.map(_compute_feature_matrix_chunk, payloads))
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    if normalized == "classic_seed_context":
        rows = [
            compute_classic_seed_context_features(gene=gene, mirna=mirna)
            for gene, mirna in zip(genes, mirnas)
        ]
    elif normalized == "classic_seed_context_extended":
        rows = [
            compute_classic_seed_context_extended_features(gene=gene, mirna=mirna)
            for gene, mirna in zip(genes, mirnas)
        ]
    else:
        raise ValueError(f"未知 handcrafted feature set: {feature_set!r}")
    if not rows:
        return np.zeros((0, len(get_feature_names(feature_set))), dtype=np.float32)
    return np.vstack(rows).astype(np.float32, copy=False)


def _compute_feature_matrix_chunk(payload: Tuple[str, List[str], List[str]]) -> np.ndarray:
    feature_set, genes, mirnas = payload
    return compute_feature_matrix(feature_set, genes, mirnas, workers=1)


def augment_dataframe_with_features(
    df: pd.DataFrame,
    feature_set: str,
    gene_column: str = "gene",
    mirna_column: str = "noncodingRNA",
    include_groups: Optional[Sequence[str] | str] = None,
    exclude_groups: Optional[Sequence[str] | str] = None,
    workers: int = 1,
) -> pd.DataFrame:
    all_feature_names = get_feature_names(feature_set)
    selected_feature_names = resolve_selected_feature_names(
        feature_set=feature_set,
        include_groups=include_groups,
        exclude_groups=exclude_groups,
    )
    if not selected_feature_names:
        return df
    matrix = compute_feature_matrix(
        feature_set=feature_set,
        genes=df[gene_column].astype(str).tolist(),
        mirnas=df[mirna_column].astype(str).tolist(),
        workers=workers,
    )
    augmented = df.copy()
    feature_index_map = {name: index for index, name in enumerate(all_feature_names)}
    for feature_name in selected_feature_names:
        feature_index = feature_index_map[feature_name]
        column_name = HANDCRAFTED_FEATURE_PREFIX + feature_name
        augmented[column_name] = matrix[:, feature_index]
    return augmented


def summarize_feature_columns(
    df: pd.DataFrame,
    feature_set: str,
    include_groups: Optional[Sequence[str] | str] = None,
    exclude_groups: Optional[Sequence[str] | str] = None,
) -> Dict[str, Any]:
    columns = get_selected_prefixed_feature_columns(
        feature_set=feature_set,
        include_groups=include_groups,
        exclude_groups=exclude_groups,
    )
    if not columns:
        return {"feature_set": "none", "columns": [], "row_count": int(len(df))}
    return {
        "feature_set": feature_set,
        "available_feature_groups": get_feature_groups(feature_set),
        "selected_feature_groups": {
            group_name: feature_names
            for group_name, feature_names in get_feature_groups(feature_set).items()
            if any(
                HANDCRAFTED_FEATURE_PREFIX + feature_name in columns
                for feature_name in feature_names
            )
        },
        "selected_feature_names": resolve_selected_feature_names(
            feature_set=feature_set,
            include_groups=include_groups,
            exclude_groups=exclude_groups,
        ),
        "columns": columns,
        "row_count": int(len(df)),
        "column_stats": {
            column: {
                "mean": float(df[column].mean()),
                "std": float(df[column].std(ddof=0)),
                "min": float(df[column].min()),
                "max": float(df[column].max()),
            }
            for column in columns
        },
    }
