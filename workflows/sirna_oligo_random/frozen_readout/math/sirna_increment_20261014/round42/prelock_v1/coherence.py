"""Descriptive qualification of an a priori regional coefficient path."""
from itertools import permutations
import numpy as np

EDGES = ((0, 2), (2, 4), (1, 3), (3, 5))
PAIRS = ((0, 2), (0, 4), (2, 4), (1, 3), (1, 5), (3, 5))
REGION_PERMUTATIONS = tuple(permutations(range(3)))


def matrix(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (6, 32) or not np.isfinite(value).all():
        raise ValueError("Finite six-pool by 32-channel matrix required")
    return value


def laplacian():
    result = np.zeros((6, 6))
    for a, b in EDGES:
        v = np.eye(6)[a] - np.eye(6)[b]
        result += np.outer(v, v)
    return result


def energy(value):
    value = matrix(value)
    # Remove two irrelevant common vectors; avoids translation cancellation.
    c = value.reshape(3, 2, 32)
    c = (c - c.mean(axis=0, keepdims=True)).reshape(6, 32)
    pair = np.array([np.sum((c[a] - c[b]) ** 2) for a, b in PAIRS])
    total = float(pair.sum())
    adjacent = float(sum(np.sum((c[a] - c[b]) ** 2) for a, b in EDGES))
    permuted = []
    for pi in REGION_PERMUTATIONS:
        cp = c.reshape(3, 2, 32)[list(pi)].reshape(6, 32)
        permuted.append(float(sum(np.sum((cp[a] - cp[b]) ** 2) for a, b in EDGES)))
    null_mean = float(np.mean(permuted))
    # If regions are identical (or numerically so), their ordering has no evidence.
    active = bool(total > 1e-24)
    return dict(active=active, adjacent_energy=adjacent, all_pair_energy=total,
                permutation_mean_energy=null_mean,
                smoothness_ratio=adjacent / null_mean if active else None,
                permuted_path_energies=permuted,
                pair_energies=pair.tolist())


def aggregate(rows):
    if len(rows) != 5:
        raise ValueError("Five original inner fits required")
    ratios = [row["smoothness_ratio"] for row in rows]
    all_active = all(row["active"] and x is not None and np.isfinite(x)
                     for row, x in zip(rows, ratios))
    mean = float(np.mean(ratios)) if all_active else None
    count = int(sum(x is not None and x < 1.0 for x in ratios))
    passed = bool(all_active and mean <= .9 and count >= 4)
    return dict(all_active=all_active, mean_smoothness_ratio=mean,
                below_permutation_mean_count=count, qualified=passed)


def qualify(records):
    core = [r for r in records if r["source"] == "core_real"]
    if len(core) != 3 or len({r["setting"] for r in core}) != 3:
        raise ValueError("Three distinct primary pools required")
    count = sum(r["qualified"] for r in core)
    return dict(qualified=bool(count >= 2), qualified_core_pools=int(count))
