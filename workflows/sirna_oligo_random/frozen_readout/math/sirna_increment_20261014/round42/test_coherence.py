from itertools import permutations
import numpy as np
import pytest
from coherence import energy, laplacian, aggregate, qualify


def test_graph_quadratic_matches_path_and_psd():
    c = np.random.default_rng(31).normal(size=(6, 32))
    l = laplacian()
    assert np.allclose(l, l.T)
    assert np.min(np.linalg.eigvalsh(l)) >= -1e-12
    assert np.allclose(np.linalg.eigvalsh(l), [0, 0, 1, 1, 3, 3])
    assert np.isclose(np.trace(c.T @ l @ c), energy(c)["adjacent_energy"])


def test_linear_spike_constant_profiles():
    c = np.arange(3)[:, None, None] * np.ones((3, 2, 32))
    # Each channel: adjacent=1+1=2, all-pairs=1+4+1=6, null=4.
    assert np.isclose(energy(c.reshape(6, 32))["smoothness_ratio"], .5)
    c[0] = c[2] = 0
    c[1] = 1
    assert np.isclose(energy(c.reshape(6, 32))["smoothness_ratio"], 1.5)
    e = energy(np.ones((6, 32)))
    assert not e["active"] and e["smoothness_ratio"] is None


def test_permutation_expectation_is_fixed_two_thirds():
    c = np.random.default_rng(4).normal(size=(3, 2, 32))
    values = [energy(c[list(pi)].reshape(6, 32)) for pi in permutations(range(3))]
    assert np.isclose(np.mean([e["smoothness_ratio"] for e in values]), 1.)
    assert all(np.isclose(e["permutation_mean_energy"], 2 * e["all_pair_energy"] / 3) for e in values)


def test_shared_channel_basis_invariance():
    rng = np.random.default_rng(8)
    c = rng.normal(size=(6, 32))
    q, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    assert np.isclose(energy(c)["smoothness_ratio"], energy(c @ q)["smoothness_ratio"])


def test_scale_common_translation_and_no_mutation():
    rng = np.random.default_rng(10)
    c = rng.normal(size=(6, 32)); saved = c.copy()
    common = np.tile(rng.normal(size=(2, 32)), (3, 1))
    assert np.isclose(energy(c)["smoothness_ratio"], energy(4 * c + common)["smoothness_ratio"])
    assert np.array_equal(c, saved)


def test_independent_scalar_pair_reconstruction():
    c = np.random.default_rng(2).normal(size=(6, 32))
    adjacent = sum((c[2 * r + t, j] - c[2 * (r + 1) + t, j]) ** 2
                   for t in range(2) for r in range(2) for j in range(32))
    all_pairs = sum((c[2 * a + t, j] - c[2 * b + t, j]) ** 2
                   for t in range(2) for a in range(3) for b in range(a + 1, 3) for j in range(32))
    result = energy(c)
    assert np.isclose(result["adjacent_energy"], adjacent)
    assert np.isclose(result["smoothness_ratio"], adjacent / (2 * all_pairs / 3))


def test_invalid_inputs_rejected():
    for c in [np.zeros((32, 6)), np.zeros((6, 31)), np.full((6, 32), np.inf), np.full((6, 32), np.nan)]:
        with pytest.raises(ValueError):
            energy(c)
    with pytest.raises(ValueError):
        aggregate([])
    with pytest.raises(ValueError):
        qualify([])


def test_gate_requires_consistent_ordering_not_just_average():
    rows = [dict(active=True, smoothness_ratio=.8) for _ in range(5)]
    assert aggregate(rows)["qualified"]
    rows[0]["smoothness_ratio"] = 1.01
    rows[1]["smoothness_ratio"] = 1.01
    assert not aggregate(rows)["qualified"]
    good = [dict(setting=s, source="core_real", qualified=True) for s in ("A", "B", "C")]
    good[0]["qualified"] = False
    assert qualify(good)["qualified"]
    good[1]["qualified"] = False
    assert not qualify(good)["qualified"]
    good.append(dict(setting="A", source="full_real", qualified=True))
    assert not qualify(good)["qualified"]
