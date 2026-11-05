# Explicit pre-data boundary compatibility repair

The original synthetic fixture failed for a 60nt pair with exactly 54 identical bases. Changing word sizes did not repair it. Source inspection found that `tiden_no / (float)len_eff1` was compared to a double 0.9 threshold. Preserve the original source, binary, failed fixture, and all diagnostics. No real sequence search or new metrics preceded this repair.

A separate CD-HIT 4.8.1 build compares integer identity counts to threshold times alignment length (1e-9 roundoff tolerance), and uses ceil rather than floor for both coverage requirements. These four source-line replacements implement the already stated inclusive 90% identity and >=80% coverage; they do not change the scientific cutoff. Patched CPP SHA256: 9f767b65ef33ab1ed490722e2320b412f4127147d7ebf68e5c4811b6ea2dcec4; binary SHA256: f3288f84b057d9d0db3bf8edb0b2db2c3c8de88627eeafa2067e321f993f2b99.

`purge_sequences_compat.py` binds the separate tool identity and compatibility lock, stores new synthetic results separately, and adds coverage-boundary fixtures (40/51 vs 41/51). All old files remain. Synthetic sequences changed deterministically to include the additional 51nt reference; no measured data were used.
