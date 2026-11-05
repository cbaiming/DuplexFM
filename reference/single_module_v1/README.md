# Single-branch implementation

This directory retains the three separately trained source configurations: nucleotide-pair, contextual cross-interaction and target reactivity. Coordinates, convolution, site pooling and classification remain shared architecture components. Model masks and the active-branch execution paths are in `code/duplexfm_mechanism/model.py` and `protocol.py`.

No single-branch performance tables, predictions or fitted weights are distributed here. The implementation and synthetic tests are retained so that the configurations can be examined and rerun with separately supplied source data, foundations and initialization assets. Changing active computation also changes dropout streams; equivalence to the earlier full-computation training trajectory is not claimed.

Run the isolated CPU checks with `python scripts/run_cpu_checks.py --only single_module` from the repository root. Tests use synthetic tensors and mocked foundation encoders. They do not establish performance or reproduce source training.
