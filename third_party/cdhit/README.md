# Modified CD-HIT source used by the external-input filter

Upstream: [CD-HIT V4.8.1](https://github.com/weizhongli/cdhit/tree/V4.8.1), independently distributed under the accompanying GPLv2 license. This subdirectory is excluded from the project's original-material notice.

`cdhit-common.c++` is the preserved boundaryfix2 file with a prominent modification/date comment added during public packaging. The scientific modifications implement inclusive90%identity, ceil coverage boundaries, and final coverage acceptance against both true sequence lengths. Algorithm thresholds and word filters remain unchanged. Packaging comments change the file SHA but not its compiled logic. The original patched-file SHA is `e92c9efbed0bc5ef9b555ca9faec77b69c0dd185ac426df4286aee37f583142d`.

Obtain the remaining source from V4.8.1, replace its `cdhit-common.c++` with this declared file, and compile in a new directory. The historical binary SHA was `2428f73fa17c03a5539a4a690675aecaff802f07f0db9dba7026afe2eba519a9`; a different build requires functional fixture validation and its own identity, rather than relabelling it the historical binary.

The corresponding Python fixtures and final coverage audit are in `workflows/accessibility/balanced/sequence_purge/`. Their historical input/execution locks are archival requirements; do not fabricate them from a different run. The saved final masks and match evidence permit reviewing the completed filtering without rerunning it.
