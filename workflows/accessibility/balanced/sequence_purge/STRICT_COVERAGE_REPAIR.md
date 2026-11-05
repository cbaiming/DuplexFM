# Coverage repair before any new metric computation

The first search was rejected by the independent audit: 1,997 reported hits did not cover 80% of the true query length. CD-HIT-EST-2D uses the shortest indexed reference length for candidate pruning and reused that value for query coverage when queries may be longer than references. The first provisional 34,808 exclusions are not the accepted final filter. No new-subset performance was computed.

Preserve all first-search outputs and input_audit_failed_v1.json. A separate boundaryfix2 build adds final acceptance checks against the actual query and reference lengths, with the same 90% identity and 80% coverage thresholds. Original word filters remain; absence of hits is still operational, not exhaustive homology certification. Add mixed-length regression fixtures (47/60 must not qualify, 48/60 must qualify), and repeat sequence search against the same reference/query files. No labels, predictions, model changes, or inference are involved.

New results go to strict_v2; original prepared reference/query FASTA are reused read-only. Binary SHA256 2428f73fa17c03a5539a4a690675aecaff802f07f0db9dba7026afe2eba519a9. Patched source SHA256 e92c9efbed0bc5ef9b555ca9faec77b69c0dd185ac426df4286aee37f583142d. The original model/training artifacts are unchanged.
