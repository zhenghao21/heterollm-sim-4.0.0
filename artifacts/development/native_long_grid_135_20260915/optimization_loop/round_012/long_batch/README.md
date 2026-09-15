# Round 10 dual-reference probe
New immutable v3, copied from round008/r2. No old failures are overwritten or promoted. No LLM is measured and no simulator cost changes.

Q5_0 now preserves dequantized-weight/original-F32 mathematical errors (original 0.05+0.03|ref|) AND independently calculates the source Q8_1 activation path: roundf activation integer, half scale, half original XOR-warp sum, unsigned Q5 dot and -16 sum correction. A bounded path tolerance 1e-4+1e-5|ref| is frozen for K896/1024, N4096, M1/2/4. This budgets final float reduction differences; it is not a universal kernel claim. Q8_0 keeps original mathematical correctness gate. Schema v3 means old results cannot be accepted under new criteria.

source_reference.h is shared with a host-only executable tested against Python and existing numeric samples only. All six regenerated weight/input SHA pairs match, C++ and Python references match exactly, native numeric samples differ <=1.34e-6; no timing is used to set parameters. Old aligned controls have now been inspected and are development/regression evidence, not independent final validation.

The 12 configurations, 1024 calls per batch (changed from 64), first+5 warmup+30 formal, actual backend stream event, common backend synchronize, timing/dispersion/perturbation gates are unchanged. Full raw audit rederives both numeric errors and flags, retains math errors even where path gate passes. quantize.cu/vecdotq.cuh are frozen references; runtime/source build equivalence remains unproven and no cost calibration is emitted.

Build compile.cmd, run host tests and test_source_reference.py, freeze with freeze_build.py; invoke.ps1 without -Run verifies identities only. GPU matrix must await root review and an idle window. Never run source-generation scripts over an existing freeze.
