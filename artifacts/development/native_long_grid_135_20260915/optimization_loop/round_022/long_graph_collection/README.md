# R22 long synthetic graph collection

Ready for root-controlled later execution. This directory has no execution freeze or measured run.

- Two configs:262144F32 elements,64 or256 non-inplace SCALE nodes. Odd stages multiply0.5, even stages2; exact binary reference avoids underflow degeneration. Scheduler async compute once and graph-end scheduler synchronize once per graph.
- Buffered first1 + warmup5 + formal30. Numeric all-stage checkpoints afterfirst/warmup/formal; no1000ms settle and no per-call output validation claim.
- Three direct/profile pairs per config,12native +6export stages. Plan alternates config/mode order and preserves the full18-stage denominator. Failed stages are retained, remaining planned stages still execute subject to identity/clock safety; no retries.
- At stage6, `first_pair_checkpoint.json` and available partial summary are written without waiting for a reply. An explicit `stop.json` with `{"freeze_sha256":"<approved SHA>","stop":true}` stops at the next natural stage boundary. There is no300second review timeout or continue.json.
- The collector owns one process at a time, waits for natural exit despite bookkeeping interruption, collects clock brackets, records all failures, and attempts clock reset in finally. Root must wait for R21 commit and all simulator work to become idle before creating execution_freeze or starting the series.
- Thresholds unchanged: formal P90/P10≤1.5; profile/direct host perturbation≤20%; three profile kernel medians relative deviation≤5%; SM2400±30MHz; all formal windows bracketed within25ms. No cost coefficient fit.
- Trace preserves API begin/end/globalTid/correlation, paired kernel start/end/device/stream, all launch APIs and separate graph-end synchronization API boundaries. Actual kernel/API counts and supported/unsupported dispatch family are derived from trace; no one-node-one-kernel substitution. Count differences alone are not fusion proof. Existing launch demand is not considered missing, and wall-minus-kernel is not a CPU latency constant.
- `../long_graph_probe/identity_check.py` is the environment-complete identity-only entry. The frozen invoke.ps1 helper only changes PATH and expects its caller to have already applied protocol environment; its initial environment rejection is retained in failed_identity_01.json. Use the Python entry for self-contained identity checks.

Later root commands: `python collect.py prepare`, then `python collect.py verify --expected-freeze-sha256 <SHA>`, then `python collect.py run --expected-freeze-sha256 <SHA>`. These were not executed during preparation. A finished or stopped series is never resumed in place.
