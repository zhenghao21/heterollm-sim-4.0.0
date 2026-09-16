# R30 MMVQ nominal-HBM ablation — preparation only

No actual freeze, source snapshot, native model hash, prediction or GPU run is
performed during preparation. Root must complete host tests and an independent
read-only review, commit these helpers/tests, then pass that full reviewed commit.
The current candidate production files need to be committed before that execution.

## Fixed comparison

Base: `../round_027/on/freeze.json`, fixed131 native selection. Both arms retain
final output selection, issue contracts, retained KV state and all other R27/on
settings. Off means `legacy_mma_output_wave`; on means
`nominal_bandwidth_analytical_fallback`. Only that field may differ in static
inputs. Exact prior config/native/model/hardware/runtime identity remains checked.
This is development evidence; independent B remains unvalidated, even if A passes.

Each arm has115 source files. Four come directly from the supplied reviewed Git
commit: cost_models.py, kernel_query_ledger.py, planner.py, and
predict_stable_native_dataset.py at their normal src/tools paths. Other111 files
are byte-identical R27/on inheritance. Main-worktree WIP is never used to fill
these replacements. No source copy is made until the explicit freeze command.
Both arms must have identical relative source bytes. Snapshot paths and helper
copy paths are checked against their exact roots, not recursively normalized.

## Execution phases after root review and commit

Run from this directory using the project Python interpreter:

```text
python -m pytest . -q -p no:cacheprovider
python freeze_candidate.py --reviewed-commit <full40-char-reviewed-commit>
python run_candidate.py lock
python run_candidate.py full --workers 4 --timeout-seconds 600
python run_candidate.py score
python summarize_groups.py
```

Freeze and lock each invoke separate-process131-cell preflights for both arms,
using only each arm's frozen Python imports and actual resume-start verifier.
Full validates identities before, between and after arms, preserves existing
predicted/failed/incomplete terminals, and passes only missing cell IDs to the
inherited coordinator. No automatic retry of a terminal is allowed. A completed
arm does not even relaunch the coordinator. Both131 terminal sets must be validated
before the262 barrier is written; score requires that existing barrier.

Workers, preflights, coordinators and Git readers use explicit Popen natural
waiting. The600-second setting is a symmetric soft observation deadline, never
a hard wall-time limit. Deadline crossing records late and keeps waiting. Late
exit0 with complete identities remains scoreable; nonzero exit publishes failure
even if the private raw attempt says predicted. Raw results and deadline/exit
receipts remain immutable. No live or unresolved attempt can be resumed by
launching another worker, and the coordinator lock is cleared only after all
started workers exit and their terminals are sealed. Observer interruption stops
new launches while allowing started workers to finish naturally.

Protocol, helper/test commit identities, exact115 source membership, all relevant
native evidence and frozen cell identity are checked at freeze/lock/start/end.
Rejected preparation/phase receipts and partial files are retained. Re-running a
failed freeze cannot adopt partial files. Group reports retain failures in131 and
393 denominators and distinguish three Engine metrics, signed/absolute errors,
median/P90/worst and per-model/deployment groups. No average can replace per-cell
strict<10% or justify a new accuracy guarantee.
