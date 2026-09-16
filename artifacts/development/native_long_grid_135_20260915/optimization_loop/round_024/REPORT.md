# R24 identity repair result

Fixed native development set; independent acceptance B unvalidated.

131 terminal cells: 130 predicted, 1 failed. Strict all-three-below-10%: 9/131.

|Metric|Median APE %|P90 %|Worst %|Median absolute ms|
|---|---:|---:|---:|---:|
|engine_ttft_ms|42.651|66.967|72.496|88.720|
|engine_tpot_ms|25.143|51.746|64.773|2.574|
|engine_e2e_ms|29.877|54.474|66.189|364.967|

Distributions cover 130 scored cells; the failure remains in the fixed131 denominator. Previous R23 104 numeric predictions all match exactly. Of 27 previous proof failures, 26 now predict; one full-model SHA failure is preserved without retry. R22 common GPU comparison: 24/25 exact, one failed here. Whole-model phase checks before and after returned normally; intermittent SHA failure cause remains unresolved. This restores execution coverage, not a numerical cost improvement.
