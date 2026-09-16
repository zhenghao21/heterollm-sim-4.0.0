# R27 paired output-selection evaluation

Development only; fixed native 131 cells. Independent B: unvalidated.

|Group|Arm|Cells|Scored|All three <10%|TTFT median/P90/max %|TPOT median/P90/max %|E2E median/P90/max %|
|---|---|---:|---:|---:|---|---|---|
|qwen25 / explicit_gpu_layers_-1|off|17|17|0|68.186/70.652/72.496|48.226/60.329/61.493|51.262/64.095/65.882|
|qwen25 / explicit_gpu_layers_-1|on|17|17|0|68.388/70.951/72.668|48.159/60.417/61.563|51.475/64.216/66.178|
|qwen35 / explicit_gpu_layers_-1|off|22|22|0|31.244/59.536/59.881|46.749/53.146/54.311|44.227/54.125/54.511|
|qwen35 / explicit_gpu_layers_-1|on|22|22|0|31.226/59.523/59.869|46.709/53.118/54.280|44.194/54.100/54.482|
|qwen38 / explicit_gpu_layers_0|off|20|19|4|8.102/38.228/43.049|21.276/27.110/32.073|11.936/20.602/28.811|
|qwen38 / explicit_gpu_layers_0|on|20|20|5|8.093/38.156/43.048|20.149/26.527/32.073|11.771/20.484/28.810|
|qwen38_gpu / explicit_gpu_layers_66|off|27|26|3|19.838/36.697/37.498|11.541/18.153/23.686|12.809/21.277/30.001|
|qwen38_gpu / explicit_gpu_layers_66|on|27|27|4|19.712/36.653/37.578|11.189/18.156/23.693|12.539/21.263/30.055|
|smollm2 / explicit_gpu_layers_-1|off|23|23|0|51.205/65.683/72.155|19.443/40.231/63.485|29.157/47.475/65.790|
|smollm2 / explicit_gpu_layers_-1|on|23|23|0|51.863/65.894/72.313|19.387/40.386/63.678|29.753/47.823/65.920|
|tinyllama / explicit_gpu_layers_-1|off|22|22|0|55.934/62.548/66.650|36.111/48.557/64.773|39.269/55.435/66.189|
|tinyllama / explicit_gpu_layers_-1|on|22|22|0|56.472/63.087/66.865|36.110/48.708/64.975|39.283/55.780/66.477|

Pairwise outcomes use only common scored metrics; all unscored entries remain in the fixed denominator.
Signed error and absolute milliseconds (median/P90/worst), all failures, and exact paired deltas are in the JSON.
No latency targets were used by this summarizer to select scenes, alter predictions or fit coefficients.
