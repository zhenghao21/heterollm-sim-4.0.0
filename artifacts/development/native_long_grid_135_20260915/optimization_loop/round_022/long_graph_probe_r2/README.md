# R22 long graph probe r2

R22 r1 was rejected before timing because this locked upstream `ggml_backend_sched_new` asserts that the final scheduler backend is CPU. R2 is a separate source, protocol, build, and collection path; it never overwrites the rejected r1 output.

The scheduler list is `[CUDA, CPU]`: CUDA is first and therefore higher priority; CPU is included only to satisfy the upstream terminal-backend contract. Before graph allocation, r2 explicitly assigns the graph input and every SCALE node to CUDA. After allocation it rejects a non-CUDA graph tensor, a scheduler order mismatch, a split count other than one, or any nonzero CPU scheduler allocation. Thus CPU is not silently used for this accepted synthetic CUDA graph. The final scheduler synchronize visits both registered backends by upstream design, but only CUDA has graph work. A later CUPTI trace remains the required evidence of actual CUDA kernels and stream identity.

The graph remains fixed at 64/256 F32 non-inplace SCALE nodes and 262144 elements. Formal runs remain buffered first1 + warmup5 + formal30 only. `--smoke --config <id> --output <jsonl>` performs one scheduler async compute plus one scheduler synchronize and exact all-stage D2H verification; it deliberately emits no per-call QPC/NVTX timing window and is not a formal measurement or a calibration result. Identity-only still has no GPU access.
