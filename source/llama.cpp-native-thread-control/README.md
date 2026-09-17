# Native CPU thread control runtime

This independent runtime copies the annotation-control runtime and replaces only `ggml-cpu.dll`. It rebuilds 16 CPU translation units and their two PCH units from identical copied source using the locked compile commands with `-DGGML_USE_OPENMP` and `-openmp` removed. All optimization and instruction-set flags remain unchanged. It does not compile or relink CUDA, llama, server, or any model.

The retained `LLAMA_TRACE_ANNOTATIONS` switch controls CUDA operator/fusion/phase annotations. Host annotations remain compiled out for both values because `LLAMA_SERVER_HOST_TRACE` is disabled in the inherited server build. Token counters, token timestamp arrays and sampling boundaries are unchanged.

Reproduce from the simulator root with `E:/anaconda/python.exe source/llama.cpp-native-thread-control/build_native_threads.py`. Build commands, readonly source/object/PCH/runtime hashes, logs and before/after CPU DLL identity are in `evidence`. Baseline input preservation, absence of OpenMP compiler flags/imports, startup version, and runtime CPU feature comparison pass. The only feature removed is `OPENMP`; all other feature values remain identical.

This is a software-runtime change, so target model measurements need a new runtime identity and cannot reuse old OpenMP actual as matched truth. No performance measurements were executed by this build task. CPU poll/affinity/priority settings can now be evaluated against the native ggml threadpool path; lower variance remains unproven until those tests run.
