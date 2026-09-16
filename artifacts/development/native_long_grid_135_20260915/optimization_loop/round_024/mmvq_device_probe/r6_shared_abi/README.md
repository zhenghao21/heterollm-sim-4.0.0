# R6 shared ABI qualification (static only)

R4's convert definition accepts four integer arguments (`type`, `k`, `m`,
`padded_k`); its separately handwritten driver prototype incorrectly accepted
three. C linkage hides parameter types from the linker, so successful symbol
resolution was insufficient.

R6 includes `mmvq_probe_abi.h` from both the generated CUDA definitions and the
link driver. The driver additionally asserts the exact function-pointer types.
All three CUDA translation units and the driver compile and link successfully.
The negative regression deliberately reintroduces R4's declaration and requires
the compiler to reject it (MSVC C2733). R4 evidence is preserved unchanged.

The original 1–1404-line CUDA source prefix remains byte-exact. Static extraction
of the new executable confirms that MMVQ and Q8_1 conversion cubins still match
the locked target DLL: 5,450,408 and 222,816 bytes respectively. See
`device_identity.json` and `source_slice_receipt.json` for full SHA identities.

Reproduction, from this directory, uses the project's Python environment:

```text
python build_shared_abi.py
python -m pytest test_shared_abi.py -q
python verify_device_identity.py
```

No executable was launched. No GPU computation, timing, target-LLM measurement,
or simulator parameter change is part of R6. This closes the shared host ABI
preparation defect only. Conversion correctness, target and wrapper dispatch,
arguments, grid/block, streams, cache state and observation overhead still need
runtime qualification before any performance sample can be admitted.
