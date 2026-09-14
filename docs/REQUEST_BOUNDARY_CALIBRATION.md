# Request boundary calibration

Request lifecycle timing is represented by three optional marker actions:
`request_begin`, `first_token`, and `request_end`.  These actions are
separate from CUDA phase launch/synchronization timing because a marker
interval can include host request handling that has no one-to-one CUDA API
row.

A profile may opt into this path only with:

```json
{
  "request_marker_policy": "additive_once_per_request",
  "request_marker_ns": {
    "request_begin": 120.0,
    "first_token": 80.0,
    "request_end": 45.0
  },
  "request_marker_evidence": {
    "request_begin": {"status": "calibrated"},
    "first_token": {"status": "calibrated"},
    "request_end": {"status": "calibrated"}
  },
  "request_shape": {
    "prompt_tokens": 2,
    "output_tokens": 8,
    "prompt_fingerprint": "..."
  }
}
```

The caller must explicitly pass `apply_request_boundary=true` to
`apply_native_calibration` (the CLI flag is
`--apply-request-boundary-calibration`).  The planner then adds at most one
task for each marker that is present in the request: the begin action follows
the arrival marker, the first-token action precedes the first emitted token,
and the end action precedes `request_done`.

The action is fail-closed.  It is skipped when the policy, marker evidence,
coverage status, model/hardware/runtime identity, prompt/output shape, or
prompt fingerprint is absent or mismatched.  No marker value is distributed
over operators, and a phase aggregate cannot be used as a request marker
fallback.  Existing profiles without these fields therefore retain their
previous behavior exactly.
