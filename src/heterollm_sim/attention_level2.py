"""Diagnostic Attention Level-2 registry kept separate from generated MMQ rows."""
LEVEL2_ATTENTION_RUNTIME_SHA256 = None
LEVEL2_ATTENTION_RUNTIME_SHA256_PREFIX = '2c002e'
LEVEL2_ATTENTION_SURFACE_STATUS = {
    'prefill': {'kernel_family': 'flash_attention_prefill_l2',
                'mask_domains': ('none', 'causal'), 'kv_formats': ('fp16',),
                'holdout': 'artifacts/development/attention_holdout_20260929/holdout_results.json',
                'stream_k_holdout': 'artifacts/development/attention_streamk_20260929/holdout_results.json',
                'production_qualified': False,
                'reason': 'runtime SHA mismatch and causal M=64 fixup APE 16.03%'},
    'decode': {'kernel_family': 'paged_attention_decode_l2',
               'mask_domains': ('none', 'causal'), 'kv_formats': ('fp16',),
               'holdout': 'artifacts/development/attention_holdout_20260929/holdout_results.json',
               'production_qualified': False,
               'reason': 'no same-runtime paged/page-table trace; retain analytical fallback'},
}

def attention_level2_status(phase, *, mask, kv_format='fp16', stream_k=False):
    phase, mask, kv_format = str(phase).casefold(), str(mask).casefold(), str(kv_format).casefold()
    row = LEVEL2_ATTENTION_SURFACE_STATUS.get(phase)
    if row is None:
        return {'accepted': False, 'reason': 'unknown_attention_phase', 'phase': phase}
    if mask not in row['mask_domains'] or kv_format not in row['kv_formats']:
        return {'accepted': False, 'reason': 'attention_domain_not_calibrated', 'phase': phase,
                'mask': mask, 'kv_format': kv_format, 'stream_k': bool(stream_k)}
    if stream_k and phase == 'decode':
        return {'accepted': False, 'reason': 'stream_k_only_bound_to_prefill_uniform_fixup', 'phase': phase}
    return {**row, 'accepted': bool(row['production_qualified']), 'phase': phase,
            'mask': mask, 'kv_format': kv_format, 'stream_k': bool(stream_k),
            'runtime_sha256': LEVEL2_ATTENTION_RUNTIME_SHA256,
            'runtime_sha256_prefix': LEVEL2_ATTENTION_RUNTIME_SHA256_PREFIX}

__all__ = ['LEVEL2_ATTENTION_RUNTIME_SHA256', 'LEVEL2_ATTENTION_RUNTIME_SHA256_PREFIX',
           'LEVEL2_ATTENTION_SURFACE_STATUS', 'attention_level2_status']
