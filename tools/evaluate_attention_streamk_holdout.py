"""Evaluate a predeclared causal Attention stream-K holdout, never LLM timings."""
import argparse
import hashlib
import json
import sqlite3
import statistics
from pathlib import Path


def stream_k_blocks(context, *, kv_tile_tokens, output_tiles, max_active_blocks):
    """Locked fattn-common.cuh Ada+ stream-K scheduling (dense KV only)."""
    from heterollm_sim.kernel_model import attention_stream_k_schedule
    return attention_stream_k_schedule(context, output_tiles, kv_tile_tokens,
                                       max_active_blocks)['main_blocks']


def read_case(directory, context):
    stem = directory / f'm64_l{context}'
    raw = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8-sig'))
    if not raw['correctness']['passed'] or not raw['modules_stable']:
        raise ValueError('numerical/module gate failed')
    if raw['shape'] != {'queries': 64, 'context': context, 'head_dim': 64, 'query_heads': 4, 'kv_heads': 2}:
        raise ValueError('outside the fixed measured shape domain')
    if raw['mask'] != 'causal' or raw['cache'] != 'hot_same_buffers':
        raise ValueError('incompatible mask/cache protocol')
    with sqlite3.connect(stem.with_suffix('.sqlite')) as conn:
        rows = conn.execute('''select s.value,k.end-k.start,k.gridX,k.gridY,k.gridZ,
            k.blockX,k.blockY,k.blockZ,k.registersPerThread,k.dynamicSharedMemory
            from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName
            order by k.start''').fetchall()
    if len(rows) != 48 or [r[0] for r in rows] != [rows[0][0], rows[1][0]] * 24:
        raise ValueError('expected exactly 24 ordered main/fixup pairs')
    stages = []
    for index in (0, 1):
        group = rows[index::2]
        if len({r[2:] for r in group}) != 1:
            raise ValueError('launch specialization changed during measurement')
        stages.append({'symbol': group[0][0], 'median_ns': statistics.median(r[1] for r in group[4:]),
                       'grid': group[0][2:5], 'block_register_smem': group[0][5:]})
    if 'stream_k_fixup_uniform' not in stages[1]['symbol']:
        raise ValueError('only source-proven uniform fixup supported')
    return raw, stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--holdout', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.holdout / 'protocol.json').read_text(encoding='utf-8-sig'))
    contexts = [*protocol['training_contexts'], protocol['new_holdout_context']]
    cases = [read_case(args.training if i < 2 else args.holdout, l) for i, l in enumerate(contexts)]
    schedules = [stream_k_blocks(l, kv_tile_tokens=protocol['kv_tile_tokens'],
                  output_tiles=protocol['output_tiles'], max_active_blocks=protocol['max_active_blocks']) for l in contexts]
    for (raw, stages), blocks in zip(cases, schedules):
        if raw['loaded_modules'] != cases[0][0]['loaded_modules']:
            raise ValueError('training/holdout module identity differs')
        if stages[0]['grid'] != (blocks, 1, 1):
            raise ValueError('source schedule does not reproduce measured launch')
        if blocks % protocol['output_tiles']:
            raise ValueError('nonuniform partition is outside this model')
        for i, stage in enumerate(stages):
            reference = cases[0][1][i]
            if i == 1 and stage['grid'] != reference['grid']:
                raise ValueError('fixup grid changed')
            if stage['symbol'] != reference['symbol'] or stage['block_register_smem'] != reference['block_register_smem']:
                raise ValueError('training/holdout specialization differs')
    results = []
    for stage in (0, 1):
        coordinate = contexts if stage == 0 else [b // protocol['output_tiles'] - 1 for b in schedules]
        weight = (coordinate[2] - coordinate[0]) / (coordinate[1] - coordinate[0])
        if not 0 <= weight <= 1:
            raise ValueError('holdout outside calibration coordinate range')
        low, high, actual = [case[1][stage]['median_ns'] for case in cases]
        prediction = low + weight * (high - low)
        results.append({'symbol': cases[2][1][stage]['symbol'], 'coordinate': coordinate,
                        'prediction_ns': prediction, 'actual_ns': actual,
                        'ape_percent': 100 * abs(prediction - actual) / actual})
    source = Path(__file__).resolve().parents[1] / 'source/llama.cpp-semantic/ggml/src/ggml-cuda/fattn-common.cuh'
    result = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'stages': results, 'source_predicted_blocks': schedules,
              'accepted_diagnostic_domain': all(r['ape_percent'] <= 10 for r in results),
              'production_qualified': False,
              'scope': 'hot synthetic dense causal Attention, source-derived uniform fixup only'}
    (args.holdout / 'holdout_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
