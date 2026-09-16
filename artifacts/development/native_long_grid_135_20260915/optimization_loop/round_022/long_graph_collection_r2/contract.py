"""Frozen R22 long-graph domain; pure validation, no native calls."""
CONFIGS=[{'id':f'scale_f32_e262144_g{n}','nodes':n,'elements':262144} for n in (64,256)]
EXECUTABLE_SHA='8cb92bfb4d7056d5636f81980b33ee9931b566b789e204af57b991596b9bd336'
def require(ok,message):
    if not ok:raise ValueError(message)
def domain(protocol,manifest):
    e=protocol['execution'];ids=[x['id'] for x in CONFIGS]
    require(protocol.get('schema')=='long-graph-probe-protocol/v1' and protocol.get('revision')=='r2' and e.get('configs')==ids and manifest.get('configs')==ids,'R22 r2 config/protocol domain mismatch')
    require('Scheduler is ordered [CUDA, CPU]' in protocol.get('controlled_invariants',{}).get('scheduler',''),'R22 r2 scheduler semantics missing')
    require([{k:x[k] for k in ('id','nodes','elements')} for x in protocol['configs']]==CONFIGS,'R22 graph shapes')
    require([e.get(k) for k in ('first','warmup','formal','calls_per_process','pairs_per_config','native_processes','expected_exports')]==[1,5,30,36,3,12,6],'R22 denominators')
    require(e.get('arms')==['buffered'] and e.get('modes')==['direct','profile'] and e.get('settle_loop_present') is False and e.get('settle_ms')==0,'R22 buffered no-settle domain')
    require(manifest.get('status')=='compiled_host_tested_not_gpu_executed' and manifest.get('gpu_access') is False and manifest.get('executable',{}).get('sha256')==EXECUTABLE_SHA,'R22 compiled executable identity')
    return e
def stage_identity(stage,manifest):
    cfg=stage.get('config');require(cfg in CONFIGS and stage.get('condition')==cfg['id'] and stage.get('arm')=='buffered' and 'settle_ms' not in stage,'stage config/no-settle binding')
    require(type(stage.get('pair')) is int and 1<=stage['pair']<=3 and stage.get('pair_id')==f"r22_long_graph.{cfg['id']}.pair{stage['pair']}",'stage pair binding')
    expected=[manifest['executable']['path'],'--run','--config',cfg['id'],'--arm','buffered','--pair-id',stage['pair_id'],'--output',stage['raw']]
    require(stage.get('app_argv')==expected,'stage argv binding')
