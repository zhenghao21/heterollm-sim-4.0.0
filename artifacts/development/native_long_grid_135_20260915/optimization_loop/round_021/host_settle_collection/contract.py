"""Frozen R21 pilot domain shared by planning and raw auditing; no native calls."""
CONFIG = {'id':'scale_f32_e262144_g8','nodes':8,'elements':262144}
CONDITIONS = [{'id':'short','settle_ms':0},{'id':'settled','settle_ms':1000}]
EXECUTABLE_SHA = '050b89f9f1ecbcc868df15a2c807341540e90da41fcc5d19bf759f027f67b3c1'

def require(ok, message):
    if not ok: raise ValueError(message)

def domain(protocol, manifest):
    execution=protocol['execution']; pilot=execution['pilot']
    require(pilot.get('conditions')==CONDITIONS and execution.get('conditions')==CONDITIONS and manifest.get('conditions')==CONDITIONS, 'protocol/manifest condition domain mismatch')
    require(pilot.get('config')==CONFIG['id'] and execution.get('pilot_config')==CONFIG['id'] and pilot.get('arms')==['buffered'], 'R21 buffered config domain')
    require(pilot.get('pairs_per_arm')==3 and execution.get('pairs_per_condition')==3 and pilot.get('native_processes')==12 and pilot.get('expected_exports')==6 and pilot.get('modes')==['direct','profile'], 'R21 pilot process denominator')
    require([execution.get(k) for k in ('first','warmup','formal','calls_per_process')]==[1,5,30,36], 'R21 recorded call denominator')
    require(manifest.get('status')=='compiled_host_tested_not_gpu_executed' and manifest.get('gpu_access') is False and manifest.get('executable',{}).get('sha256')==EXECUTABLE_SHA, 'immutable r2 compiled manifest required')
    return pilot

def stage_identity(stage, manifest):
    condition=next((x for x in CONDITIONS if x['id']==stage.get('condition')),None)
    require(condition is not None and type(stage.get('settle_ms')) is int and stage['settle_ms']==condition['settle_ms'] and stage.get('arm')=='buffered' and stage.get('config')==CONFIG, 'stage condition/config domain')
    require(type(stage.get('pair')) is int and 1<=stage['pair']<=3 and stage.get('pair_id')==f"r21_host_settle.{stage['condition']}.pair{stage['pair']}", 'stage pair binding')
    expected=[manifest['executable']['path'],'--run','--config',CONFIG['id'],'--arm','buffered','--settle-ms',str(stage['settle_ms']),'--pair-id',stage['pair_id'],'--output',stage['raw']]
    require(stage.get('app_argv')==expected, 'stage argv condition binding')
