"""Bounded controller/real-NAND GC diagnostic, with formal physical geometry.

Seeds a programmed medium with invalid data rather than replaying a long LLM.
No production monkeypatches: method-identity eligibility checks remain intact.
"""
from __future__ import annotations
import argparse, cProfile, json, pstats, sys, time, subprocess, types
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_hbf_staging_runtime import setup, task
from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
from heterollm_sim.physical_presets import resolve_physical_config
from heterollm_sim.hbf_staging import _BlockPool, _Page
from heterollm_sim.memory_mapping import map_nand_address

def summary(profile):
    stats=pstats.Stats(profile).stats
    rows=[]
    for (file,line,name),(primitive,calls,self_s,cumulative,callers) in stats.items():
        rows.append({'file':file,'line':line,'function':name,'calls':calls,'primitive_calls':primitive,
                     'self_seconds':self_s,'inclusive_seconds':cumulative})
    rows.sort(key=lambda r:-r['inclusive_seconds'])
    interesting=[r for r in rows if 'heterollm_sim' in r['file'] and (
      'hbf_staging' in r['file'] or 'hbf_page_state' in r['file'] or
      'io_execution' in r['file'] or 'nand' in r['file'])]
    return {'total_profile_self_seconds':sum(r['self_seconds'] for r in rows),
            'top_functions':rows[:70],'controller_io_functions':interesting}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=('empty_gc','live_copy_gc','journal_only','snapshot_only'),default='empty_gc')
    parser.add_argument('--writes',type=int,default=32)
    parser.add_argument('--pages',type=int,default=256)
    parser.add_argument('--tag',default='baseline')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--live-pages-per-block',type=int,default=1)
    parser.add_argument('--bind-read-state',action='store_true')
    parser.add_argument('--source-revision',help='read one historical staging module via git show, only in this diagnostic process')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/hbf_gc_profile')
    args=parser.parse_args()
    if args.source_revision:
        source=subprocess.check_output(['git','show',args.source_revision+':src/heterollm_sim/hbf_staging.py'],cwd=ROOT).decode('utf8')
        module=types.ModuleType('heterollm_sim.hbf_staging')
        module.__package__='heterollm_sim'
        module.__file__='git:'+args.source_revision+':src/heterollm_sim/hbf_staging.py'
        sys.modules[module.__name__]=module
        exec(compile(source,module.__file__,'exec'),module.__dict__)
        global _BlockPool,_Page
        _BlockPool,_Page=module._BlockPool,module._Page
    output_dir=args.output_dir.resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    cfg=resolve_physical_config('hbf_ocp_0_7')
    if args.writes < 1 or args.pages < 1 or not 0 < args.live_pages_per_block < cfg.pages_per_block:
        parser.error('writes/pages must be positive and live pages must leave room in a block')
    _,binding=setup()
    runtime=PhysicalRuntimeContext(capture_details=False)
    # One legal scalar write creates and warms the real NAND/host/link contexts.
    resolve_physical_task(task(cfg,binding,'write',count=4096,size=4096,buffer='warmup'),runtime,0.)
    resolve_physical_task(task(cfg,binding,'write',count=2048,size=4096,buffer='warm_host'),runtime,0.)
    resolve_physical_task(task(cfg,binding,'write',offset=2048,count=2048,size=4096,buffer='warm_host'),runtime,0.)
    resolve_physical_task(task(cfg,binding,'write',count=65536,size=65536,buffer='warm_bulk'),runtime,0.)
    ctl=runtime.hbf_staging['flash']['controller']
    for name in ('warmup','warm_host','warm_bulk'):ctl.release(name,0)
    core=runtime.runtimes['flash'].core
    core._nand_publish_numeric(exposed=True)
    setup_start=time.perf_counter()
    ctl._used.clear()
    ctl._used.update({block:cfg.pages_per_block for block in range(ctl._block_count-1)})
    ctl._owners.clear();ctl._active.clear();ctl._free_blocks=_BlockPool(0,0)
    # Full programmed frontiers are physically consistent with the seeded FTL.
    for block in range(ctl._block_count-1):
        address=ctl._address(block,0)
        core._set_program_frontier(core._block_key(map_nand_address(cfg,address)),cfg.pages_per_block)
    if args.case=='live_copy_gc':
        indexes={}
        for block in range(ctl._block_count-1):
            owners={}
            for page in range(args.live_pages_per_block):
                index=block*args.live_pages_per_block+page
                key=('retained',0,index)
                owners[page]=key
                ctl._pages[key]=_Page([(0,4096)],[],None,(block,page),0.)
                indexes[index]=None
            ctl._owners[block]=owners
        ctl._buffer_pages[('retained',0)]=indexes
    setup_s=time.perf_counter()-setup_start
    profile=cProfile.Profile()
    costs=defaultdict(lambda:defaultdict(float))
    completion=0.
    per_write=[]
    before_stats=dict(ctl.stats)
    started=time.perf_counter()
    if args.profile:profile.enable()
    if args.case=='journal_only':
        for n in range(args.writes):
            token=ctl.begin_transaction()
            block=next(iter(ctl._used))
            value=ctl._used.pop(block)
            ctl._used[block]=value
            ctl.commit_transaction(token)
    elif args.case=='snapshot_only':
        for n in range(args.writes):
            token=ctl.snapshot()
            del token
    else:
        for n in range(args.writes):
            t=time.perf_counter()
            if args.bind_read_state:ctl._read_state(('retained',0))
            result=resolve_physical_task(task(cfg,binding,'write',count=args.pages*4096,
                size=args.pages*4096,buffer='rewrite'),runtime,completion)
            completion=result.metadata['physical_completion_ns']
            per_write.append(time.perf_counter()-t)
            for owner,cost in result.metadata['physical_execution_by_owner'].items():
                for key,value in cost.items():
                    if isinstance(value,(int,float)) and not isinstance(value,bool):
                        costs[owner][key]+=value
    if args.profile:profile.disable()
    wall=time.perf_counter()-started
    stats={key:value-before_stats.get(key,0) for key,value in ctl.stats.items()}
    row={'case':args.case,'tag':args.tag,'profiled':args.profile,'source_revision':args.source_revision or 'workspace',
         'scope':'controller-level seeded programmed medium; no LLM or detailed reference run',
         'geometry':{'banks':cfg.array_units,'blocks_per_bank':cfg.blocks_per_plane,
          'pages_per_block':cfg.pages_per_block,'page_bytes':cfg.page_bytes,'physical_blocks':ctl._block_count},
         'seed_setup_seconds_excluded':setup_s,'writes':args.writes,'pages_per_write':args.pages,
         'retained_pages':len(ctl._pages),'live_pages_per_block':args.live_pages_per_block,
         'bind_read_state_per_write':args.bind_read_state,'wall_seconds':wall,'per_write_wall_seconds':per_write,
         'simulated_completion_ns':completion,'stats_delta':stats,
         'physical_costs_sum':{k:dict(v) for k,v in costs.items()},
         'final_metadata_counts':{'used':len(ctl._used),'owners':len(ctl._owners),
           'gc_heap':None if ctl._gc_heap is None else len(ctl._gc_heap),
           'gc_entries':len(ctl._gc_entries)}}
    if args.profile:
        row['profile']=summary(profile)
        profile.dump_stats(str(output_dir/f'{args.tag}_{args.case}_{args.pages}p.prof'))
    dest=output_dir/f'{args.tag}_{args.case}_{args.pages}p.json'
    dest.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in row.items() if k not in {'profile','physical_costs_sum','per_write_wall_seconds'}},ensure_ascii=False))
    if args.profile:
        print(json.dumps(row['profile']['top_functions'][:20],ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
