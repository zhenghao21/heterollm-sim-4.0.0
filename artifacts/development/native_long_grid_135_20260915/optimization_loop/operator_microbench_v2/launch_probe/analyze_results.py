"""Offline fixed-gate diagnostic projection. No CUDA/LLM execution and no fitting."""
from pathlib import Path
import argparse,hashlib,json,math,statistics,sys
METRICS=('host_enqueue_call_total_ns','host_sync_call_total_ns','total_wall_ns','cuda_event_span_ns')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def checked(ref):
    assert digest(ref['path'])==ref['sha256'],'SHA256 changed: '+ref['path']
    return Path(ref['path'])
def quantile(values,p):
    x=sorted(values);position=(len(x)-1)*p;i=int(position);return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(position-i)
def stats(values):
    med=statistics.median(values);mad=statistics.median(abs(v-med) for v in values)
    return {'count':len(values),'median_ns':med,'min_ns':min(values),'max_ns':max(values),'p05_ns':quantile(values,.05),
        'p95_ns':quantile(values,.95),'mad_ns':mad,'mad_over_median':mad/med if med>0 else None,
        'p95_minus_p05_over_median':(quantile(values,.95)-quantile(values,.05))/med if med>0 else None,
        'first_half_vs_second_half_median_relative':abs(statistics.median(values[:len(values)//2])-statistics.median(values[len(values)//2:]))/med if med>0 else None}
def expected_order(protocol):
    configurations=[(k,m,b) for k in protocol['kernels'] for m in protocol['supply_modes'] for b in protocol['burst_lengths']]
    result=[]
    for phase,count in [('warmup',protocol['warmup_repeats_per_configuration']),('formal',protocol['formal_repeats_per_configuration'])]:
        for repeat in range(count):
            for position in range(len(configurations)):
                order=15-position if phase=='formal' and repeat%2 else position
                result.append((phase,repeat,*configurations[(order+repeat*5)%16]))
    return result

def expected_checksum(protocol,kernel,burst):
    geometry=protocol['kernel_geometry'];words=[geometry['untouched_word']]*geometry['output_words']
    if kernel=='low_load_integer':
        for invocation in range(burst):
            salt=(geometry['low_load_seed']+invocation*0x9e3779b9)&0xffffffff
            for lane in range(geometry['threads']):
                value=salt^((lane*0x9e3779b9)&0xffffffff)
                for _ in range(geometry['low_load_integer_iterations']):
                    value=(value*1664525+1013904223)&0xffffffff;value^=value>>13
                words[invocation*geometry['threads']+lane]=value
    value=14695981039346656037
    for word in words:
        for shift in (0,8,16,24):value=((value^((word>>shift)&255))*1099511628211)&0xffffffffffffffff
    return format(value,'016x')

def summarize(raw_rows,protocol):
    headers=[r for r in raw_rows if r.get('type')=='run_identity'];assert len(headers)==1
    header=headers[0];freq=header['qpc_frequency_hz'];assert type(freq) is int and freq>0
    samples=[r for r in raw_rows if r.get('type')=='sample']
    completed=[r for r in raw_rows if r.get('type')=='run_complete']
    assert len(completed)==1 and completed[0]['status']=='complete' and completed[0]['sample_count']==len(samples)
    actual=[(r['phase'],r['repeat'],r['kernel'],r['supply_mode'],r['burst_length']) for r in samples]
    assert actual==expected_order(protocol),'samples do not match predeclared order/count'
    controls=[r for r in raw_rows if r.get('type')=='qpc_control'];assert len(controls)==protocol['measurement']['qpc_control_pairs']
    assert all(r['end']>=r['begin'] for r in controls)
    qpc_median=statistics.median((r['end']-r['begin'])*1e9/freq for r in controls)
    groups={};invalid=[];checksums={(kernel,burst):expected_checksum(protocol,kernel,burst) for kernel in protocol["kernels"] for burst in protocol["burst_lengths"]}
    for ordinal,r in enumerate(samples):
        output=r['output_validation']
        if (not output['passed'] or output['mismatches']!=0 or output['checked_words']!=protocol['kernel_geometry']['output_words'] or output['checksum_fnv1a64']!=checksums[(r['kernel'],r['burst_length'])]):invalid.append([ordinal,'output_check'])
        if any(value!=0 for value in r['cuda_status'].values()) or any(x['cuda_status']!=0 for x in r['synchronize_qpc']):invalid.append([ordinal,'cuda_status'])
        assert r['ordinal']==ordinal
        n=r['burst_length'];enq=r['enqueue_qpc'];sync=r['synchronize_qpc'];assert len(enq)==n and len(sync)==(n+1 if r['supply_mode']=='stream_sync_each' else 1)
        wall=r['wall_qpc'];intervals=[*enq,*[(x['begin'],x['end']) for x in sync],r['start_event_record_qpc'],r['end_event_record_qpc']]
        assert all(wall[0]<=b<=e<=wall[1] for b,e in intervals)
        ordered=sorted((tuple(pair) for pair in intervals));assert all(a[1]<=b[0] for a,b in zip(ordered,ordered[1:])),'named disjoint API intervals overlap'
        compute=lambda pairs:sum((e-b)*1e9/freq for b,e in pairs)
        expected={'host_enqueue_call_total_ns':compute(enq),'host_sync_call_total_ns':compute([(x['begin'],x['end']) for x in sync]),'total_wall_ns':compute([wall])}
        for key,value in expected.items():assert math.isclose(r[key],value,rel_tol=1e-8,abs_tol=.001),'derived host duration disagrees with QPC'
        if not all(isinstance(r[key],(int,float)) and math.isfinite(r[key]) and r[key]>=0 for key in METRICS):invalid.append([ordinal,'nonfinite_or_negative_metric'])
        if r['cuda_event_span_ms'] is not None:assert math.isclose(r['cuda_event_span_ns'],r['cuda_event_span_ms']*1e6,rel_tol=1e-8,abs_tol=.001)
        if r['phase']=='formal':groups.setdefault((r['kernel'],r['supply_mode'],n),[]).append(r)
    gate=protocol['predeclared_variability_gate'];reports=[]
    for (kernel,mode,n),rows in sorted(groups.items()):
        assert len(rows)==gate['required_formal_count'];metrics={};reasons=[]
        for key in METRICS:
            values=[r[key] for r in rows]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in values):metrics[key]={'status':'invalid'};reasons.append(key+':invalid');continue
            record=stats(values);fail=[]
            if record['median_ns']<gate['minimum_positive_median_ns']:fail.append('nonpositive_or_unidentifiable_median')
            for field,limit in [('mad_over_median','mad_over_median_max'),('p95_minus_p05_over_median','p95_minus_p05_over_median_max'),('first_half_vs_second_half_median_relative','first_half_vs_second_half_median_relative_max')]:
                if record[field] is None or record[field]>gate[limit]:fail.append(field)
            record['gate_status']='stable' if not fail else 'unstable';record['failed_gates']=fail
            if key=='cuda_event_span_ns' and record['median_ns']<protocol['measurement']['device_span_identifiability_floor_ns']:
                record['gate_status']='resolution_limited';fail.append('device_event_resolution_floor')
            metrics[key]=record;reasons.extend(key+':'+v for v in fail)
        per_launch=metrics['host_enqueue_call_total_ns'].get('median_ns',0)/n
        observer_ratio=qpc_median/per_launch if per_launch>0 else None
        if observer_ratio is None or observer_ratio>gate['qpc_pair_median_over_per_launch_host_median_max']:reasons.append('host_QPC_observer_limited')
        reports.append({'kernel':kernel,'supply_mode':mode,'burst_length':n,'formal_count':len(rows),
            'status':'stable_diagnostic' if not reasons and not invalid else 'unresolved_diagnostic','reasons':reasons,
            'qpc_pair_over_host_call_ratio':observer_ratio,'metrics':metrics,'transferable_kernel_launch_ns':None})
    return {'schema':'cuda-launch-sync-diagnostic-summary/v1','status':'valid_raw' if not invalid else 'invalid_raw',
        'raw_warmup_count':sum(r['phase']=='warmup' for r in samples),'raw_formal_count':sum(r['phase']=='formal' for r in samples),
        'invalid_samples':invalid,'qpc_pair_median_ns':qpc_median,'configurations':reports,
        'no_outliers_removed':True,'overlapping_intervals_added':False,'device_span_is_pure_kernel_sum':False,
        'model_profile_updated':False,'coefficient_candidate':None,'transfer_validation_required':True,
        'interpretation':protocol['transferability'],'protocol_gate':gate}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-directory',type=Path,required=True);args=parser.parse_args()
    directory=args.run_directory.resolve();receipt=json.loads((directory/'run_receipt.json').read_text());assert receipt['returncode']==0
    protocol=json.loads(checked(receipt['protocol_ref']).read_text(encoding='utf-8'))
    raw=[json.loads(line) for line in checked(receipt['raw_ref']).read_text().splitlines()]
    checked(receipt['binary_ref']);checked(receipt['source_ref']);checked(receipt['build_receipt_ref'])
    summary=summarize(raw,protocol)
    header=next(r for r in raw if r['type']=='run_identity')
    for key,refkey in [('protocol_sha256','protocol_ref'),('source_sha256','source_ref'),('binary_sha256','binary_ref')]:assert header[key]==receipt[refkey]['sha256']
    assert header['gpu_uuid']==protocol['device']['required_gpu_uuid']
    snapshots=[]
    for key in ('device_before_ref','device_after_ref'):
        snap=json.loads(checked(receipt[key]).read_text());assert snap['returncode']==0
        matches=[line.split(',') for line in snap['stdout'].splitlines() if line.split(',')[0].strip()==header['gpu_uuid']]
        assert len(matches)==1 and matches[0][2].strip()==protocol['device']['recorded_driver_version']
        snapshots.append(snap)
    summary['identity']={key:header[key] for key in ('gpu_uuid','gpu_name','compute_capability','sm_count','cuda_driver_api_version','cuda_runtime_version','compiled_cudart_version','compiled_msvc_version','nvcc_version','source_sha256','protocol_sha256','binary_sha256')}
    output=directory/'diagnostic_summary.json'
    with output.open('x',encoding='utf-8') as handle:json.dump(summary,handle,ensure_ascii=False,indent=2);handle.write('\n')
    print(json.dumps({'status':summary['status'],'summary':str(output),'profile_updated':False}));return 0 if summary['status']=='valid_raw' else 2
if __name__=='__main__':sys.exit(main())
