"""Fixed configuration-level diagnostics. Never emit fitted cost coefficients."""
import statistics
POLICY={
    'config_count':1,'arm_count':1,'condition_count':2,'process_pairs':3,'calls_per_process':36,'formal_calls_per_process':30,
    'pair_order_rule':'config_index_plus_pair_index_parity_direct_first_when_even',
    'checkpoint_seconds':180,'terminate_on_checkpoint':False,
    'kernel_formal_p90_div_p10_max':1.50,'profile_process_median_max_relative_deviation':0.05,
    'profile_direct_host_median_max_relative_difference':0.20,
    'direct_host_formal_p90_div_p10_max':1.50,
    'all_first_warmup_formal_numeric_rows_pass_required':True,
    'all_profile_process_kernel_chain_and_source_path_required':True,
    'clock_sampling_method':'high_resolution_waitable_timer_absolute_QPC_SM_read_windows',
    'high_frequency_fields':['sm_mhz'],'full_state_telemetry_in_formal':False,
    'quantile_method':'linear interpolation at (n-1)*q',
    'median_deviation_center':'median of three profiled process medians',
    'host_ratio_denominator':'paired direct process host median',
    'within_profile_dispersion':'each of three formal kernel-union distributions',
    'target_sm_clock_mhz':2400,'sm_clock_tolerance_mhz':30,'telemetry_period_seconds':0.005,
    'maximum_formal_clock_bracket_gap_ms':25,'every_formal_interval_bracketed_required':True,
    'diagnostic_labels_only':True,'coefficients_emitted':False,'validation_group_never_fitted':True,
}


def quality(config,pairs):
    issues=[]
    if len(pairs)!=3 or {p.get('pair') for p in pairs}!={0,1,2}:issues.append('missing_or_duplicate_process_pair')
    ratios=[];medians=[]
    signatures=[p.get('profile_raw',{}).get('signature') for p in pairs if p.get('profile_raw',{}).get('signature') is not None]
    if len(signatures)==3 and any(s!=signatures[0] for s in signatures[1:]):issues.append('input_or_runtime_identity_changed_across_pairs')
    for pair in pairs:
        label='pair_'+str(pair.get('pair'))
        if pair.get('clock_domain_validated') is not True:issues.append(label+':SM_clock_domain_unvalidated')
        if pair.get('numerics_all_rows') is not True:issues.append(label+':numeric_or_identity_failure')
        if pair.get('trace_chain_complete') is not True:issues.append(label+':unsupported_or_incomplete_trace_chain')
        if pair.get('source_path_matches_observed_family') is not True:issues.append(label+':source_path_dispatch_mismatch')
        if pair.get('trace_warning_free') is not True:issues.append(label+':profiler_warning_or_diagnostic')
        profile=pair.get('profile_kernel',{});direct=pair.get('direct_host',{});host=pair.get('profile_host',{})
        if profile.get('count')!=30 or direct.get('count')!=30 or host.get('count')!=30:issues.append(label+':formal_sample_denominator')
        if profile.get('p90_div_p10',float('inf'))>POLICY['kernel_formal_p90_div_p10_max']:issues.append(label+':kernel_p90_p10')
        if direct.get('p90_div_p10',float('inf'))>POLICY['direct_host_formal_p90_div_p10_max']:issues.append(label+':direct_host_p90_p10')
        if host.get('median_ns',0)>0 and direct.get('median_ns',0)>0:
            ratio=abs(host['median_ns']-direct['median_ns'])/direct['median_ns'];ratios.append(ratio)
            if ratio>POLICY['profile_direct_host_median_max_relative_difference']:issues.append(label+':profile_direct_perturbation')
        else:issues.append(label+':missing_host_median')
        if profile.get('median_ns',0)>0:medians.append(profile['median_ns'])
    deviation=None
    if len(medians)==3:
        center=statistics.median(medians);deviation=max(abs(v-center)/center for v in medians)
        if deviation>POLICY['profile_process_median_max_relative_deviation']:issues.append('across_profile_process_medians')
    else:issues.append('missing_profile_process_medians')
    eligible=not issues
    return {'config':config,'status':'diagnostic_quality_accepted' if eligible else 'diagnostic_quality_rejected',
        'issues':issues,'fixed_policy':POLICY,'pairs_observed':len(pairs),'pairs_required':3,'profile_direct_host_relative_differences':ratios,
        'profile_process_median_max_relative_deviation':deviation,'measurement_cost_eligible':eligible,
        'calibration_eligible':False,'fit_performed':False,'holdout_not_for_fit':True,'experiment_group':'host_settle_diagnostic' ,
        'transfer_scope':'Empirical same-DLL captured kernel path only; source equivalence and LLM transfer remain conditional.',
        'pairs':pairs}
