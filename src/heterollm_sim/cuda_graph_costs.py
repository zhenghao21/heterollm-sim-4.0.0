"""Independent host runtime costs for explicitly bound CUDA invocations.

Host API timings never replace GPU kernel or DRAM service. Ordinary async
submissions are chained to one another, not to the preceding GPU kernel.
"""
from dataclasses import replace
from typing import Mapping

from .contracts import ResourceDemand, TaskCategory, TaskSpec
from .runtime_residual import PHASES, OPTIONAL_PHASES


def replace_measured_driver_submission_costs(tasks, profiles, submission_resource_id):
    """Give a measured CUDA API envelope sole ownership of host driver submit.

    This needs the full DAG: cohort driver boundaries are outside captured
    device membership. Keep those boundaries and dependencies, but remove only
    the verified host-only demand for the same device and physical invocation.
    """
    active = {target: profile for target, profile in profiles.items()
              if profile is not None and getattr(profile, 'runtime_calibration', None) is not None}
    if not active:
        return tuple(tasks)
    task_map = {task.task_id: task for task in tasks}
    if len(task_map) != len(tasks):
        raise ValueError('duplicate task identity in CUDA driver submission ownership')
    owners, host_audits = {}, {}
    for task in tasks:
        meta = task.metadata
        audit = meta.get('cuda_runtime_cost')
        if isinstance(audit, Mapping) and audit.get('schema') == 'cuda-runtime-cost/v1':
            host_audits.setdefault(audit.get('audit_id'), []).append(task)
        if 'cuda_graph_lifecycle_events' not in meta:
            continue
        target = meta.get('cuda_graph_device', meta.get('target_component'))
        if target in active:
            key = (target, meta.get('operator_invocation_group_id'))
            owners.setdefault(key, []).append(task)
    ancestor_cache = {}
    def ancestors_of(owner):
        if owner.task_id not in ancestor_cache:
            ancestors, pending = set(), list(owner.dependencies)
            while pending:
                dependency = pending.pop()
                if dependency not in ancestors:
                    ancestors.add(dependency)
                    if dependency in task_map:
                        pending.extend(task_map[dependency].dependencies)
            ancestor_cache[owner.task_id] = ancestors
        return ancestor_cache[owner.task_id]
    result, replaced = [], set()
    for task in tasks:
        meta = task.metadata
        target = meta.get('target_component')
        if meta.get('event_kind') != 'host_cohort_submit':
            result.append(task)
            continue
        if target not in active:
            if any(task.task_id in ancestors_of(owner)
                   for matches in owners.values() for owner in matches):
                raise ValueError('CUDA driver submission device differs from its measured invocation')
            result.append(task)
            continue
        groups = meta.get('physical_invocation_group_ids', ())
        if (not isinstance(groups, (tuple, list)) or len(groups) != 1
                or not isinstance(groups[0], str) or not groups[0]
                or meta.get('physical_invocation_group_count') != 1
                or meta.get('submission_count') != 1
                or meta.get('runtime_phase') != 'driver_submit'
                or meta.get('cost_owner') != 'driver_submission_queue'
                or meta.get('batching_contract') != 'one_driver_submit_per_physical_invocation_group'):
            raise ValueError('CUDA driver submission requires one explicit matching physical invocation group')
        key = (target, groups[0])
        matches = owners.get(key, ())
        if len(matches) != 1 or key in replaced:
            raise ValueError('CUDA driver submission has no unique measured runtime cost owner')
        owner = matches[0]
        lifecycle = owner.metadata.get('cuda_graph_lifecycle', {})
        audit_id = lifecycle.get('invocation_id')
        hosts = host_audits.get(audit_id, ())
        if len(hosts) != 1:
            raise ValueError('CUDA driver submission lacks its independent host API cost')
        host = hosts[0]
        audit = host.metadata['cuda_runtime_cost']
        if (host.metadata.get('operator_invocation_group_id') != groups[0]
                or not host.metadata.get('phase', '').startswith('cuda_runtime_')
                or host.metadata.get('cuda_graph_lifecycle') != lifecycle
                or audit.get('physical_body_costs_preserved') is not True
                or len(host.demands) != 1
                or host.demands[0].resource_id != active[target].runtime_host_resource_id):
            raise ValueError('CUDA driver submission measured owner identity differs')
        # A matching label alone must not remove an unrelated submission.
        if task.task_id not in ancestors_of(owner):
            raise ValueError('CUDA driver submission is not an input boundary of its measured invocation')
        if (len(task.demands) != 1 or task.demands[0].resource_id != submission_resource_id
                or task.demands[0].bytes_moved != 0 or task.demands[0].energy_pj != 0
                or task.demands[0].work_units != 1):
            raise ValueError('CUDA driver submission demand is not exclusively host driver service')
        result.append(replace(task, demands=(), metadata={**meta,
            'cost_owner': 'cuda_runtime_host_api',
            'cost_owner_replacement': {
                'previous_owner': 'driver_submission_queue',
                'previous_resource_id': task.demands[0].resource_id,
                'removed_service_ns': task.demands[0].service_ns,
                'runtime_audit_id': audit_id,
                'runtime_cost_task_id': host.task_id,
                'runtime_host_resource_id': active[target].runtime_host_resource_id,
                'device_component_id': target,
                'physical_invocation_group_id': groups[0],
                'basis': 'measured_cuda_api_envelope_includes_host_driver_submission'}}))
        replaced.add(key)
    return tuple(result)


def apply_cuda_graph_runtime_costs(tasks, profile):
    """Return a task DAG with separate CPU submission/lifecycle tasks.

    Binding must identify complete device bodies and their external inputs.
    Device enqueue demands remain governed by the device profile; host-only
    measurements cannot justify removing them when CUDA Graphs are enabled.
    """
    calibration = profile.runtime_calibration
    if calibration is None or not profile.runtime_host_resource_id:
        raise ValueError('CUDA runtime pricing requires independent costs and a CPU resource')
    task_map = {task.task_id: task for task in tasks}
    if len(task_map) != len(tasks):
        raise ValueError('duplicate task identity in CUDA runtime pricing')
    owners = [task for task in tasks if 'cuda_graph_lifecycle_events' in task.metadata]
    if not owners:
        raise ValueError('runtime calibration requires explicit CUDA lifecycle bindings')
    replacements, preceding, before = {}, {}, {}
    assigned = set()
    for owner in owners:
        meta = owner.metadata
        ids = tuple(meta.get('cuda_graph_member_task_ids', ()))
        if not ids or len(ids) != len(set(ids)) or not set(ids).issubset(task_map):
            raise ValueError('incomplete CUDA runtime invocation membership')
        if assigned.intersection(ids):
            raise ValueError('overlapping CUDA runtime invocation membership')
        assigned.update(ids)
        members = set(ids)
        ordered = [task for task in tasks if task.task_id in members]
        launches = [task for task in ordered if task.metadata.get('phase') == 'kernel_launch']
        if not launches or launches[0].task_id != owner.task_id:
            raise ValueError('lifecycle cost owner must be the first device launch')
        lifecycle = meta.get('cuda_graph_lifecycle')
        if not isinstance(lifecycle, Mapping):
            raise ValueError('source-derived CUDA lifecycle is missing')
        if lifecycle.get('pricing_ready') is False:
            raise ValueError('CUDA lifecycle contains unresolved executable update behavior')
        if lifecycle.get('evicted_executable_ids'):
            raise ValueError('CUDA graph cache eviction requires per-entry destruction costs')
        events = tuple(meta['cuda_graph_lifecycle_events'])
        if not events or any(event not in PHASES + OPTIONAL_PHASES for event in events):
            raise ValueError('CUDA lifecycle event has no independent measured cost')
        captured = meta.get('cuda_graph_captured') is True
        if captured and not profile.graph_enabled:
            raise ValueError('captured invocation contradicts disabled CUDA Graph profile')
        if captured != (lifecycle.get('decision') == 'graph_launch'):
            raise ValueError('CUDA capture membership contradicts lifecycle decision')
        node_count = meta.get('cuda_graph_node_count')
        topology = meta.get('cuda_graph_topology')
        measurement_mode = getattr(profile, 'runtime_measurement_mode', 'qualified')
        phase_structures = meta.get('cuda_graph_phase_structures')
        if phase_structures is not None and (not isinstance(phase_structures, Mapping) or set(phase_structures) != set(events)):
            raise ValueError('CUDA lifecycle phase structure bindings are incomplete')
        costs, uncertainty = {}, {}
        def identity(descriptor):
            return {key: descriptor[key] for key in ('topology', 'node_count')}
        if measurement_mode == 'experimental_exact_structure':
            if phase_structures is None and any(event in {'update_failure', 'destroy_exec', 'destroy_graph'} for event in events):
                raise ValueError('CUDA destruction/update failure requires explicit old/new structure bindings')
            for event in events:
                descriptor = phase_structures[event] if phase_structures is not None else {'topology': topology, 'node_count': node_count}
                if event == 'update_failure':
                    if 'previous' not in descriptor or 'current' not in descriptor:
                        raise ValueError('failed CUDA update requires old and new structural identities')
                    measurement = calibration.update_pair_measurement(identity(descriptor['previous']), identity(descriptor['current']))
                    costs[event], uncertainty[event] = measurement['median_ns'], measurement
                else:
                    if event == 'update' and 'previous' in descriptor and identity(descriptor['previous']) != identity(descriptor['current']):
                        raise ValueError('successful changed-structure update has no independent measurement')
                    costs[event] = calibration.experimental_costs(descriptor['topology'], descriptor['node_count'], phases=(event,))[event]
                    uncertainty[event] = calibration.uncertainty(descriptor['topology'], descriptor['node_count'], phases=(event,))[event]
        elif measurement_mode == 'qualified':
            for event in events:
                descriptor = phase_structures[event] if phase_structures is not None else {'topology': topology, 'node_count': node_count}
                costs[event] = calibration.costs(descriptor['topology'], descriptor['node_count'], phases=(event,))[event]
        else:
            raise ValueError('unknown CUDA runtime measurement mode')
        if not events or any(event not in costs for event in events):
            raise ValueError('CUDA lifecycle event has no independent measured cost')
        if not captured and events != ('ordinary_submit',):
            raise ValueError('ordinary invocation has invalid CUDA lifecycle events')
        external = tuple(dict.fromkeys(dep for task in ordered for dep in task.dependencies if dep not in members))
        resource = profile.runtime_host_resource_id
        previous_host = preceding.get(resource)
        base = tuple(dict.fromkeys((*external, *((previous_host,) if previous_host else ()))))
        audit = {'schema': 'cuda-runtime-cost/v1', 'topology': topology,
                 'node_count': node_count, 'evidence': calibration.evidence,
                 'qualified': calibration.qualified,
                 'prediction_qualified': calibration.qualified,
                 'runtime_measurement_mode': measurement_mode,
                 'validation_relative_error': calibration.validation_relative_error,
                 'host_device_timings_added_together': False,
                 'physical_body_costs_preserved': True,
                 'device_dispatch_cost_source': 'existing_device_profile',
                 'ordinary_submission_distribution': 'equal_share_of_measured_enqueue_loop'}
        if measurement_mode == 'experimental_exact_structure':
            audit['independent_measurement_uncertainty'] = uncertainty
            audit['independent_measurement_limitations'] = calibration.limitations
            audit['phase_structures'] = phase_structures
        structure_id = meta.get('cuda_graph_structure_id')
        compact_phase_structures = None
        if structure_id is not None:
            if not isinstance(structure_id, str) or not structure_id:
                raise ValueError('CUDA graph structural registry identity is invalid')
            def reference(descriptor):
                if not isinstance(descriptor.get('structure_id'), str) or not descriptor['structure_id']:
                    raise ValueError('CUDA phase structure lacks its run-level registry identity')
                result = {key: descriptor[key] for key in ('structure_id', 'node_count')}
                for role in ('previous', 'current'):
                    if role in descriptor:
                        result[role] = reference(descriptor[role])
                return result
            if phase_structures is not None:
                compact_phase_structures = {event: reference(descriptor) for event, descriptor in phase_structures.items()}
            audit.pop('topology')
            audit['structure_id'] = structure_id
            if 'phase_structures' in audit:
                audit['phase_structures'] = compact_phase_structures
        # The invocation identity survives task renaming when serving instantiates
        # a compiled cohort. Keep the large structural audit once on the first
        # CPU event; every other event points to it without changing scheduling.
        audit_id = lifecycle['invocation_id']
        audit['audit_id'] = audit_id
        compact_audit = {'schema': 'cuda-runtime-cost-ref/v1', 'audit_id': audit_id,
                         'qualified': calibration.qualified,
                         'prediction_qualified': calibration.qualified,
                         'runtime_measurement_mode': measurement_mode}
        host_tasks = []

        def host_task(event, amount, deps, ordinal):
            task_id = f'{owner.task_id}.cuda_host.{ordinal}.{event}'
            if task_id in task_map:
                raise ValueError('CUDA runtime host task identity already exists')
            task = TaskSpec(task_id=task_id, request_id=owner.request_id,
                name=f'{owner.name}.cuda_host.{event}', category=TaskCategory.COMPUTE,
                dependencies=deps, demands=(ResourceDemand(resource, amount),),
                earliest_start_ns=owner.earliest_start_ns, token_index=owner.token_index,
                metadata={**{key: value for key, value in owner.metadata.items()
                             if key.startswith('operator_invocation_group_')
                             or key in ('request_ids', 'execution_phase', 'logical_token_index', 'rank')},
                          'phase': 'cuda_runtime_' + event,
                          'cuda_runtime_cost': audit if ordinal == 0 else compact_audit,
                          'cuda_graph_lifecycle': lifecycle, 'target_component': resource.split('.')[0]})
            host_tasks.append(task)
            return task_id

        if captured:
            deps = base
            for ordinal, event in enumerate(events):
                last = host_task(event, costs[event], deps, ordinal)
                deps = (last,)
            for launch in launches:
                replacements[launch.task_id] = replace(launch,
                    dependencies=tuple(dict.fromkeys((*launch.dependencies, last))),
                    metadata={**launch.metadata, 'cuda_runtime_cost': compact_audit})
        else:
            per_call = costs['ordinary_submit'] / len(launches)
            deps = base
            for ordinal, launch in enumerate(launches):
                last = host_task('ordinary_submit', per_call, deps, ordinal)
                deps = (last,)
                replacements[launch.task_id] = replace(launch,
                    dependencies=tuple(dict.fromkeys((*launch.dependencies, last))),
                    metadata={**launch.metadata, 'cuda_runtime_cost': compact_audit})
        preceding[resource] = last
        if structure_id is not None:
            revised_owner = replacements[owner.task_id]
            owner_metadata = dict(revised_owner.metadata)
            owner_metadata.pop('cuda_graph_topology', None)
            if phase_structures is not None:
                owner_metadata['cuda_graph_phase_structures'] = compact_phase_structures
            replacements[owner.task_id] = replace(revised_owner, metadata=owner_metadata)
        before[owner.task_id] = host_tasks
    unbound = [task.task_id for task in tasks
               if task.metadata.get('phase') == 'kernel_launch' and task.task_id not in assigned]
    if unbound:
        raise ValueError('CUDA runtime launch tasks have no lifecycle owner: ' + unbound[0])
    result = []
    for task in tasks:
        result.extend(before.get(task.task_id, ()))
        result.append(replacements.get(task.task_id, task))
    return tuple(result)
