"""Independent host runtime costs for explicitly bound CUDA invocations.

Host API timings never replace GPU kernel or DRAM service. Ordinary async
submissions are chained to one another, not to the preceding GPU kernel.
"""
from dataclasses import replace
import math
from typing import Mapping

from .contracts import ResourceDemand, TaskCategory, TaskSpec
from .runtime_residual import PHASES, OPTIONAL_PHASES


def _graph_device_dispatch_config(profile):
    """Read the optional, explicitly owned device replay envelope.

    Host lifecycle measurements and device dispatch measurements have different
    owners.  Keep the device value opt-in on the current
    ``KernelModelProfile`` contract.
    """
    value = profile.graph_device_launch_ns
    evidence = profile.graph_device_launch_evidence
    qualified = profile.graph_device_launch_qualified
    if type(qualified) is not bool:
        raise ValueError('graph device launch qualification must be boolean')
    if value is None:
        if evidence is not None or qualified:
            raise ValueError('graph device launch evidence requires graph_device_launch_ns')
        return None, None, False
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < 0):
        raise ValueError('graph_device_launch_ns must be a finite nonnegative number')
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('graph device launch calibration requires evidence')
    return float(value), evidence, qualified


def _apply_graph_device_dispatch(launches, profile, *, captured):
    """Apply an explicitly calibrated device replay envelope.

    The returned replacements only change frontend service time.  Bytes and
    useful work from all member launch tasks are aggregated onto the
    charged replay owner so graph repricing cannot silently lose physical
    accounting.  With no explicit profile this is a no-op and the ordinary
    per-kernel device model remains authoritative.
    """
    configured, evidence, qualified = _graph_device_dispatch_config(profile)
    audit = {
        'schema': 'cuda-graph-device-dispatch/v1',
        'captured': captured,
        'launch_task_count': len(launches),
        # Keep the ordinary path constant-time.  The aggregate is only
        # needed when an explicit device replay envelope is actually applied.
        'original_device_service_ns': None,
        'configured_graph_device_launch_ns': configured,
        # An explicit replay envelope is only authoritative when it is
        # actually applied to a captured graph.  Ordinary submissions retain
        # the existing per-kernel device profile even if a graph envelope is
        # present on the profile object.
        'cost_source': ('explicit_graph_device_launch_ns'
                        if captured and configured is not None
                        else 'existing_device_profile'),
        'evidence': evidence,
        'prediction_qualified': bool(configured is not None and qualified),
        'applied': False,
    }
    if not captured or configured is None:
        return {}, audit
    if not launches:
        raise ValueError('captured CUDA Graph has no device launch tasks')
    audit['original_device_service_ns'] = sum(
        d.service_ns for task in launches for d in task.demands)
    resources = {d.resource_id for task in launches for d in task.demands}
    if len(resources) != 1 or any(len(task.demands) != 1 for task in launches):
        raise ValueError('graph device dispatch calibration requires one frontend demand per launch')
    first = launches[0]
    total = {
        'bytes_moved': sum(d.bytes_moved for task in launches for d in task.demands),
        'work_units': sum(d.work_units for task in launches for d in task.demands),
    }
    charged = replace(first.demands[0], service_ns=configured, **total)
    replacements = {
        first.task_id: replace(first, demands=(charged,), metadata={
            **first.metadata,
            'cuda_graph_device_dispatch': {**audit, 'applied': True,
                'charged_device_service_ns': configured,
                'cost_owner': first.task_id},
        })}
    for task in launches[1:]:
        replacements[task.task_id] = replace(task, demands=tuple(
            replace(demand, service_ns=0.0, bytes_moved=0, work_units=0.0)
            for demand in task.demands), metadata={
                **task.metadata,
                'cuda_graph_device_dispatch': {**audit, 'applied': True,
                    'charged_device_service_ns': configured,
                    'cost_owner': first.task_id},
            })
    audit.update(applied=True, charged_device_service_ns=configured,
                 cost_owner=first.task_id)
    return replacements, audit


def replace_measured_driver_submission_costs(tasks, profiles, submission_resource_id):
    """Give a measured CUDA API envelope sole ownership of host driver submit.

    This needs the full DAG: cohort driver boundaries are outside captured
    device membership. Keep those boundaries and dependencies, but remove only
    the verified host-only demand for the same device and physical invocation.
    """
    active = {target: profile for target, profile in profiles.items()
              if profile is not None and profile.runtime_calibration is not None}
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
        target = meta.get('cuda_graph_device')
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
                or task.demands[0].bytes_moved != 0
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
        measurement_mode = profile.runtime_measurement_mode
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
            # Copies and cost-free markers can also be independent roots of
            # the captured device body. Every root waits for host submission;
            # descendants retain the original device dependency ordering.
            for task in ordered:
                if (task.metadata.get('phase') != 'kernel_launch'
                        and any(dep in members for dep in task.dependencies)):
                    continue
                replacements[task.task_id] = replace(task,
                    dependencies=tuple(dict.fromkeys((*task.dependencies, last))),
                    metadata={**task.metadata, 'cuda_runtime_cost': compact_audit})
            # Host lifecycle pricing does not imply a device replay saving.
            # Apply one only when a separate device envelope is explicitly
            # supplied; otherwise preserve every analytical per-kernel launch.
            # The normal host-only profile has no device replay envelope.  Do
            # not build a second launch tuple in that hot path; the helper
            # still validates the profile and returns an O(1) no-op audit.
            device_launch_ns = profile.graph_device_launch_ns
            dispatch_inputs = (
                tuple(replacements.get(launch.task_id, launch) for launch in launches)
                if device_launch_ns is not None else launches)
            device_replacements, device_dispatch_audit = _apply_graph_device_dispatch(
                dispatch_inputs, profile, captured=True)
            replacements.update(device_replacements)
        else:
            per_call = costs['ordinary_submit'] / len(launches)
            deps = base
            for ordinal, launch in enumerate(launches):
                last = host_task('ordinary_submit', per_call, deps, ordinal)
                deps = (last,)
                replacements[launch.task_id] = replace(launch,
                    dependencies=tuple(dict.fromkeys((*launch.dependencies, last))),
                    metadata={**launch.metadata, 'cuda_runtime_cost': compact_audit})
            _unused_replacements, device_dispatch_audit = _apply_graph_device_dispatch(
                launches, profile, captured=False)
        # Host calibration alone cannot qualify a captured invocation when an
        # explicitly configured device replay envelope is also part of the
        # prediction.  Keep the existing host-only qualification unchanged
        # when no envelope is configured, because the ordinary per-kernel
        # device profile remains authoritative in that case.
        if (captured
                and device_dispatch_audit['configured_graph_device_launch_ns'] is not None):
            audit['prediction_qualified'] = bool(
                audit['prediction_qualified']
                and device_dispatch_audit['prediction_qualified'])
        compact_audit['prediction_qualified'] = audit['prediction_qualified']
        audit['device_dispatch'] = device_dispatch_audit
        audit['device_dispatch_cost_source'] = device_dispatch_audit['cost_source']
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
