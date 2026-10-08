"""Prepare GGUF-identical local Graph validation cases without running inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re

from heterollm_sim.config import HostOutputContract, SamplingPolicy, scenario_from_dict
from heterollm_sim.cuda_graph_lifecycle import SOURCE_REVISION
from heterollm_sim.cuda_graph_contract import build_cuda_graph_contract, require_matching_cuda_graph_contract
from heterollm_sim.serde import to_primitive
from heterollm_sim.web import scenario_to_payload
from prepare_frontend_native_cases import prepare


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT.parent / 'models'
CASES = [
    ('qwen3_0_6b_f16', MODELS / 'Qwen3-0.6B-f16.gguf'),
    ('qwen3_1_7b_q8_0', MODELS / 'Qwen3-1.7B-Q8_0.gguf'),
    ('qwen3_4b_q4_k_m', MODELS / 'Qwen3-4B-Q4_K_M.gguf'),
    ('qwen3_8b_q4_k_m', MODELS / 'Qwen3-8B-Q4_K_M.gguf'),
    ('qwen3_8_27b_mixed', Path('C:/Users/A/.lmstudio/models/canhdu/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS-GGUF/Qwen3.8-27B-IQ3_S-FFN-IQ4_XS.gguf')),
]


def apply_configured_output_contract(scenario, source_revision):
    """Bind the pinned server's explicit CPU output/sampling configuration.

    This declaration precedes native measurement; later paired validation must
    still check the server's observed settings against it.
    """
    if source_revision != SOURCE_REVISION:
        raise ValueError("configured output contract requires the pinned native source revision")
    contract = HostOutputContract("hostmem0", scenario.model.vocabulary_size, "fp32", 4)
    sampling = SamplingPolicy("greedy", temperature=0.0, implementation="llama_cpp_cpu_chain",
        top_k=40, top_p=0.95, min_p=0.05, min_keep=0)
    return replace(scenario, host_output_contract=contract, sampling_policy=sampling,
        workload=replace(scenario.workload, metadata={**scenario.workload.metadata,
            "native_output_contract_evidence": {
                "source_revision": source_revision,
                "configuration_source": "common/common.h common_params_sampling defaults; request temperature=0",
                "logits_source": "llama-context.cpp: n_outputs*n_vocab*sizeof(float) asynchronous host copy",
                "sampling_source": "common/sampling.cpp: ordered CPU sampler chain",
                "native_settings_record": None, "measured_latency_used": False,
                "settings_observed": False,
                "remaining_partial_costs": ["completion interrupt latency", "sampler heap repair/sort/branch/cache/compiler behavior",
                    "disabled sampler and EOS logit-bias overhead", "GET_ROWS F16 conversion compute"],
            }}))


def attach_graph_experiment(output, slug, fragment_path, costs_path, *, base_scenario=None):
    """Build matched on/off inputs from explicit source-compiled structure."""
    if not isinstance(slug, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", slug) is None:
        raise ValueError("case ID must contain only lowercase letters, digits, underscores or hyphens")
    from heterollm_sim.runtime_residual import load_runtime_structure_measurements
    from heterollm_sim.serde import to_primitive
    fragment = json.loads(fragment_path.read_text(encoding='utf-8'))
    costs = load_runtime_structure_measurements(costs_path)
    raw_costs = to_primitive(costs)
    base_path = base_scenario if base_scenario is not None else output / f'scenario_{slug}_512_128.json'
    base = json.loads(base_path.read_text(encoding='utf-8'))
    for mode in ('off', 'on'):
        payload = deepcopy(base)
        payload['name'] = f'frontend-cuda-graph/{slug}/{mode}/512-128'
        workload = payload['workload']
        request_template = workload['requests'][0]
        workload['requests'] = [{**deepcopy(request_template), **request}
                                for request in fragment['requests']]
        workload['request_count'] = len(workload['requests'])
        workload['prompt_tokens'] = 0
        workload['output_tokens'] = 0
        program = deepcopy(fragment['program'])
        program['graph_enabled'] = mode == 'on'
        workload['metadata'].update(
            cuda_graph_structural_program=program,
            cuda_graph_comparison_request_id=fragment['comparison_request_id'],
            native_ctx_checkpoints=0,
            cuda_graph_experiment={
                'mode': 'experimental_exact_structure',
                'qualified_for_generalization': False,
                'target_llm_latency_used': False,
                'cost_evidence': str(costs_path.resolve()),
                'native_checkpoint_policy': 'explicitly_disabled_for_controlled_graph_comparison',
            })
        workload['scheduler']['max_num_seqs'] = 1
        # hardware_input is the authoring authority; scenario_to_payload below
        # regenerates the redundant runtime hardware/profile sections.
        for component in payload['hardware_input']['hardware']['components']:
            if component['kind'] != 'gpu':
                continue
            kernel = component['execution_profile']['parameters']['kernel_model']
            kernel.update(graph_enabled=mode == 'on', runtime_calibration=raw_costs,
                          runtime_host_resource_id='cpu0.cuda_submission',
                          runtime_measurement_mode='experimental_exact_structure',
                          runtime_submission_ns=None, graph_launch_ns=None)
            identities = ('hardware_id', 'runtime_id', 'architecture')
            kernel['runtime_calibration_binding'] = {
                **{'kernel_' + key: kernel[key] for key in identities},
                **{'measured_' + key: getattr(costs, key) for key in identities},
            }
        path = output / f'scenario_{slug}_graph_{mode}.json'
        scenario = scenario_from_dict(payload)
        require_matching_cuda_graph_contract(build_cuda_graph_contract(scenario), program.get('contract'),
                                             source='fragment attachment')
        prepared = scenario_to_payload(scenario)
        path.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/cuda_graph_validation_2026-10-08')
    parser.add_argument('--source-root', type=Path, default=Path('F:/codex_project/_runtime_sources/llama.cpp'))
    parser.add_argument('--server', type=Path)
    parser.add_argument('--attach-case', help='case ID of an existing scenario_<ID>_512_128.json')
    parser.add_argument('--base-scenario', type=Path, help='explicit frontend-exported base for --attach-case')
    parser.add_argument('--program-fragment', type=Path)
    parser.add_argument('--experimental-costs', type=Path)
    args = parser.parse_args()
    if args.attach_case:
        if args.program_fragment is None or args.experimental_costs is None:
            parser.error('--attach-case requires --program-fragment and --experimental-costs')
        attach_graph_experiment(args.output, args.attach_case, args.program_fragment, args.experimental_costs,
                                base_scenario=args.base_scenario)
        return
    server = args.server or args.source_root / 'build-native-5080-sm120/bin/llama-server.exe'
    preparation = prepare(args.output, args.source_root, server, CASES)
    for case in preparation['cases']:
        slug = case['case_id']
        path = Path(case['scenario_path'])
        scenario = scenario_from_dict(json.loads(path.read_text(encoding='utf-8')))
        native = args.output / f'native_{slug}_graph_off.json'
        scenario = apply_configured_output_contract(scenario, preparation['source_commit'])
        payload = scenario_to_payload(scenario)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        case['native_output_path'] = str(native.resolve())
        case['native_command'][case['native_command'].index('--output') + 1] = str(native.resolve())
        case['graph_mode'] = 'off'
        case['native_command'].extend(['--cuda-graphs', 'off', '--ctx-checkpoints', '0'])
        case['scenario_configuration'].update(
            host_output=to_primitive(scenario.host_output_contract), sampling=to_primitive(scenario.sampling_policy))
        preparation['methodology']['native_output_contract'] = 'explicit pinned-source CPU output/sampler configuration; native settings and latency not read'
    (args.output / 'preparation.json').write_text(
        json.dumps(preparation, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(args.output / 'preparation.json')


if __name__ == '__main__':
    main()
