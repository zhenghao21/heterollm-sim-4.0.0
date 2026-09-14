import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEBUI = ROOT / "src" / "heterollm_sim" / "webui"


class UiV4StaticContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (WEBUI / "index.html").read_text(encoding="utf-8")
        cls.app = (WEBUI / "app.js").read_text(encoding="utf-8")
        cls.css = (WEBUI / "styles.css").read_text(encoding="utf-8")
        cls.preload = (WEBUI / "ui-settings-preload.js").read_text(encoding="utf-8")
        cls.topology_core = (WEBUI / "topology-core.js").read_text(encoding="utf-8")

    def test_strict_csp_uses_external_scripts_only(self):
        scripts = re.findall(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", self.html, re.DOTALL)
        self.assertEqual(len(scripts), 6)
        for attrs, body in scripts:
            self.assertRegex(attrs, r"\bsrc=([\"']).+?\1")
            self.assertEqual(body.strip(), "")
        self.assertIn('src="./ui-settings-preload.js"', self.html)
        self.assertIn('src="./topology-core.js" defer', self.html)
        self.assertIn('src="./model-graph-core.js" defer', self.html)
        self.assertIn('src="./trace-view-core.js" defer', self.html)
        self.assertIn('src="./ui-i18n.js" defer', self.html)
        self.assertIn('src="./app.js" defer', self.html)
        self.assertLess(self.html.index('src="./ui-i18n.js" defer'), self.html.index('src="./app.js" defer'))

    def test_six_theme_whitelists_and_semantic_palettes_stay_in_sync(self):
        themes = ("graphite", "bluegray", "black", "ivory", "mist", "softgray")
        for theme in themes:
            with self.subTest(theme=theme):
                self.assertIn(f'value="{theme}"', self.html)
                self.assertIn(f'"{theme}"', self.app)
                self.assertIn(f'"{theme}"', self.preload)
                if theme != "graphite":
                    self.assertIn(f':root[data-theme="{theme}"]', self.css)
        self.assertIn('content="dark light"', self.html)
        self.assertIn("root.style.colorScheme", self.app)
        self.assertIn("root.style.colorScheme", self.preload)

    def test_model_preset_dialog_and_api_contract(self):
        for element_id in (
            "modelPresetsDialog",
            "presetSearchInput",
            "presetFamilyFilter",
            "presetArchitectureFilter",
            "presetSupportFilter",
            "presetList",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn('apiRequest("/model-presets"', self.app)
        self.assertIn(
            'if (!Array.isArray(payload?.items)) throw new Error("V4 模型预设 API 必须返回分页 items。")',
            self.app,
        )
        self.assertIn("const items = payload.items", self.app)
        self.assertIn('`/model-presets/${encodeURIComponent(id)}`', self.app)
        self.assertIn('state.scenario.model = nextModel', self.app)
        self.assertIn('state.scenario.placement.model_name = nextModel.name', self.app)
        self.assertIn(
            'normalizeV4ModelAuthoring(nextModel, nextModel.graph || payload?.graph)',
            self.app,
        )
        self.assertIn('V4 model.graph 是必填的唯一执行定义', self.app)
        self.assertIn('function renderControlPlaneStatus()', self.app)
        self.assertIn('control_plane', self.app)
        self.assertNotIn('isLayerScopedMappingKey', self.app)
        self.assertIn('clearParallelLayerToStage(state.scenario.placement)', self.app)
        self.assertIn('parallel.layer_to_stage = {}', self.app)
        self.assertIn('rank_mapping', self.app)
        self.assertIn('collective_algorithm', self.app)
        self.assertNotIn("nextLayerIds", self.app)
        self.assertIn("即使新模型复用同名层 ID", self.html)
        self.assertIn("模型专属的层到阶段映射", self.html)
        self.assertIn("TP/PP/EP 度数、Rank 映射（Rank Mapping）、集合通信算法（Collective Algorithm）", self.html)
        self.assertIn("模型权重（Model Weights）、KV 缓存（KV Cache）等通用映射保持不变", self.html)
        self.assertIn("支持显式 requests 或有效 synthetic workload", self.html)
        self.assertIn("analytical_approximation", self.html)
        self.assertIn("out_of_domain", self.html)
        self.assertRegex(self.app, r"support-out_of_domain|out_of_domain")

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the UI behavior contract")
    def test_preset_cleanup_clears_model_specific_pp_segments_but_preserves_parallel_strategy(self):
        helper_start = self.app.index("function clearParallelLayerToStage")
        helper_open = self.app.index("{", helper_start)
        depth = 0
        helper_end = None
        for index in range(helper_open, len(self.app)):
            if self.app[index] == "{":
                depth += 1
            elif self.app[index] == "}":
                depth -= 1
                if depth == 0:
                    helper_end = index + 1
                    break
        self.assertIsNotNone(helper_end)
        helper_source = self.app[helper_start:helper_end]
        script = r"""
const asObject = (value) => value && typeof value === "object" && !Array.isArray(value) ? value : {};
const placement = {
  parallel: {
    tp_degree: 2,
    pp_degree: 2,
    ep_degree: 4,
    rank_mapping: [{rank: 0, tp_rank: 0, pp_rank: 0, ep_rank: 0}],
    collective_algorithm: "ring",
    routing_policy: "lowest_latency",
    layer_to_stage: {"layer-000": 0, "layer-001": 1},
  },
};
eval(process.argv[1]);
const before = JSON.stringify(placement.parallel);
const result = clearParallelLayerToStage(placement);
const after = placement.parallel;
if (result.removed !== 2 || result.preserved !== 0 || Object.keys(after.layer_to_stage).length !== 0
    || after.tp_degree !== 2 || after.pp_degree !== 2 || after.ep_degree !== 4
    || after.collective_algorithm !== "ring" || after.routing_policy !== "lowest_latency"
    || JSON.stringify(after.rank_mapping) !== JSON.stringify([{rank: 0, tp_rank: 0, pp_rank: 0, ep_rank: 0}])) {
  process.stderr.write(JSON.stringify({before, result, after}));
  process.exit(1);
}
"""
        completed = subprocess.run(
            ["node", "-e", script, helper_source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the GPU profile behavior contract")
    def test_dense_gpu_throughput_round_trip_derives_only_mma_cycles(self):
        helper_sources = []
        for helper_name in ("denseGpuThroughputTops", "denseGpuThroughputUnit", "denseGpuCyclesForThroughput"):
            helper_start = self.app.index(f"function {helper_name}")
            helper_open = self.app.index("{", helper_start)
            depth = 0
            helper_end = None
            for index in range(helper_open, len(self.app)):
                if self.app[index] == "{":
                    depth += 1
                elif self.app[index] == "}":
                    depth -= 1
                    if depth == 0:
                        helper_end = index + 1
                        break
            self.assertIsNotNone(helper_end)
            helper_sources.append(self.app[helper_start:helper_end])
        script = r"""
eval(process.argv[1]);
const profile = {
  default_tensor_dtype: "bf16",
  tensor_core: {
    sm_count: 84,
    tensor_cores_per_sm: 4,
    frequency_ghz: 2.617,
    mma_m: 16,
    mma_n: 16,
    mma_k: 16,
    cycles_per_mma: 32,
    supported_dtypes: ["bf16"],
    dtype_throughput_scale: {bf16: 0.5},
  },
};
const before = JSON.stringify(profile);
const baseline = denseGpuThroughputTops(profile, "bf16");
const recoveredCycles = denseGpuCyclesForThroughput(profile, baseline, "bf16");
const target = 112.6;
const targetCycles = denseGpuCyclesForThroughput(profile, target, "bf16");
const targetProfile = JSON.parse(JSON.stringify(profile));
targetProfile.tensor_core.cycles_per_mma = targetCycles;
const roundTrip = denseGpuThroughputTops(targetProfile, "bf16");
if (!Number.isFinite(baseline) || !Number.isFinite(recoveredCycles)
    || Math.abs(recoveredCycles - profile.tensor_core.cycles_per_mma) > 1e-10
    || !Number.isFinite(targetCycles) || !Number.isFinite(roundTrip)
    || Math.abs(roundTrip - target) > 1e-10
    || denseGpuThroughputUnit("bf16") !== "TFLOPS"
    || denseGpuThroughputUnit("int8") !== "TOPS"
    || denseGpuCyclesForThroughput(profile, 0, "bf16") !== null
    || denseGpuCyclesForThroughput(profile, Number.NaN, "bf16") !== null
    || denseGpuThroughputTops(profile, "missing") !== null
    || JSON.stringify(profile) !== before) {
  process.stderr.write(JSON.stringify({baseline, recoveredCycles, targetCycles, roundTrip, profile}));
  process.exit(1);
}
"""
        completed = subprocess.run(
            ["node", "-e", script, "\n".join(helper_sources)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_control_plane_status_is_read_only(self):
        self.assertIn("function renderControlPlaneStatus()", self.app)
        self.assertIn("function controlPlaneStatusView(", self.app)
        self.assertNotIn("function runAutoMapping(", self.app)
        self.assertNotIn("/api/auto-map", self.app)

    def test_group_count_summary_refreshes_after_group_mutations(self):
        self.assertIn("const groups = asArray(state.topologyView?.groups).length;", self.app)
        self.assertIn("`${components} 组件 · ${links} 链路 · ${groups} 组`", self.app)
        self.assertIn("selectTopologyGroup(result.group.group_id);\n    renderAll();", self.app)
        self.assertIn("setComponentSelection(group.members, group.root);\n  renderAll();", self.app)

    def test_runtime_summary_exposes_kv_reads_writes_and_modeling_limits(self):
        for label in (
            "KV 读写流量（KV Traffic）",
            "预填读取（Prefill Read）",
            "预填写入（Prefill Write）",
            "解码读取（Decode Read）",
            "解码追加（Decode Append）",
            "KV 迁移与限制（KV Movement）",
            "预取距离（Prefetch Distance）",
        ):
            self.assertIn(label, self.app)
        for key in (
            "logical_prefill_read_bytes",
            "physical_prefill_write_bytes",
            "logical_decode_read_bytes",
            "physical_decode_append_bytes",
            "migration_events",
            "prefetch_distance_modeled",
            "max_live_tokens_per_request",
        ):
            self.assertIn(key, self.app)
        self.assertIn("未显式建模（仅策略元数据）", self.app)
        self.assertIn(".runtime-value-label", self.css)

    def test_representative_parameter_labels_are_chinese_first_bilingual(self):
        labels = (
            "隐藏维度（Hidden Size）",
            "张量并行度（TP Degree）",
            'uiText("缓存组件", "Cache Component")',
            "首 Token 延迟（TTFT）",
            "读取延迟（Read Latency, ns）",
            "传输粒度（Transfer Granularity, B/KiB…PiB）",
            "DMA 延迟（DMA Latency, ns）",
            "请求 ID（Request ID）",
            "目标端点（Target）",
        )
        combined = self.html + self.app
        for label in labels:
            with self.subTest(label=label):
                self.assertIn(label, combined)
        self.assertNotIn("TBT p_50", combined)
        self.assertNotIn("TBT p^50", combined)

    def test_hbf_ssd_kinds_defaults_and_connection_rules(self):
        kinds = ("gpu", "hbm", "hbf", "ssd", "high_io_ssd", "digital_sram_cim")
        for kind in kinds:
            self.assertIn(f'data-add-kind="{kind}"', self.html)
            self.assertIn(f'"{kind}"', self.app)
        self.assertIn("512 * 1024 ** 3", self.app)
        self.assertIn("read_bandwidth_gbps: 24000", self.app)
        self.assertIn('source: "official-reference-upper-bound"', self.app)
        self.assertIn('hasHbf && protocol !== "UCIe"', self.app)
        self.assertIn('hasSsd && !["PCIe", "CXL"].includes(protocol)', self.app)
        self.assertIn('["hbm", "hbm_stack"].includes', self.app)
        self.assertNotIn('["hbm", "hbf", "hbm_stack"]', self.app)
        for metadata_key in ("read_latency_ns", "write_latency_ns", "transfer_granularity_bytes", "dma_latency_ns"):
            self.assertIn(metadata_key, self.app)
        self.assertIn("高 I/O 固态硬盘（High-I/O SSD）", self.app + self.html)
        self.assertNotIn("高 IO 固态硬盘", self.app + self.html)

    def test_model_weights_backing_status_uses_control_plane_evidence(self):
        self.assertIn("model_weights_backing", self.app)
        self.assertIn("model_weight_capacity", self.app)
        self.assertIn("placement.metadata.control_plane.decision.rank_weight_shards", self.app)
        self.assertIn("weight_tensor_details", self.app)
        self.assertNotIn("placement.tensor_to_component.model_weights", self.app)
        self.assertNotIn("placement.tensor_bytes.model_weights", self.app)
        self.assertIn("storage_component_id", self.app)
        self.assertIn("physical_bytes", self.app)

    def test_web_normalizer_targets_v4_and_rejects_explicit_old_schema(self):
        self.assertIn("function requireScenarioSchemaV4", self.app)
        self.assertIn('value.schema_version !== AUTHORING_SCHEMA_VERSION', self.app)
        self.assertIn("schema_version 必须严格等于 ${AUTHORING_SCHEMA_VERSION}", self.app)
        self.assertNotIn('??= "0.5"', self.app)
        self.assertNotIn('??= "0.2"', self.app)

    def test_topology_v04_collision_clipboard_and_route_failure_contracts(self):
        for helper in (
            "resolveCollisionPlacement",
            "validateClipboardPayload",
            "routeOrthogonal",
            "planOrthogonalRoutes",
            "normalizeTopologyView",
        ):
            self.assertIn(f"function {helper}", self.topology_core)
        self.assertIn("routePlan = Topology.planOrthogonalRoutes", self.app)
        self.assertIn("topologyStoredRouteSegments(element)", self.app)
        self.assertIn("topologyRouteErrors", self.app)
        self.assertIn('id="topologyRouteStatus"', self.html)
        self.assertIn("resolveTopologyPlacement(state.nodePositions, desired)", self.app)

    def test_catalog_pagination_ignores_stale_requests_and_restores_controls(self):
        self.assertIn("new AbortController()", self.app)
        self.assertIn("++presetCatalogGeneration", self.app)
        self.assertIn("generation !== presetCatalogGeneration", self.app)
        self.assertIn("presetCatalogController?.abort()", self.app)
        self.assertIn("syncPresetPagination()", self.app)
        self.assertIn("state.modelPresetsLoading || offset <= 0", self.app)
        self.assertIn("state.modelPresetsLoading || nextOffset == null", self.app)

    def test_new_topology_and_catalog_actions_are_bilingual(self):
        for label in (
            "选择（Select）",
            "建立组（Group）",
            "粘贴（Paste）",
            "整理布局（Layout）",
            "本地目录（Local Catalog）",
            "上一页（Previous）",
            "导入到本地目录（Import）",
        ):
            with self.subTest(label=label):
                self.assertIn(label, self.html)
        self.assertNotIn("平移（Pan）", self.html)

    def test_font_scale_200_percent_regression_contract(self):
        self.assertRegex(self.html, r'id="fontScaleInput"[^>]+min="80"[^>]+max="200"')
        self.assertRegex(self.html, r'id="fontScaleNumberInput"[^>]+min="80"[^>]+max="200"')
        self.assertIn("const MAX_FONT_SCALE = 200", self.app)
        self.assertIn("Math.min(200, Math.max(80", self.preload)
        self.assertIn(':root[data-font-band="extreme"] .topbar', self.css)
        self.assertIn(':root[data-font-band="extreme"] .architecture-grid', self.css)
        self.assertIn(':root[data-font-band="extreme"] .metric-grid', self.css)
        self.assertRegex(self.css, r':root\[data-font-band="extreme"\] \.topbar\s*\{[^}]*overflow:\s*auto', re.DOTALL)
        self.assertRegex(self.css, r'\.preset-dialog-shell\s*\{[^}]*max-height:[^}]*100dvh[^}]*overflow:\s*auto', re.DOTALL)
        self.assertRegex(self.css, r':root\[data-font-band="extreme"\] \.model-presets-dialog\s*\{[^}]*100dvh[^}]*overflow:\s*auto', re.DOTALL)
        self.assertRegex(self.css, r':root\[data-font-band="extreme"\] \.preset-dialog-shell\s*\{[^}]*display:\s*block[^}]*overflow-y:\s*auto', re.DOTALL)
        self.assertRegex(self.css, r':root\[data-font-band="extreme"\] \.preset-dialog-shell > \.dialog-head\s*\{[^}]*position:\s*static', re.DOTALL)
        self.assertRegex(self.css, r':root\[data-font-band="extreme"\] \.hardware-preset-shell\s*\{[^}]*display:\s*block[^}]*overflow-y:\s*auto', re.DOTALL)
        self.assertRegex(self.css, r':root\[data-font-band="extreme"\] \.hardware-preset-panel\s*\{[^}]*height:\s*auto[^}]*overflow:\s*visible', re.DOTALL)
        self.assertIn(':root[data-font-band="extreme"] .preset-filters', self.css)
        self.assertIn(':root[data-font-band="extreme"] .preset-list', self.css)

    def test_non_result_number_formatting_uses_significant_digits_and_scientific_notation(self):
        self.assertIn('function formatNumber(value, significantDigits = 4)', self.app)
        self.assertIn('maximumSignificantDigits: precision', self.app)
        self.assertIn('absolute >= 1_000_000 || (absolute > 0 && absolute < 0.001)', self.app)
        self.assertIn('number.toExponential(precision - 1)', self.app)
        self.assertIn('<sup>${exponent}</sup>', self.app)

    def test_layout_and_motion_settings_are_chinese_first_bilingual(self):
        for label in (
            "紧凑布局（Compact Layout）",
            "拓扑网格（Topology Grid）",
            "减少动画（Reduce Motion）",
        ):
            with self.subTest(label=label):
                self.assertIn(label, self.html)

    def test_rank_mapping_is_grouped_filterable_paginated_and_read_only(self):
        for element_id in (
            "effectiveMappingSummary",
            "effectiveMappingSearchInput",
            "effectiveMappingRankFilter",
            "effectiveMappingComponentFilter",
            "effectiveOpPreviousButton",
            "effectiveOpNextButton",
            "effectiveTensorPreviousButton",
            "effectiveTensorNextButton",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn("function groupEffectiveMappingRows", self.app)
        self.assertIn("function effectiveMappingPage", self.app)
        self.assertIn("条逐 Rank 数据", self.app)
        self.assertIn("mapping-layout-readonly", self.html)
        self.assertIn("controlPlaneStatus", self.html)
        self.assertNotIn("CONTROL_PLANE_LOCK_FIELDS", self.app)
        self.assertNotIn("function setMappingLocked", self.app)

    def test_hardware_preset_library_has_one_entry_and_two_independent_tabs(self):
        self.assertEqual(self.html.count('id="hardwarePresetsButton"'), 1)
        self.assertNotIn('id="architecturePresetsButton"', self.html)
        self.assertNotIn('id="componentPresetsButton"', self.html)
        for element_id in (
            "hardwarePresetsDialog",
            "hardwarePresetComponentTab",
            "hardwarePresetArchitectureTab",
            "componentPresetsDialog",
            "architecturePresetsDialog",
        ):
            self.assertIn(f'id="{element_id}"', self.html)
        self.assertIn('apiRequest("/component-presets"', self.app)
        self.assertIn('apiRequest("/architecture-presets"', self.app)
        self.assertIn("hardwarePresetScroll", self.app)
        self.assertIn("组件预设 · 追加", self.html)
        self.assertIn("架构预设 · 替换", self.html)
        self.assertIn("function architecturePresetHardwareMarkup", self.app)
        self.assertIn("仿真组件与物理组成（Simulator Components &amp; Physical Composition）", self.app)
        self.assertIn("OPS 口径（OPS Basis）", self.app)
        self.assertIn("architecture-component-summary-grid", self.css)

    def test_viewport_help_and_kind_adaptive_inspector_contract(self):
        self.assertIn("const CONCEPT_HELP = Object.freeze", self.app)
        self.assertIn("function positionFieldHelp", self.app)
        self.assertRegex(self.css, r"\.field-help-viewport-popover\s*\{[^}]*position:\s*fixed[^}]*z-index:\s*2000", re.DOTALL)
        self.assertRegex(self.css, r"\.field-help-trigger\s*\{[^}]*cursor:\s*help", re.DOTALL)
        self.assertRegex(self.css, r"\.field-help-trigger\s*\{[^}]*text-decoration:\s*none", re.DOTALL)
        self.assertNotIn("text-decoration-style: dotted", self.css)
        for kind in ("cpu", "fabric_switch", "io_die"):
            self.assertIn(f'"{kind}"', self.app)
        self.assertIn("function componentInspectorProfile", self.app)
        self.assertIn('data-inspector-port-field="bandwidth_gbps"', self.app)
        self.assertNotIn("当前 Kind 的隐藏字段", self.app)
        self.assertIn("物理容量与传输能力", self.app)
        self.assertIn("Profile 是执行/内存成本权威", self.app)
        self.assertIn("CPU V4 执行成本 Profile", self.app)
        self.assertNotIn("CPU 没有执行性能模型", self.app)


if __name__ == "__main__":
    unittest.main()
