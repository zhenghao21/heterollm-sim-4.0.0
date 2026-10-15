"""Catalog consumers use current envelopes, metadata and filter fields."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from heterollm_sim.architecture_presets import architecture_preset_page, architecture_preset_detail
from heterollm_sim.component_presets import component_preset_page, component_preset_detail
from heterollm_sim.model_presets import list_model_presets


def test_browser_catalog_helpers_preserve_current_data_and_ignore_retired_aliases():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser contract checks")
    source = (Path(__file__).parents[1] / "src/heterollm_sim/webui/app.js").read_text(encoding="utf-8")
    names = ("componentPresetType", "componentPresetSpec", "componentPresetKind",
             "componentPresetDisplayName", "componentPresetVendor", "componentPresetSources",
             "componentPresetDetailItem", "componentPresetFacetValues", "architecturePresetDetailItem",
             "architecturePresetCompatibility", "architecturePresetDetailsMarkup",
             "architecturePresetLevel", "architecturePresetCategory", "architecturePresetSupport",
             "architecturePresetFidelity", "architecturePresetFidelityLabel", "architecturePresetVendor",
             "architecturePresetIsLoadable", "architecturePresetLimitations", "componentPresetEvidence",
             "componentPresetEvidenceLabel", "presetId", "presetSupportLevel", "presetText",
             "presetFactValue", "joinChineseList", "remoteModelPresetCardMarkup")
    functions = []
    for name in names:
        match = re.search(r"^function " + name + r"\([^\n]*\) \{.*?^\}", source, re.M | re.S)
        assert match, name
        functions.append(match.group())
    components = component_preset_page()
    architectures = architecture_preset_page()
    payload = dict(components=components, architectures=architectures, models=list_model_presets(),
                   component_details=[component_preset_detail(row["id"]) for row in components["items"]],
                   architecture_details=[architecture_preset_detail(row["id"]) for row in architectures["items"]])
    script = """
const assert = require('node:assert/strict');
const asObject = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const asArray = value => Array.isArray(value) ? value : [];
const deepClone = structuredClone;
const normalizedComponentKind = value => String(value).toLowerCase();
const escapeHtml = value => String(value);
const componentPresetFactMarkup = value => JSON.stringify(value);
const componentPresetListText = value => JSON.stringify(value);
const componentPresetSourceMarkup = value => JSON.stringify(value.sources);
const architecturePresetHardwareMarkup = () => '';
""" + "\n".join(functions) + "\nconst payload = " + json.dumps(payload) + ";\n" + """
for (const row of payload.components.items) {
  assert.equal(componentPresetSpec(row).kind, row.component_kind);
  assert.equal(componentPresetDisplayName(row), row.name);
  assert.equal(componentPresetKind(row), row.component_kind);
  assert.deepEqual(componentPresetSources(row), row.sources);
  assert.equal(componentPresetVendor(row), row.sources[0]?.publisher || '未注明厂商');
}
for (const raw of payload.component_details) {
  const detail = componentPresetDetailItem(raw);
  assert.deepEqual(componentPresetSpec(detail), raw.component);
}
for (const [key, values] of Object.entries(payload.components.filters)) {
  assert.deepEqual(componentPresetFacetValues(payload.components, key), values);
}
for (const [key, values] of Object.entries(payload.architectures.filters)) {
  if (key !== 'loadable') assert.deepEqual(componentPresetFacetValues(payload.architectures, key), values);
}
for (const raw of payload.architecture_details) {
  const detail = architecturePresetDetailItem({...raw, planner_executable: false,
      topology_evidence: {level: 'old'}, parameter_basis: {old: true}});
  assert.deepEqual(detail.topology_evidence, raw.preset.topology_evidence);
  assert.deepEqual(detail.parameter_basis, raw.preset.parameter_basis);
  assert.equal(detail.planner_executable, undefined);
  assert.deepEqual(detail.compatibility, raw.compatibility);
  const rendered = architecturePresetDetailsMarkup({...detail, planner_executable: false});
  assert.ok(rendered.includes('"Planner 可执行（Planner Executable）":"是（Yes）"'));
}
assert.deepEqual(componentPresetSpec({kind: 'gpu', capacity_bytes: 123}), {});
assert.deepEqual(componentPresetSources({metadata: {sources: [{title: 'old'}]}, source: 'old'}), []);
assert.equal(componentPresetVendor({vendor: 'old', manufacturer: 'old'}), '未注明厂商');
assert.equal(componentPresetDisplayName({display_name: 'old'}), '未命名组件预设');
assert.deepEqual(componentPresetFacetValues({facets: {component_kind: ['gpu']}}, 'component_kind'), []);
assert.deepEqual(componentPresetFacetValues({filters: {component_kind: [{value: 'gpu'}]}}, 'component_kind'), []);
for (const row of payload.models) {
  assert.equal(presetId(row), row.id);
  assert.equal(presetSupportLevel(row), row.support_level);
  assert.equal(presetText(row, 'parameter_scale'), row.parameter_scale);
  assert.equal(presetFactValue(row, 'layer_count'), String(row.layer_count));
}
assert.equal(presetId({preset_id: 'old'}), '');
assert.equal(presetSupportLevel({support: 'exact'}), 'out_of_domain');
const remote = remoteModelPresetCardMarkup({repo_id: 'example/current', sha: 'abc',
  license: 'mit', last_modified: 'today', id: 'old', name: 'old', revision: 'old'});
assert.ok(remote.includes('example/current'));
assert.ok(remote.includes('data-import-revision="abc"'));
assert.ok(!remote.includes('old'));
"""
    result = subprocess.run([node], input=script, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr
