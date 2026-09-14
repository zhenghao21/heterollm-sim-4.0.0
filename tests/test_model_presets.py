import http.client
import io
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.parse import urlsplit
from urllib.request import Request

from heterollm_sim.control_plane_planner import _derive_requirements
from heterollm_sim.config import model_from_dict
from heterollm_sim.ir import SCHEMA_VERSION, model_graph_execution_view
from heterollm_sim.model_catalog import (
    HuggingFaceClient,
    ModelCatalog,
    RemoteCatalogError,
    _HuggingFaceRedirectHandler,
)
from heterollm_sim.model_presets import (
    APPROXIMATION,
    ARCHITECTURE_EVIDENCE_STATUSES,
    CONFIG_VERIFIED_NO_OFFICIAL_DIAGRAM,
    DIAGRAM_VERIFIED,
    EXACT,
    GATED_CONFIG,
    METADATA_ONLY_UNSUPPORTED_IR,
    OUT_OF_DOMAIN,
    SUPPORT_LEVELS,
    UNPINNED_SOURCE,
    UnsupportedPresetError,
    list_model_presets,
    materialize_model_payload,
    model_preset_detail,
)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import compile_serving_plan
from heterollm_sim.web import build_server
from tests.model_helpers import execution_layers


class ModelPresetCatalogTests(unittest.TestCase):
    def test_catalog_has_many_distinct_architectures_families_and_scales(self):
        presets = list_model_presets()

        self.assertEqual(len(presets), 147)
        self.assertGreaterEqual(len(presets), 100)
        self.assertEqual(len({item["id"] for item in presets}), len(presets))
        self.assertGreaterEqual(len({item["family"] for item in presets}), 20)
        self.assertGreaterEqual(len({item["parameter_scale"] for item in presets}), 25)
        self.assertTrue(
            {
                "Qwen",
                "Qwen2.5",
                "Qwen3",
                "Mistral",
                "Mixtral",
                "DeepSeek-V3",
                "Phi",
                "OLMo",
                "BLOOM",
                "Pythia",
                "Falcon",
                "InternLM",
                "Yi",
                "Baichuan",
                "GLM",
                "SmolLM",
                "StarCoder2",
                "MPT",
                "DBRX",
                "Arctic",
            }.issubset({item["family"] for item in presets})
        )

    def test_metadata_has_provenance_license_and_conservative_openness(self):
        presets = list_model_presets()

        for item in presets:
            self.assertRegex(item["id"], r"^[a-z0-9][a-z0-9_-]*$")
            self.assertIn(item["support_level"], SUPPORT_LEVELS)
            self.assertIn(item["openness"], {"open_source", "open_weight"})
            self.assertTrue(item["source_repo"])
            self.assertTrue(item["source_revision"] or item["source_sha"])
            self.assertTrue(item["license"])
            self.assertTrue(item["notes"])
            self.assertTrue(item["architecture"])
            self.assertGreater(item["layer_count"], 0)
            self.assertIn(item["commercial_use"], {"allowed", "conditional", "prohibited", "unknown"})
            self.assertIn(item["access"], {"public", "gated", "requires_authentication"})
            self.assertIn(item["coverage"], {"full_language_model", "text_backbone_only", "metadata_only"})
            self.assertIsInstance(item["limitations"], list)
            self.assertIsInstance(item["modalities"], list)
            self.assertIn("config_hash", item)
            evidence = item["architecture_evidence"]
            self.assertIn(evidence["status"], ARCHITECTURE_EVIDENCE_STATUSES)
            self.assertTrue(evidence["source_type"])
            self.assertTrue(evidence["source_url"].startswith("https://"))
            config_url = evidence["config_url"]
            self.assertRegex(
                config_url,
                r"^https://huggingface\.co/.+/blob/(?:[0-9a-f]{40}|main)/config\.json$",
            )
            if item["source_sha"] is not None:
                self.assertIn(item["source_sha"], config_url)
            else:
                self.assertTrue(config_url.endswith("/blob/main/config.json"))
        custom_licenses = {
            item["id"]: item for item in presets if item["openness"] == "open_weight"
        }
        self.assertIn("bloom-176b", custom_licenses)
        self.assertIn("falcon-180b", custom_licenses)
        self.assertIn("glm4-9b", custom_licenses)
        self.assertIn("dbrx-132b-a36b", custom_licenses)
        self.assertGreaterEqual(
            sum(item["source_sha"] is not None for item in presets), 100
        )

    def test_architecture_evidence_marks_uncertain_and_unsupported_presets(self):
        by_id = {item["id"]: item for item in list_model_presets()}

        unpinned_repos = {
            item["source_repo"]
            for item in by_id.values()
            if item["architecture_evidence"]["status"] == UNPINNED_SOURCE
        }
        self.assertEqual(
            unpinned_repos,
            {
                "databricks/dbrx-base",
                "THUDM/GLM-4-32B-0414",
                "mosaicml/mpt-1b-redpajama-200b",
                "mosaicml/mpt-30b",
                "mosaicml/mpt-7b",
            },
        )
        self.assertEqual(
            {
                item["source_repo"]
                for item in by_id.values()
                if item["architecture_evidence"]["status"] == GATED_CONFIG
            },
            {"tiiuae/falcon-180B", "mistralai/Mistral-Large-Instruct-2407"},
        )

        for preset_id in (
            "deepseek-v4-pro-base",
            "falcon-h1-34b",
            "falcon-mamba-7b",
            "gpt-j-6b",
            "gpt-neo-2_7b",
            "kimi-k3",
            "qwen3-next-80b-a3b",
        ):
            item = by_id[preset_id]
            self.assertEqual(
                item["architecture_evidence"]["status"],
                METADATA_ONLY_UNSUPPORTED_IR,
            )
            self.assertFalse(item["generation_allowed"])
            self.assertIn("IR", item["architecture_evidence"]["notes"])

        self.assertEqual(
            by_id["qwen3_8-27b"]["architecture_evidence"]["status"],
            CONFIG_VERIFIED_NO_OFFICIAL_DIAGRAM,
        )
        self.assertIn(
            by_id["qwen3_8-27b"]["source_sha"],
            by_id["qwen3_8-27b"]["architecture_evidence"]["config_url"],
        )

    def test_architecture_evidence_counts_and_exact_diagram_scope(self):
        presets = list_model_presets()
        by_id = {item["id"]: item for item in presets}
        self.assertEqual(
            Counter(item["architecture_evidence"]["status"] for item in presets),
            Counter(
                {
                    DIAGRAM_VERIFIED: 15,
                    CONFIG_VERIFIED_NO_OFFICIAL_DIAGRAM: 117,
                    METADATA_ONLY_UNSUPPORTED_IR: 8,
                    UNPINNED_SOURCE: 5,
                    GATED_CONFIG: 2,
                }
            ),
        )

        diagram_ids = {
            "snowflake-arctic-480b",
            "bloom-560m",
            "bloom-1b1",
            "bloom-1b7",
            "bloom-3b",
            "bloom-7b1",
            "bloom-176b",
            "deepseek-moe-16b",
            "deepseek-v2-lite",
            "deepseek-v2-236b",
            "deepseek-v3-671b",
            "falcon-7b",
            "falcon-40b",
            "mistral-7b-v0_1",
            "mixtral-8x7b-v0_1",
        }
        self.assertEqual(
            {
                item["id"]
                for item in presets
                if item["architecture_evidence"]["status"] == DIAGRAM_VERIFIED
            },
            diagram_ids,
        )
        for preset_id in diagram_ids:
            evidence = by_id[preset_id]["architecture_evidence"]
            self.assertIn("Fig", evidence["notes"])
            self.assertTrue(evidence["uncertainty"])
            self.assertIn(
                by_id[preset_id]["source_sha"],
                evidence["config_url"],
            )

        phi_small = by_id["phi-3-small-128k"]
        self.assertEqual(
            phi_small["architecture_evidence"]["status"],
            METADATA_ONLY_UNSUPPORTED_IR,
        )
        self.assertFalse(phi_small["generation_allowed"])
        self.assertIn("block_sparse_attention", phi_small["unsupported_subgraphs"])
        self.assertEqual(
            phi_small["architecture_evidence"]["source_url"],
            "https://arxiv.org/pdf/2404.14219",
        )
        self.assertEqual(
            phi_small["architecture_evidence"]["source_type"],
            "official_technical_report_figure",
        )
        self.assertIn("Figure 1", phi_small["architecture_evidence"]["notes"])
        self.assertIn("PDF p.3", phi_small["architecture_evidence"]["notes"])
        self.assertIsNone(model_preset_detail("phi-3-small-128k")["model"])
        with self.assertRaises(UnsupportedPresetError):
            materialize_model_payload("phi-3-small-128k")

        qwen_next = by_id["qwen3-next-80b-a3b"]
        self.assertEqual(
            qwen_next["architecture_evidence"]["status"],
            METADATA_ONLY_UNSUPPORTED_IR,
        )
        self.assertFalse(qwen_next["generation_allowed"])
        self.assertIn("gated_delta_net", qwen_next["unsupported_subgraphs"])
        self.assertEqual(
            qwen_next["architecture_evidence"]["source_type"],
            "official_pinned_model_card_architecture_diagram",
        )
        self.assertIn(
            "README",
            qwen_next["architecture_evidence"]["source_url"],
        )
        self.assertIn(
            qwen_next["source_sha"],
            qwen_next["architecture_evidence"]["config_url"],
        )

        falcon_180b = by_id["falcon-180b"]["architecture_evidence"]
        self.assertEqual(falcon_180b["status"], GATED_CONFIG)
        self.assertIn("arxiv.org", falcon_180b["source_url"])

    def test_catalog_audit_parameter_conflicts_are_corrected(self):
        expected_intermediate_sizes = {
            "qwen-1_8b": 11008,
            "qwen-7b": 22016,
            "qwen-14b": 27392,
            "internlm2-1_8b": 8192,
        }

        for preset_id, expected in expected_intermediate_sizes.items():
            model = model_from_dict(materialize_model_payload(preset_id))
            self.assertEqual(execution_layers(model)[0].intermediate_size, expected)

    def test_every_supported_payload_parses_and_has_unique_layer_ids(self):
        supported = [
            item for item in list_model_presets() if item["support_level"] != OUT_OF_DOMAIN
        ]

        for item in supported:
            payload = materialize_model_payload(item["id"])
            model = model_from_dict(payload)
            layers = execution_layers(model)
            self.assertEqual(model.schema_version, SCHEMA_VERSION)
            self.assertEqual(model.num_layers, item["layer_count"])
            self.assertEqual(
                len({layer.layer_id for layer in layers}), model.num_layers
            )
            self.assertTrue(all(layer.dtype for layer in layers))
            self.assertTrue(
                all(layer.experts_per_token <= layer.num_experts for layer in layers)
            )

    def test_supported_payload_shapes_are_not_chat_alias_duplicates(self):
        signatures = {}
        for item in list_model_presets():
            if not item["generation_allowed"]:
                continue
            model = model_from_dict(materialize_model_payload(item["id"]))
            layers = execution_layers(model)
            signature = (
                model.architecture,
                model.vocabulary_size,
                model.max_sequence_length,
                tuple(
                    (
                        layer.kind,
                        layer.hidden_size,
                        layer.intermediate_size,
                        layer.attention_heads,
                        layer.kv_heads,
                        layer.num_experts,
                        layer.experts_per_token,
                    )
                    for layer in layers
                ),
            )
            family_signature = (item["family"], signature)
            self.assertNotIn(
                family_signature,
                signatures,
                (item["id"], signatures.get(family_signature)),
            )
            signatures[family_signature] = item["id"]

    def test_approximation_and_out_of_domain_are_explicit(self):
        presets = list_model_presets()
        by_id = {item["id"]: item for item in presets}

        self.assertEqual(
            Counter(item["support_level"] for item in presets),
            Counter({EXACT: 116, APPROXIMATION: 23, OUT_OF_DOMAIN: 8}),
        )

        self.assertEqual(
            by_id["deepseek-v3-671b"]["support_level"],
            "analytical_approximation",
        )
        self.assertEqual(
            by_id["qwen3-next-80b-a3b"]["support_level"], OUT_OF_DOMAIN
        )
        self.assertFalse(by_id["qwen3-next-80b-a3b"]["generation_allowed"])
        self.assertIsNone(model_preset_detail("qwen3-next-80b-a3b")["model"])
        with self.assertRaises(UnsupportedPresetError):
            materialize_model_payload("qwen3-next-80b-a3b")

    def test_catalog_and_payload_materialization_are_deterministic(self):
        first = list_model_presets()
        second = list_model_presets()
        self.assertEqual(first, second)
        self.assertEqual(
            materialize_model_payload("mixtral-8x7b-v0_1"),
            materialize_model_payload("mixtral-8x7b-v0_1"),
        )

    def test_qwen35_and_qwen38_use_pinned_hybrid_schema_04_fields(self):
        by_id = {item["id"]: item for item in list_model_presets()}

        self.assertEqual(
            {
                "qwen3_5-0_8b",
                "qwen3_5-2b",
                "qwen3_5-4b",
                "qwen3_5-9b",
                "qwen3_5-27b",
                "qwen3_5-35b-a3b",
                "qwen3_5-122b-a10b",
                "qwen3_5-397b-a17b",
            },
            {item["id"] for item in by_id.values() if item["family"] == "Qwen3.5"},
        )
        giant = by_id["qwen3_8-2_4t-a95b"]
        self.assertEqual(giant["layer_count"], 92)
        self.assertEqual(giant["model_kind"], "moe")
        self.assertEqual(giant["license"], "Qwen3.8-Max License")
        self.assertEqual(giant["commercial_use"], "conditional")
        self.assertRegex(giant["source_sha"], r"^[0-9a-f]{40}$")
        self.assertRegex(giant["config_hash"], r"^[0-9a-f]{64}$")

        payload = materialize_model_payload("qwen3_8-2_4t-a95b")
        self.assertNotIn("layers", payload)
        self.assertNotIn("mtp_prediction_layers", payload)
        self.assertNotIn("mtp_aux_head", payload)
        self.assertFalse(payload["text_backbone_only"])
        self.assertEqual(payload["supported_modalities"], ["text"])
        giant_model = model_from_dict(payload)
        layers = execution_layers(giant_model)
        self.assertEqual(layers[0].sequence_mixer, "linear_attention")
        self.assertEqual(layers[3].sequence_mixer, "full_attention")
        self.assertEqual(layers[0].num_experts, 512)
        self.assertEqual(layers[0].experts_per_token, 10)
        self.assertEqual(layers[0].shared_expert_intermediate_size, 2048)
        self.assertEqual(layers[0].linear_attention.value_heads, 128)
        execution_view = model_graph_execution_view(giant_model.graph)
        predictions = tuple(
            item
            for item in execution_view.mtp_descriptors
            if item.prediction_index is not None
        )
        auxiliaries = tuple(
            item
            for item in execution_view.mtp_descriptors
            if item.prediction_index is None
        )
        self.assertEqual(len(predictions), 1)
        self.assertEqual(len(auxiliaries), 1)
        self.assertEqual(
            predictions[0].weight_bytes,
            8192 * 8192 * 2,
        )
        self.assertEqual(
            auxiliaries[0].weight_bytes,
            giant_model.vocabulary_size * 8192 * 2,
        )
        scenario = build_reference_scenario()
        configured = replace(
            scenario,
            model=giant_model,
            placement=replace(
                scenario.placement, model_name=giant_model.name
            ),
        )
        derived_weights = {
            requirement.tensor_id: requirement.tensor_bytes
            for requirement in _derive_requirements(configured)
            if requirement.tensor_id and "weight" in requirement.tensor_id
        }
        self.assertEqual(
            giant_model.total_declared_weight_bytes,
            sum(derived_weights.values()),
        )

        multimodal = materialize_model_payload("qwen3_8-27b")
        self.assertTrue(multimodal["text_backbone_only"])
        self.assertEqual(multimodal["supported_modalities"], ["text"])
        self.assertEqual(
            multimodal["excluded_subgraphs"],
            ["vision_encoder", "multimodal_projector"],
        )

    def test_qwen35_and_qwen38_explicit_head_dim_compiles_serving_kv(self):
        base = build_reference_scenario()
        for preset_id in ("qwen3_5-27b", "qwen3_8-27b"):
            model = model_from_dict(materialize_model_payload(preset_id))
            self.assertTrue(
                all(
                    layer.attention_head_dim == 256
                    for layer in execution_layers(model)
                )
            )
            scenario = replace(
                base,
                model=model,
                placement=replace(base.placement, model_name=model.name),
            )
            self.assertGreater(compile_serving_plan(scenario).kv_policy.bytes_per_page, 0)

        giant = model_from_dict(
            materialize_model_payload("qwen3_8-2_4t-a95b")
        )
        kv_policy = replace(base.placement.kv_policy, dtype=None)
        giant_scenario = replace(
            base,
            model=giant,
            placement=replace(
                base.placement,
                model_name=giant.name,
                kv_policy=kv_policy,
            ),
        )
        plan = compile_serving_plan(giant_scenario)
        full_attention_layers = sum(
            not layer.is_linear_attention for layer in execution_layers(giant)
        )
        expected_bytes_per_token = full_attention_layers * 2 * 4 * 256 * 2
        self.assertEqual(
            plan.kv_policy.bytes_per_page // plan.kv_policy.tokens_per_page,
            expected_bytes_per_token,
        )

    def test_kimi_and_deepseek_v4_presets_are_pinned_and_fail_closed(self):
        by_id = {item["id"]: item for item in list_model_presets()}

        expected_sources = {
            "kimi-k2-instruct": (
                "moonshotai/Kimi-K2-Instruct",
                "fd1984e2b7a3350dbf7305fe73a4ede25c14de50",
                "8c13ae1049df55f29b3bdcae69a562433f243ff70dac251d819ecad8dbdf7439",
            ),
            "kimi-k2-instruct-0905": (
                "moonshotai/Kimi-K2-Instruct-0905",
                "ac6c49f04883bd0a0598b790693a72061c676629",
                "4cd10b0f3cb1c1dfdbfcdd2f303057500f570b24f671ce5eb7650b6a511807cc",
            ),
            "kimi-k2_5": (
                "moonshotai/Kimi-K2.5",
                "4d01dfe0332d63057c186e0b262165819efb6611",
                "acd5bb01a16f64b309599cd6ed196be056f613c99d6bc9300692b82cd10882f6",
            ),
            "kimi-k3": (
                "moonshotai/Kimi-K3",
                "a590ce090cb049c93a33dfe8c208ec652aa20503",
                "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
            ),
            "deepseek-v4-pro-base": (
                "deepseek-ai/DeepSeek-V4-Pro-Base",
                "98730c030fbdbaca4950788280a35c4642b208a9",
                "718c188325de5ec90885a2782990b6e49fe773ef701c3aed03c55c8c21b99059",
            ),
        }
        for preset_id, (repo, source_sha, config_hash) in expected_sources.items():
            self.assertEqual(by_id[preset_id]["source_repo"], repo)
            self.assertEqual(by_id[preset_id]["source_sha"], source_sha)
            self.assertEqual(by_id[preset_id]["config_hash"], config_hash)
            self.assertEqual(
                by_id[preset_id]["provenance_status"],
                "pinned_commit_and_config",
            )

        for preset_id in (
            "kimi-k2-instruct",
            "kimi-k2-instruct-0905",
            "kimi-k2_5",
        ):
            item = by_id[preset_id]
            self.assertEqual(item["support_level"], "analytical_approximation")
            self.assertTrue(item["generation_allowed"])
            self.assertEqual(item["license"], "Modified MIT License")
            self.assertEqual(item["openness"], "open_weight")
            self.assertEqual(item["commercial_use"], "conditional")
            payload = materialize_model_payload(preset_id)
            model = model_from_dict(payload)
            layers = execution_layers(model)
            self.assertEqual(layers[0].kind, "dense")
            self.assertEqual(layers[1].kind, "moe")
            self.assertEqual(layers[1].num_experts, 384)
            self.assertEqual(layers[1].experts_per_token, 8)
            self.assertEqual(
                layers[1].shared_expert_intermediate_size,
                18432,
            )
            self.assertIn("MLA", " ".join(item["limitations"]))
            self.assertIn("61 层", item["notes"])

        k25 = by_id["kimi-k2_5"]
        self.assertEqual(k25["coverage"], "text_backbone_only")
        self.assertEqual(k25["modalities"], ["text", "image", "video"])
        self.assertEqual(k25["supported_modalities"], ["text"])
        self.assertEqual(
            k25["unsupported_subgraphs"],
            ["vision_encoder", "video_encoder", "multimodal_projector"],
        )
        k25_payload = materialize_model_payload("kimi-k2_5")
        self.assertTrue(k25_payload["text_backbone_only"])
        self.assertEqual(
            k25_payload["excluded_subgraphs"],
            ["vision_encoder", "video_encoder", "multimodal_projector"],
        )
        self.assertIn("视频塔", " ".join(k25["limitations"]))

        k3 = by_id["kimi-k3"]
        self.assertEqual(k3["support_level"], OUT_OF_DOMAIN)
        self.assertFalse(k3["generation_allowed"])
        self.assertEqual(k3["coverage"], "metadata_only")
        self.assertEqual(k3["license"], "Kimi K3 License")
        self.assertEqual(k3["openness"], "open_weight")
        self.assertEqual(k3["commercial_use"], "conditional")
        self.assertEqual(k3["layer_count"], 93)
        self.assertTrue(
            {
                "kda_linear_attention",
                "attention_residual_blocks",
                "vision_encoder",
            }.issubset(set(k3["unsupported_subgraphs"]))
        )
        self.assertIn("KDA", " ".join(k3["limitations"]))
        self.assertIn("AttnRes", " ".join(k3["limitations"]))
        self.assertIsNone(model_preset_detail("kimi-k3")["model"])
        with self.assertRaises(UnsupportedPresetError):
            materialize_model_payload("kimi-k3")

        v4 = by_id["deepseek-v4-pro-base"]
        self.assertEqual(v4["support_level"], OUT_OF_DOMAIN)
        self.assertFalse(v4["generation_allowed"])
        self.assertEqual(v4["coverage"], "metadata_only")
        self.assertEqual(v4["parameter_scale"], "unspecified")
        self.assertEqual(v4["license"], "MIT")
        self.assertEqual(v4["openness"], "open_source")
        self.assertEqual(v4["commercial_use"], "allowed")
        self.assertEqual(
            v4["unsupported_subgraphs"],
            [
                "hash_attention",
                "context_compression",
                "sliding_window_attention",
                "special_attention",
            ],
        )
        limitation_text = " ".join(v4["limitations"])
        self.assertIn("Hash Attention", limitation_text)
        self.assertIn("上下文压缩", limitation_text)
        self.assertIn("滑动窗口", limitation_text)
        self.assertIn("特殊注意力", limitation_text)
        self.assertIn("总参数", limitation_text)
        self.assertIn("61 层", v4["notes"])
        self.assertIsNone(model_preset_detail("deepseek-v4-pro-base")["model"])
        with self.assertRaises(UnsupportedPresetError):
            materialize_model_payload("deepseek-v4-pro-base")

        with tempfile.TemporaryDirectory() as directory:
            page = ModelCatalog(Path(directory) / "catalog-cache").page(
                family="Kimi K2",
                support_level="analytical_approximation",
                limit=10,
            )
        self.assertEqual(page["total"], 2)
        self.assertEqual(
            {item["id"] for item in page["items"]},
            {"kimi-k2-instruct", "kimi-k2-instruct-0905"},
        )

    def test_anonymously_unresolved_repositories_are_not_claimed_as_pinned(self):
        unresolved = {
            item["source_repo"]: item
            for item in list_model_presets()
            if item["source_sha"] is None
        }

        self.assertEqual(
            set(unresolved),
            {
                "databricks/dbrx-base",
                "THUDM/GLM-4-32B-0414",
                "mosaicml/mpt-1b-redpajama-200b",
                "mosaicml/mpt-30b",
                "mosaicml/mpt-7b",
            },
        )
        for item in unresolved.values():
            self.assertEqual(item["access"], "requires_authentication")
            self.assertEqual(
                item["provenance_status"],
                "unverified_requires_authentication",
            )
            self.assertTrue(item["limitations"])
            limitation = " ".join(item["limitations"])
            self.assertIn("cannot be anonymously resolved and pinned", limitation)
            self.assertIn("内置 config 快照", limitation)
            self.assertIn("source_sha 保持为空", limitation)


class ModelPresetApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = build_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=5.0)
        cls.server.server_close()

    def get(self, path):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            raw = response.read()
            return response.status, raw, json.loads(raw.decode("utf-8"))
        finally:
            connection.close()

    def post(self, path, payload, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        try:
            body = json.dumps(payload).encode("utf-8")
            request_headers = {"Content-Type": "application/json"}
            request_headers.update(headers or {})
            connection.request("POST", path, body=body, headers=request_headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, raw, json.loads(raw.decode("utf-8"))
        finally:
            connection.close()

    def test_list_detail_and_unknown_id_endpoints(self):
        status, raw, page = self.get("/api/model-presets")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(page["total"], 100)
        self.assertEqual(len(page["items"]), page["limit"])
        self.assertNotIn("layers", page["items"][0])
        self.assertNotIn("model", page["items"][0])

        repeat_status, repeat_raw, repeat_page = self.get("/api/model-presets")
        self.assertEqual(repeat_status, 200)
        self.assertEqual(raw, repeat_raw)
        self.assertEqual(page, repeat_page)

        status, _, detail = self.get("/api/model-presets/mixtral-8x7b-v0_1")
        self.assertEqual(status, 200)
        self.assertEqual(detail["preset"]["id"], "mixtral-8x7b-v0_1")
        model = model_from_dict(detail["model"])
        self.assertEqual(execution_layers(model)[0].kind, "moe")

        status, _, payload = self.get("/api/model-presets/does-not-exist")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"]["code"], "not_found")

    def test_out_of_domain_detail_is_display_only(self):
        status, _, detail = self.get("/api/model-presets/falcon-h1-34b")

        self.assertEqual(status, 200)
        self.assertEqual(detail["preset"]["support_level"], OUT_OF_DOMAIN)
        self.assertFalse(detail["preset"]["generation_allowed"])
        self.assertIsNone(detail["model"])
        self.assertFalse(detail["graph"]["executable"])

    def test_page_envelope_filtering_is_the_only_v3_list_contract(self):
        status, _, page = self.get(
            "/api/model-presets?limit=2&offset=0&family=Qwen3.8"
        )
        self.assertEqual(status, 200)
        self.assertEqual(page["catalog_version"], "0.5")
        self.assertIn("cutoff_at", page)
        self.assertEqual(page["total"], 2)
        self.assertEqual(len(page["items"]), 2)
        self.assertIsNone(page["next_offset"])
        self.assertTrue(all(item["family"] == "Qwen3.8" for item in page["items"]))
        status, _, error = self.get("/api/model-presets?format=page")
        self.assertEqual(status, 400)
        self.assertEqual(error["error"]["code"], "unknown_query_fields")

    def test_model_kind_and_architecture_filters_are_distinct(self):
        status, _, dense = self.get(
            "/api/model-presets?limit=200&model_kind=dense"
        )
        self.assertEqual(status, 200)
        self.assertGreater(dense["total"], 0)
        self.assertTrue(all(item["model_kind"] == "dense" for item in dense["items"]))
        self.assertIn("model_kind", dense["facets"])

        status, _, moe = self.get(
            "/api/model-presets?limit=200&model_kind=moe"
        )
        self.assertEqual(status, 200)
        self.assertGreater(moe["total"], 0)
        self.assertTrue(all(item["model_kind"] == "moe" for item in moe["items"]))
        status, _, no_alias = self.get(
            "/api/model-presets?limit=200&architecture=moe"
        )
        self.assertEqual(status, 200)
        self.assertEqual(no_alias["total"], 0)


class _FakeResponse(io.BytesIO):
    def __init__(self, body, url):
        super().__init__(body)
        self._url = url
        self.headers = {"Content-Length": str(len(body))}

    def geturl(self):
        return self._url


class _FakeHuggingFace:
    def __init__(self, repo_info, config, search_items=None):
        self.repo_info = repo_info
        self.config = config
        self.search_items = search_items or []
        self.urls = []

    def __call__(self, request, timeout):
        url = request.full_url
        self.urls.append((url, timeout))
        path = urlsplit(url).path
        if path == "/api/models":
            payload = self.search_items
        elif path.startswith("/api/models/"):
            payload = self.repo_info
        elif path.endswith("/config.json"):
            payload = self.config
        else:
            raise AssertionError("unexpected URL: {}".format(url))
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        return _FakeResponse(body, url)


class ImportedCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp.name) / "catalog-cache"
        self.config = {
            "architectures": ["LlamaForCausalLM"],
            "auto_map": {"AutoModel": "configuration_evil.EvilModel"},
            "model_type": "llama",
            "num_hidden_layers": 4,
            "hidden_size": 512,
            "intermediate_size": 1536,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 32000,
            "max_position_embeddings": 8192,
        }
        self.info = {
            "id": "publisher/Safe-1B",
            "author": "publisher",
            "sha": "a" * 40,
            "private": False,
            "gated": False,
            "lastModified": "2026-08-20T00:00:00Z",
            "cardData": {"license": "apache-2.0"},
            "tags": ["text-generation", "license:apache-2.0"],
        }
        self.transport = _FakeHuggingFace(self.info, self.config)
        self.catalog = ModelCatalog(
            self.cache_dir,
            hf_client=HuggingFaceClient(opener=self.transport, timeout=3),
        )

    def tearDown(self):
        self.temp.cleanup()

    def import_config(self, repo_id, config):
        info = dict(self.info)
        info["id"] = repo_id
        catalog = ModelCatalog(
            self.cache_dir,
            hf_client=HuggingFaceClient(
                opener=_FakeHuggingFace(info, config), timeout=3
            ),
        )
        return catalog.import_repository(repo_id, "main")

    def test_import_is_config_only_cached_pinned_and_deduplicated_on_update(self):
        self.assertFalse(self.cache_dir.exists())

        detail = self.catalog.import_repository("publisher/Safe-1B", "release-1")

        self.assertTrue(self.cache_dir.is_dir())
        self.assertEqual(len(list(self.cache_dir.glob("*.json"))), 1)
        self.assertEqual(detail["provenance"]["resolved_sha"], "a" * 40)
        self.assertRegex(detail["provenance"]["config_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(detail["preset"]["license"], "apache-2.0")
        self.assertEqual(detail["preset"]["openness"], "open_source")
        self.assertEqual(detail["preset"]["source"], "huggingface_import")
        self.assertTrue(detail["preset"]["generation_allowed"])
        self.assertIn("never executed", " ".join(detail["preset"]["limitations"]))
        self.assertEqual(model_from_dict(detail["model"]).num_layers, 4)

        total = len(self.catalog.list_metadata())
        self.catalog.import_repository("publisher/Safe-1B", "release-1")
        self.assertEqual(len(self.catalog.list_metadata()), total)
        self.assertEqual(len(list(self.cache_dir.glob("*.json"))), 1)
        self.assertTrue(
            all("config.json" in url or "/api/models/" in url for url, _ in self.transport.urls)
        )
        self.assertFalse(any("safetensors" in url or ".py" in url for url, _ in self.transport.urls))

    def test_imported_dtype_and_weight_quantization_drive_materialized_bytes(self):
        fp32_config = dict(self.config, torch_dtype="float32")
        fp32 = self.import_config("publisher/FP32-1B", fp32_config)
        fp32_layer = execution_layers(model_from_dict(fp32["model"]))[0]

        w4_config = dict(
            self.config,
            quantization_config={"quant_method": "gptq", "bits": 4},
        )
        w4 = self.import_config("publisher/W4-1B", w4_config)
        w4_layer = execution_layers(model_from_dict(w4["model"]))[0]

        self.assertEqual(fp32_layer.dtype, "fp32")
        self.assertIsNone(fp32_layer.quantization)
        self.assertEqual(fp32_layer.metadata["weight_storage_bits"], 32)
        self.assertEqual(w4_layer.dtype, "bf16")
        self.assertEqual(w4_layer.quantization, "w4")
        self.assertEqual(w4_layer.metadata["weight_storage_bits"], 4)
        self.assertEqual(
            fp32_layer.weight_bytes,
            8 * w4_layer.weight_bytes,
        )

    def test_nested_text_dtype_wins_over_multimodal_top_level_dtype(self):
        text_config = dict(self.config, torch_dtype="bfloat16")
        config = {
            "model_type": "multimodal_wrapper",
            "torch_dtype": "float32",
            "vision_config": {"model_type": "vision"},
            "text_config": text_config,
        }

        detail = self.import_config("publisher/Nested-Text-1B", config)
        layer = execution_layers(model_from_dict(detail["model"]))[0]

        self.assertEqual(layer.dtype, "bf16")
        self.assertTrue(detail["model"]["text_backbone_only"])

    def test_ambiguous_or_conflicting_quantization_fails_closed(self):
        configs = {
            "ambiguous": dict(
                self.config,
                quantization_config={"quant_method": "fp8"},
            ),
            "conflicting": dict(
                self.config,
                quantization_config={
                    "quant_method": "gptq",
                    "bits": 4,
                    "load_in_8bit": True,
                },
            ),
        }

        for suffix, config in configs.items():
            with self.subTest(case=suffix):
                detail = self.import_config(
                    "publisher/Quant-{}-1B".format(suffix),
                    config,
                )
                self.assertEqual(
                    detail["preset"]["support_level"], OUT_OF_DOMAIN
                )
                self.assertFalse(detail["preset"]["generation_allowed"])
                self.assertIsNone(detail["model"])
                self.assertTrue(
                    any(
                        suffix in limitation
                        for limitation in detail["preset"]["limitations"]
                    )
                )

    def test_remote_search_is_explicit_lightweight_and_does_not_write_cache(self):
        search_transport = _FakeHuggingFace(
            self.info,
            self.config,
            search_items=[
                {
                    "id": "publisher/Safe-1B",
                    "author": "publisher",
                    "sha": "a" * 40,
                    "lastModified": "2026-08-20T00:00:00Z",
                    "private": False,
                    "gated": "manual",
                    "tags": ["text-generation", "license:apache-2.0"],
                    "cardData": {"license": "apache-2.0"},
                }
            ],
        )
        catalog = ModelCatalog(
            self.cache_dir,
            hf_client=HuggingFaceClient(opener=search_transport),
        )

        items = catalog.remote_search("safe model", 5)

        self.assertFalse(self.cache_dir.exists())
        self.assertEqual(items[0]["repo_id"], "publisher/Safe-1B")
        self.assertEqual(items[0]["access"], "gated")
        self.assertEqual(items[0]["license"], "apache-2.0")
        self.assertEqual(len(search_transport.urls), 1)
        self.assertTrue(search_transport.urls[0][0].startswith("https://huggingface.co/api/models?"))

    def test_unknown_import_is_metadata_only_with_layer_override(self):
        info = dict(self.info)
        info["id"] = "publisher/Future-2B"
        info["gated"] = "manual"
        info["cardData"] = {"license": "other", "license_name": "Future Model License"}
        config = {
            "model_type": "future_hybrid",
            "num_hidden_layers": 36,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "vocab_size": 64000,
            "max_position_embeddings": 32768,
            "auto_map": {"AutoModel": "modeling_future.FutureModel"},
        }
        catalog = ModelCatalog(
            self.cache_dir,
            hf_client=HuggingFaceClient(opener=_FakeHuggingFace(info, config)),
        )

        detail = catalog.import_repository("publisher/Future-2B", "main")

        self.assertEqual(detail["preset"]["support_level"], OUT_OF_DOMAIN)
        self.assertEqual(detail["preset"]["layer_count"], 36)
        self.assertEqual(detail["preset"]["coverage"], "metadata_only")
        self.assertEqual(detail["preset"]["access"], "gated")
        self.assertFalse(detail["preset"]["generation_allowed"])
        self.assertIsNone(detail["model"])

    def test_cross_origin_redirect_is_rejected_before_the_second_hop(self):
        handler = _HuggingFaceRedirectHandler()
        initial = Request(
            "https://huggingface.co/api/models/publisher/Safe-1B",
            headers={"Authorization": "Bearer test-secret"},
        )
        sent_second_hops = []

        def follow(new_url):
            redirected = handler.redirect_request(
                initial,
                None,
                302,
                "Found",
                {},
                new_url,
            )
            sent_second_hops.append(redirected.full_url)

        with self.assertRaises(RemoteCatalogError) as raised:
            follow("https://attacker.invalid/collect-token")

        self.assertEqual(raised.exception.code, "remote_redirect_forbidden")
        self.assertEqual(sent_second_hops, [])


class ImportedCatalogApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp.name) / "api-cache"
        info = {
            "id": "publisher/Safe-1B",
            "author": "publisher",
            "sha": "b" * 40,
            "private": False,
            "gated": False,
            "cardData": {"license": "apache-2.0"},
        }
        config = {
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 256,
            "intermediate_size": 768,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
        }
        search = [
            {
                "id": "publisher/Safe-1B",
                "author": "publisher",
                "sha": "b" * 40,
                "private": False,
                "gated": False,
                "tags": ["text-generation", "license:apache-2.0"],
            }
        ]
        self.transport = _FakeHuggingFace(info, config, search)
        self.server = build_server(
            "127.0.0.1",
            0,
            catalog_cache_dir=self.cache_dir,
            hf_client=HuggingFaceClient(opener=self.transport),
        )
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=5.0)
        self.server.server_close()
        self.temp.cleanup()

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5.0)
        try:
            body = None
            request_headers = dict(headers or {})
            if payload is not None:
                body = json.dumps(payload).encode("utf-8")
                request_headers.setdefault("Content-Type", "application/json")
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, json.loads(raw.decode("utf-8"))
        finally:
            connection.close()

    def test_remote_search_then_import_and_detail(self):
        status, search = self.request(
            "GET", "/api/model-presets/remote-search?query=safe&limit=5"
        )
        self.assertEqual(status, 200)
        self.assertEqual(search["items"][0]["repo_id"], "publisher/Safe-1B")
        self.assertFalse(self.cache_dir.exists())

        status, imported = self.request(
            "POST",
            "/api/model-presets/import",
            {"repo_id": "publisher/Safe-1B", "revision": "release"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(imported["provenance"]["resolved_sha"], "b" * 40)
        self.assertTrue(self.cache_dir.is_dir())
        preset_id = imported["preset"]["id"]

        status, detail = self.request(
            "GET", "/api/model-presets/{}".format(preset_id)
        )
        self.assertEqual(status, 200)
        self.assertEqual(detail["preset"]["source"], "huggingface_import")
        model_from_dict(detail["model"])

    def test_import_errors_are_structured_and_same_origin_is_enforced(self):
        status, payload = self.request(
            "POST", "/api/model-presets/import", {"repo_id": "../bad"}
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"]["code"], "invalid_repo_id")

        status, payload = self.request(
            "POST",
            "/api/model-presets/import",
            {"repo_id": "publisher/Safe-1B"},
            {"Origin": "https://attacker.invalid"},
        )
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"]["code"], "cross_origin_forbidden")


if __name__ == "__main__":
    unittest.main()
