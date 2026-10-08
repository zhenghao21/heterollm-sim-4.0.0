"""Frontend GGUF presets and explicit, non-destructive parameter derivatives.

Records contain tensor inventories, never weights or measured model timings.
Every load rebuilds through the same architecture adapter as direct GGUF import.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, replace
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

from .gguf_parity import (
    GGUFMetadata, GGUFTensor, _TENSOR_TYPES, build_model_from_gguf, read_gguf_metadata,
    validate_gguf_inventory,
)
from .ir import model_graph_execution_view
from .model_catalog import ModelCatalog
from .serde import to_primitive


SCHEMA = "heterollm.gguf-preset-record/v1"
PARAMETERS = ("layer_count", "hidden_size", "intermediate_size", "attention_heads",
              "kv_heads", "head_dim", "vocabulary_size", "max_sequence_length")
QUANTIZATIONS = ("preserve", "F16", "BF16", "Q8_0", "Q4_K", "Q5_K", "Q6_K",
                 "IQ3_S", "IQ4_XS")
_TYPE_BY_NAME = {value[0]: (key, *value[1:]) for key, value in _TENSOR_TYPES.items()}


def inventory_to_gguf(raw):
    fields = GGUFMetadata.__dataclass_fields__
    payload = {
        key: deepcopy(raw[key]) for key, field in fields.items()
        if key != "tensor_directory" and (key in raw or field.default is MISSING)
    }
    payload["tensor_directory"] = tuple(
        GGUFTensor(**{**row, "shape": tuple(row["shape"])}) for row in raw["tensor_directory"]
    )
    return GGUFMetadata(**payload)


def _record_execution_issue(record):
    """Source presence is not executable support; use the actual graph builder."""
    from .architecture_adapters import resolve_gguf_architecture_adapter
    try:
        gguf = inventory_to_gguf(record["inventory"])
        validate_gguf_inventory(gguf)
        if resolve_gguf_architecture_adapter(gguf.architecture) is None:
            return "gguf_unsupported", f"已有 GGUF，架构待适配：{gguf.architecture}"
        build_model_from_gguf(gguf)
    except (ValueError, TypeError, KeyError) as exc:
        return "gguf_incomplete", f"GGUF 目录或模型结构未通过校验：{exc}"
    return None


def _configuration(gguf):
    model = build_model_from_gguf(gguf)
    view = model_graph_execution_view(model.graph, schema_version=model.schema_version)
    first = view.layer_instances[0].layer
    return dict(zip(PARAMETERS, (
        gguf.n_layer, gguf.n_embd, first.intermediate_size, gguf.n_head,
        gguf.n_head_kv, first.attention_head_dim, gguf.vocab_size, gguf.context_length,
    )), weight_quantization="preserve")


def derive_inventory(raw, parameters):
    """Repack explicitly edited shapes; refuse unrecognized tensor semantics."""
    gguf = inventory_to_gguf(raw)
    current = _configuration(gguf)
    if not isinstance(parameters, dict) or set(parameters) - set(current):
        raise ValueError("模型参数包含未知字段")
    target = {**current, **parameters}
    for key in PARAMETERS:
        value = target[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} 必须是正整数")
    if target["layer_count"] > 4096:
        raise ValueError("层数不能超过 4096")
    if target["attention_heads"] % target["kv_heads"]:
        raise ValueError("注意力头数必须能被 KV 头数整除")
    quant = target["weight_quantization"]
    if quant not in QUANTIZATIONS:
        raise ValueError("不支持的权重量化格式")
    changed = {key: {"before": current[key], "after": value}
               for key, value in target.items() if current[key] != value}
    if not changed:
        return deepcopy(raw), {}
    hybrid = gguf.architecture in {"qwen35", "qwen35moe"}
    if gguf.architecture not in {"llama", "qwen2", "qwen3", "qwen35"}:
        raise ValueError("当前架构尚未实现参数编辑，不能推测其张量变换")
    meta = dict(gguf.metadata)
    # Edited tensors are a virtual packed inventory, not the original shard
    # files. Their original physical origins remain in the preset record.
    for key in tuple(meta):
        if key.startswith("split."):
            del meta[key]
    prefix = str(gguf.architecture)
    metadata_names = {
        "layer_count": "block_count", "hidden_size": "embedding_length",
        "intermediate_size": "feed_forward_length", "attention_heads": "attention.head_count",
        "kv_heads": "attention.head_count_kv", "max_sequence_length": "context_length",
        "head_dim": "attention.key_length",
    }
    for key, suffix in metadata_names.items():
        meta[f"{prefix}.{suffix}"] = target[key]
    meta[f"{prefix}.attention.value_length"] = target["head_dim"]
    # MTP weights are outside this editor's executable text backbone.
    meta[f"{prefix}.nextn_predict_layers"] = 0
    meta["general.vocab_size"] = target["vocabulary_size"]
    meta["tokenizer.ggml.tokens"] = {"count": target["vocabulary_size"]}
    if "head_dim" in changed and not hybrid:
        meta[f"{prefix}.rope.dimension_count"] = target["head_dim"]
    if quant != "preserve":
        meta.pop("general.file_type", None)
    h, f, d, v = (target[key] for key in ("hidden_size", "intermediate_size", "head_dim", "vocabulary_size"))
    q, kv = target["attention_heads"] * d, target["kv_heads"] * d
    shapes = {
        "token_embd.weight": (h, v), "output.weight": (h, v), "output_norm.weight": (h,),
        "attn_norm.weight": (h,), "ffn_norm.weight": (h,), "post_attention_norm.weight": (h,),
        "attn_q.weight": (h, q * (2 if hybrid else 1)), "attn_k.weight": (h, kv),
        "attn_v.weight": (h, kv), "attn_output.weight": (q, h),
        "attn_q.bias": (q,), "attn_k.bias": (kv,), "attn_v.bias": (kv,),
        "attn_q_norm.weight": (d,), "attn_k_norm.weight": (d,),
        "ffn_gate.weight": (h, f), "ffn_up.weight": (h, f), "ffn_down.weight": (f, h),
    }
    blocks = {}
    globals_ = []
    for tensor in gguf.tensor_directory:
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor.name)
        if match:
            index = int(match[1])
            if index < gguf.n_layer:
                blocks.setdefault(index, []).append(tensor)
        else:
            globals_.append(tensor)
    geometry_changed = bool(set(changed) & {"hidden_size", "intermediate_size", "attention_heads", "kv_heads", "head_dim", "vocabulary_size"})
    selected = list(globals_)
    cycle = int(meta.get(f"{prefix}.full_attention_interval", gguf.n_layer)) if hybrid else gguf.n_layer
    for index in range(target["layer_count"]):
        source_index = index if index < gguf.n_layer else index % cycle
        if source_index not in blocks:
            raise ValueError("原始 GGUF 缺少可用于新增层的结构模板")
        for tensor in blocks[source_index]:
            selected.append(replace(tensor, name=re.sub(r"^blk\.\d+\.", f"blk.{index}.", tensor.name)))
    packed = []
    offset = 0
    for tensor in selected:
        suffix = re.sub(r"^blk\.\d+\.", "", tensor.name)
        shape = shapes.get(suffix)
        if suffix == "rope_freqs.weight" and not hybrid:
            original_rotary = gguf.metadata.get(f"{prefix}.rope.dimension_count")
            rotary = meta.get(f"{prefix}.rope.dimension_count")
            if (type(original_rotary) is not int or original_rotary <= 0 or original_rotary % 2
                    or tensor.shape != (original_rotary // 2,)
                    or type(rotary) is not int or rotary <= 0 or rotary % 2):
                raise ValueError("RoPE 频率表必须与声明的偶数旋转维度一致")
            # GGUF stores one frequency factor per pair of rotary channels.
            shape = (rotary // 2,)
        if shape is None and hybrid:
            if suffix in {"attn_qkv.weight", "attn_gate.weight", "ssm_alpha.weight", "ssm_beta.weight"}:
                shape = (h, tensor.shape[1])
            elif suffix == "ssm_out.weight":
                shape = (tensor.shape[0], h)
            elif suffix in {"ssm_a", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_norm.weight"}:
                shape = tensor.shape
        if shape is None:
            if geometry_changed:
                raise ValueError(f"尚无 {tensor.name} 的维度变换规则，拒绝保留可能过时的张量大小")
            shape = tensor.shape
        type_name = quant if quant != "preserve" and len(shape) == 2 and not suffix.startswith("ssm_conv") else tensor.type_name
        if type_name not in _TYPE_BY_NAME:
            raise ValueError(f"不支持 {tensor.name} 的存储类型 {type_name}")
        type_id, block_size, block_bytes = _TYPE_BY_NAME[type_name]
        if shape[0] % block_size:
            raise ValueError(f"{tensor.name} 的行宽 {shape[0]} 必须能被 {type_name} 量化块 {block_size} 整除")
        byte_count = math.prod(shape) // block_size * block_bytes
        offset = (offset + 31) // 32 * 32
        packed.append(GGUFTensor(tensor.name, shape, type_id, type_name, block_size, byte_count, offset))
        offset += byte_count
    derived = replace(gguf, path="", sha256="", metadata=meta, sources=(),
        tensor_directory=tuple(packed), tensor_count=len(packed), metadata_kv_count=len(meta),
        n_layer=target["layer_count"], n_embd=h, n_head=target["attention_heads"],
        n_head_kv=target["kv_heads"], vocab_size=v, context_length=target["max_sequence_length"],
        file_type=gguf.file_type if quant == "preserve" else None,
        quantization=gguf.quantization if quant == "preserve" else f"DERIVED_{quant}")
    # The actual architecture builder verifies projection geometry and graph contracts.
    build_model_from_gguf(derived)
    return derived.as_dict(), changed


class GGUFModelCatalog(ModelCatalog):
    """Keep config-only entries visible, but never materialize them in the UI."""

    def _records(self):
        root = files("heterollm_sim").joinpath("model_preset_data", "gguf")
        manifest = json.loads(root.joinpath("manifest.json").read_text(encoding="utf-8"))
        records = {key: json.loads(root.joinpath(filename).read_text(encoding="utf-8"))
                   for key, filename in manifest.items()}
        if self.cache_dir.is_dir():
            for path in self.cache_dir.glob("*.gguf-preset.json"):
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("schema_version") != SCHEMA:
                    raise ValueError(f"GGUF 预设记录格式无效：{path.name}")
                key = record["id"]
                if key in records:
                    raise ValueError(f"GGUF 预设 ID 重复：{key}")
                records[key] = record
        return records

    def _record_metadata(self, record, base=None):
        raw = record["inventory"]
        derived = bool(record.get("changes"))
        issue = _record_execution_issue(record)
        origin = record["origin"]
        result = {**(base or {}), "id": record["id"], "name": record["name"],
            "family": record.get("family", raw["architecture"]),
            "architecture": raw["architecture"], "model_kind": (
                "moe" if "moe" in raw["architecture"] else "hybrid" if raw["architecture"] == "qwen35" else "dense"),
            "layer_count": raw["n_layer"], "vocabulary_size": raw["vocab_size"],
            "max_sequence_length": raw["context_length"], "generation_allowed": True,
            "support_level": "analytical_approximation", "source": "gguf_derived" if derived else "gguf",
            "source_status": "gguf_derived" if derived else "gguf_ready",
            "quantization": raw["quantization"], "gguf_source": deepcopy(record["origin"]),
            "source_sha": None, "config_hash": None, "source_revision": origin.get("revision", "gguf-file"),
            "provenance_status": "gguf_derived" if derived else "gguf_tensor_metadata",
            "source_repo": origin.get("repo", origin.get("filename", "")),
            "notes": ("基于 GGUF 修改的模型配置；尺寸按修改后的张量及量化块重新计算。" if derived
                      else "直接读取 GGUF 张量目录构建；不包含权重数据或模型实测耗时。") + origin.get("variant_note", ""),
            "coverage": "text_backbone_only", "limitations": ["GGUF 结构一致不代表预测耗时已通过精度验证。"],
            "architecture_evidence": {"status": "gguf_tensor_metadata", "source_type": "gguf_derived" if derived else "gguf_tensor_directory",
                "source_url": origin.get("url", origin.get("filename", "")), "notes": "保留原始文件来源；派生配置不代表原文件。" if derived else "结构、类型和权重大小来自完整 GGUF 目录。"}}
        if issue:
            result.update(generation_allowed=False, source_status=issue[0],
                          support_level="out_of_domain", notes=issue[1], limitations=[issue[1]])
        return result

    def list_metadata(self):
        records = dict(self._records())
        items = []
        for metadata in super().list_metadata():
            key = metadata["id"]
            if key in records:
                items.append(self._record_metadata(records.pop(key), metadata))
            else:
                items.append({**metadata, "generation_allowed": False, "source_status": "pending_gguf",
                    "support_level": "out_of_domain", "notes": "待补齐 GGUF：保留目录条目，绑定兼容文件后才能用于仿真。",
                    "limitations": ["当前只有公开配置，尚未绑定可执行的 GGUF 张量目录。"]})
        items.extend(self._record_metadata(record) for record in records.values())
        return tuple(sorted(items, key=lambda row: row["id"]))

    def _detail(self, record):
        base = next((item for item in super().list_metadata() if item["id"] == record["id"]), {})
        preset = self._record_metadata(record, base)
        if not preset["generation_allowed"]:
            return {"preset": preset, "model": None, "graph": None}
        model = build_model_from_gguf(inventory_to_gguf(record["inventory"]))
        metadata = {**model.metadata, "model_preset_id": record["id"], "gguf_preset_origin": deepcopy(record["origin"]),
                    "gguf_preset_changes": deepcopy(record.get("changes", {}))}
        if record.get("changes"):
            metadata["gguf_derived_inventory"] = True
        payload = to_primitive(replace(model, name=record["name"], metadata=metadata))
        return {"preset": preset, "model": payload, "graph": payload["graph"]}

    def detail(self, preset_id):
        records = self._records()
        if preset_id in records:
            return self._detail(records[preset_id])
        metadata = next((item for item in self.list_metadata() if item["id"] == preset_id), None)
        if metadata is None:
            raise KeyError(preset_id)
        return {"preset": metadata, "model": None, "graph": None}

    def configuration(self, preset_id):
        record = self._records().get(preset_id)
        if record is None:
            raise ValueError("请先绑定 GGUF，再编辑模型参数")
        issue = _record_execution_issue(record)
        if issue:
            raise ValueError(issue[1])
        return {"preset_id": preset_id, "name": record["name"],
                "parameters": _configuration(inventory_to_gguf(record["inventory"])),
                "quantizations": list(QUANTIZATIONS), "origin": record["origin"]}

    def derive(self, preset_id, name, parameters, *, save=False):
        name = self._name(name)
        source = self._records().get(preset_id)
        if source is None:
            raise ValueError("只能从已绑定 GGUF 的预设创建派生配置")
        issue = _record_execution_issue(source)
        if issue:
            raise ValueError(issue[1])
        inventory, changes = derive_inventory(source["inventory"], parameters)
        record = {**deepcopy(source), "id": "gguf-user-" + uuid4().hex[:16], "name": name,
            "inventory": inventory, "parent_preset_id": preset_id,
            "changes": {**source.get("changes", {}), **changes}}
        detail = self._detail(record)
        if detail["model"] is None:
            raise ValueError(detail["preset"]["notes"])
        if save:
            self._save(record)
        return detail

    def import_gguf(self, path, *, preset_id=None, name=None):
        current = next((item for item in super().list_metadata() if item["id"] == preset_id), None)
        if preset_id:
            if current is None or preset_id in self._records():
                raise ValueError("只能为待补齐的已有条目绑定文件；已有 GGUF 请另存为新预设")
        if name is not None:
            name = self._name(name)
        gguf = read_gguf_metadata(path)
        build_model_from_gguf(gguf)
        if preset_id:
            if (gguf.n_layer != current["layer_count"] or gguf.vocab_size != current["vocabulary_size"]
                    or gguf.architecture != current["architecture"]):
                raise ValueError("GGUF 的架构、层数或词表与所选条目不匹配；请作为新预设导入")
            patterns = self._definitions[preset_id].patterns
            if patterns:
                actual = _configuration(gguf)
                expected = patterns[0]
                for field in ("hidden_size", "intermediate_size", "attention_heads", "kv_heads"):
                    if actual[field] != getattr(expected, field):
                        raise ValueError(f"GGUF 的 {field} 与所选条目不匹配；请作为新预设导入")
                if expected.attention_head_dim and actual["head_dim"] != expected.attention_head_dim:
                    raise ValueError("GGUF 的注意力头维度与所选条目不匹配")
        origin = {"filename": Path(gguf.path).name, "sha256": gguf.sha256}
        if gguf.sources:
            origin["files"] = [{"filename": Path(source["path"]).name,
                                "size_bytes": source["size"], "sha256": source["sha256"]}
                               for source in gguf.sources]
        record = {"schema_version": SCHEMA, "id": preset_id or "gguf-user-" + uuid4().hex[:16],
            "name": self._name(name or (current["name"] if current else Path(path).stem)),
            "family": current["family"] if current else gguf.architecture,
            "origin": origin,
            "inventory": {**gguf.as_dict(), "path": Path(path).name}, "changes": {}}
        detail = self._detail(record)
        self._save(record)
        return detail

    @staticmethod
    def _name(name):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 160:
            raise ValueError("模型预设名称必须为 1–160 个字符")
        return name.strip()

    def _save(self, record):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / (record["id"] + ".gguf-preset.json")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache_dir,
                                             suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Publish a complete file atomically without replacing an existing ID.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ValueError("预设已存在，请另存为新的预设") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
