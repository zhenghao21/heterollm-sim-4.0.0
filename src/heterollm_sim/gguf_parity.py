"""Minimal GGUF metadata reader and native/simulator parity checks."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import struct
from typing import Any, BinaryIO, Mapping


class GGUFError(ValueError):
    pass


_FILE_TYPE_NAMES = {
    # llama_ftype file-type enum used by GGUF metadata, distinct from the
    # ggml tensor-type enum below.  The native 0f3a71b baseline uses 15=Q4_K_M.
    0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1",
    4: "MOSTLY_Q4_1_SOME_F16", 7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S",
    12: "MOSTLY_Q3_K_M", 13: "MOSTLY_Q3_K_L", 14: "MOSTLY_Q4_K_S",
    15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S", 17: "MOSTLY_Q5_K_M",
    18: "MOSTLY_Q6_K", 19: "MOSTLY_IQ2_XXS", 20: "MOSTLY_IQ2_XS",
    21: "MOSTLY_Q2_K_S", 22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS",
    24: "MOSTLY_IQ1_S", 25: "MOSTLY_IQ4_NL", 26: "MOSTLY_IQ3_S",
    27: "MOSTLY_IQ3_M", 28: "MOSTLY_IQ2_S", 29: "MOSTLY_IQ2_M",
    30: "MOSTLY_IQ3_M", 31: "MOSTLY_IQ1_M", 32: "MOSTLY_BF16",
    # Newer llama.cpp uses 30 for the mixed IQ3_M file type.
    30: "MOSTLY_IQ3_M",
}
_TENSOR_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    # Importance-matrix quantizers used by recent Qwen GGUF releases.
    # Block geometry follows ggml_type_traits; keeping these here allows
    # metadata/shape parity and physical-byte accounting before kernel-level
    # support is added to the planner.
    # IDs are ggml_type values (not llama file-type values).  Keep the enum
    # aligned with ggml.h: IQ3_S is 21 and IQ4_XS is 23.
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    type_name: str
    block_size: int
    n_bytes: int
    offset: int


@dataclass(frozen=True)
class GGUFMetadata:
    path: str
    sha256: str
    version: int
    tensor_count: int
    metadata_kv_count: int
    architecture: str | None
    n_layer: int | None
    n_embd: int | None
    n_head: int | None
    n_head_kv: int | None
    vocab_size: int | None
    context_length: int | None
    quantization: str | None
    file_type: int | None
    metadata: Mapping[str, Any]
    tensor_directory: tuple[GGUFTensor, ...] = ()

    @property
    def n_layer_nextn(self) -> int:
        """Auxiliary MTP layers excluded from the executable trunk graph."""
        if not self.architecture:
            return 0
        try:
            return max(0, int(self.metadata.get(f"{self.architecture}.nextn_predict_layers", 0)))
        except (TypeError, ValueError):
            return 0

    @property
    def n_layer_all(self) -> int | None:
        return self.n_layer + self.n_layer_nextn if self.n_layer is not None else None

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "version": self.version,
                "tensor_count": self.tensor_count, "metadata_kv_count": self.metadata_kv_count,
                "architecture": self.architecture, "n_layer": self.n_layer,
                "n_layer_all": self.n_layer_all, "n_layer_nextn": self.n_layer_nextn,
                "n_embd": self.n_embd, "n_head": self.n_head, "n_head_kv": self.n_head_kv,
                "vocab_size": self.vocab_size, "context_length": self.context_length,
                "quantization": self.quantization, "file_type": self.file_type,
                "metadata": dict(self.metadata),
                "tensor_directory": [t.__dict__ for t in self.tensor_directory]}


def _read_string(f: BinaryIO) -> str:
    raw = f.read(8)
    if len(raw) != 8:
        raise GGUFError("truncated GGUF string length")
    (length,) = struct.unpack("<Q", raw)
    if length > 16 * 1024 * 1024:
        raise GGUFError("GGUF string exceeds 16 MiB safety limit")
    value = f.read(length)
    if len(value) != length:
        raise GGUFError("truncated GGUF string")
    return value.decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, value_type: int) -> Any:
    formats = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if value_type in formats:
        size = struct.calcsize(formats[value_type])
        raw = f.read(size)
        if len(raw) != size:
            raise GGUFError("truncated GGUF metadata value")
        return struct.unpack(formats[value_type], raw)[0]
    if value_type == 8:
        return _read_string(f)
    if value_type == 9:
        raw = f.read(12)
        if len(raw) != 12:
            raise GGUFError("truncated GGUF array header")
        subtype, count = struct.unpack("<IQ", raw)
        if count > 10_000_000:
            raise GGUFError("GGUF metadata array exceeds safety limit")
        # Keep exported metadata bounded; token types/scores can contain one
        # entry per vocabulary item and are not needed for parity.
        if count > 4096:
            for _ in range(count):
                _read_value(f, subtype)
            return {"count": count, "truncated": True}
        if subtype == 8:
            for _ in range(count):
                _read_string(f)
            return {"count": count}
        return [_read_value(f, subtype) for _ in range(count)]
    raise GGUFError(f"unsupported GGUF metadata type {value_type}")


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def read_gguf_metadata(path: str | Path) -> GGUFMetadata:
    p = Path(path)
    with p.open("rb") as f:
        head = f.read(24)
        if len(head) != 24 or head[:4] != b"GGUF":
            raise GGUFError(f"not a GGUF file: {p}")
        version, tensor_count, kv_count = struct.unpack("<IQQ", head[4:])
        metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(f)
            type_raw = f.read(4)
            if len(type_raw) != 4:
                raise GGUFError("truncated GGUF metadata type")
            metadata[key] = _read_value(f, struct.unpack("<I", type_raw)[0])
        alignment = _as_int(metadata.get("general.alignment")) or 32
        directory = []
        names_seen: set[str] = set()
        for _ in range(int(tensor_count)):
            name = _read_string(f)
            if name in names_seen:
                raise GGUFError(f"duplicate GGUF tensor name: {name}")
            names_seen.add(name)
            raw = f.read(4)
            if len(raw) != 4:
                raise GGUFError("truncated GGUF tensor rank")
            (rank,) = struct.unpack("<I", raw)
            if rank > 8:
                raise GGUFError("invalid GGUF tensor rank")
            dims_raw = f.read(8 * rank)
            type_raw = f.read(4); offset_raw = f.read(8)
            if len(dims_raw) != 8 * rank or len(type_raw) != 4 or len(offset_raw) != 8:
                raise GGUFError("truncated GGUF tensor directory")
            dims = tuple(int(x) for x in struct.unpack("<" + "Q" * rank, dims_raw))
            type_id = struct.unpack("<I", type_raw)[0]
            offset = struct.unpack("<Q", offset_raw)[0]
            spec = _TENSOR_TYPES.get(type_id)
            if spec is None:
                raise GGUFError(f"unsupported GGUF tensor type {type_id}")
            type_name, block_size, block_bytes = spec
            elements = 1
            for dim in dims: elements *= dim
            if block_size > 1 and elements % block_size:
                raise GGUFError(
                    f"GGUF tensor {name} has {elements} elements, not divisible by {block_size}"
                )
            n_bytes = (elements // block_size) * block_bytes
            directory.append(GGUFTensor(name, dims, type_id, type_name, block_size, n_bytes, offset))
        data_start = ((f.tell() + alignment - 1) // alignment) * alignment
        file_size = p.stat().st_size
        for tensor in directory:
            if data_start + tensor.offset + tensor.n_bytes > file_size:
                raise GGUFError(f"truncated GGUF tensor payload: {tensor.name}")
    digest = sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    arch = metadata.get("general.architecture")
    prefix = str(arch) if arch else ""
    pick = lambda suffix: metadata.get(f"{prefix}.{suffix}") if prefix else None
    vocab = metadata.get("tokenizer.ggml.tokens")
    if isinstance(vocab, dict):
        vocab = vocab.get("count")
    elif vocab is None:
        vocab = metadata.get("general.vocab_size")
    file_type = metadata.get("general.file_type")
    raw_layers = _as_int(pick("block_count"))
    # llama.cpp exposes n_layer() as n_layer_all - n_layer_nextn.  MTP/nextn
    # blocks are auxiliary prediction heads and must not enter the simulator's
    # decoder graph or timing geometry.
    nextn = _as_int(pick("nextn_predict_layers")) or 0
    trunk_layers = raw_layers - nextn if raw_layers is not None else None
    if trunk_layers is not None and trunk_layers <= 0:
        raise GGUFError("GGUF block_count is not larger than nextn_predict_layers")
    return GGUFMetadata(str(p.resolve()), digest.hexdigest(), int(version), int(tensor_count), int(kv_count),
                        str(arch) if arch is not None else None, trunk_layers,
                        _as_int(pick("embedding_length")), _as_int(pick("attention.head_count")),
                        _as_int(pick("attention.head_count_kv")), _as_int(vocab), _as_int(pick("context_length")),
                        _FILE_TYPE_NAMES.get(int(file_type)) if file_type is not None else None,
                        _as_int(file_type), metadata, tuple(directory))


def _expected_geometry(model: Any) -> dict[str, int | str | None]:
    from .ir import model_graph_execution_view
    view = model_graph_execution_view(model.graph, schema_version=model.schema_version)
    layer = view.layer_instances[0].layer
    return {"architecture": view.architecture, "n_layer": len(view.layer_instances), "n_embd": layer.hidden_size,
            "n_head": layer.attention_heads, "n_head_kv": layer.effective_kv_heads,
            "vocab_size": view.vocabulary_size, "context_length": view.max_sequence_length}


def compare_gguf_to_model(gguf: GGUFMetadata, model: Any, *, context_length: int | None = None,
                          expected_prompt_tokens: int | None = None, actual_prompt_tokens: int | None = None,
                          expected_output_tokens: int | None = None, actual_output_tokens: int | None = None) -> dict[str, Any]:
    expected = _expected_geometry(model)
    mismatches: list[dict[str, Any]] = []
    family = str(getattr(model, "metadata", {}).get("family", "")).lower()
    if not gguf.architecture:
        mismatches.append({"field": "architecture", "expected": family or expected["architecture"], "actual": None})
    elif family and not any(part in gguf.architecture.lower() for part in family.replace(".", " ").split()):
        # Generic graph architecture is accepted; concrete family mismatches fail closed.
        if not (family.startswith("qwen") and gguf.architecture.lower().startswith("qwen")):
            mismatches.append({"field": "architecture", "expected": family or expected["architecture"], "actual": gguf.architecture})
    for field in ("n_layer", "n_embd", "n_head", "n_head_kv", "vocab_size"):
        actual, want = getattr(gguf, field), expected[field]
        if actual != want:
            mismatches.append({"field": field, "expected": want, "actual": actual})
    # Runtime context is a limit; GGUF context_length is the model maximum.
    # A smaller runtime context is valid, while exceeding the model maximum
    # must fail closed.
    if context_length is not None and gguf.context_length is not None and context_length > gguf.context_length:
        mismatches.append({"field": "context_length", "expected_max": gguf.context_length, "actual": context_length})
    for field, want, actual in (("prompt_tokens", expected_prompt_tokens, actual_prompt_tokens), ("output_tokens", expected_output_tokens, actual_output_tokens)):
        if want is not None and actual is not None and want != actual:
            mismatches.append({"field": field, "expected": want, "actual": actual})
    return {"ok": not mismatches, "status": "pass" if not mismatches else "fail", "mismatches": mismatches,
            "expected": expected, "gguf": gguf.as_dict()}


def assert_gguf_parity(report: Mapping[str, Any]) -> None:
    if not report.get("ok"):
        fields = ", ".join(str(item.get("field")) for item in report.get("mismatches", ()))
        raise ValueError(f"GGUF/native parity gate failed: {fields or 'unknown mismatch'}")


def build_model_from_gguf(gguf: GGUFMetadata):
    """Bind a dense Qwen2/Llama GGUF directory to an executable ModelSpec.

    The graph keeps training context as the model limit; runtime ``-c`` is a
    serving setting and is deliberately not copied into ``max_sequence_length``.
    """
    from .ir import LayerSpec, ModelSpec, build_model_graph_from_layer_specs
    import re
    if gguf.architecture not in {"qwen2", "llama", "qwen35"}:
        raise GGUFError(f"unsupported GGUF architecture: {gguf.architecture}")
    required = (gguf.n_layer, gguf.n_embd, gguf.n_head, gguf.n_head_kv, gguf.vocab_size, gguf.context_length)
    if any(value is None or value <= 0 for value in required):
        raise GGUFError("GGUF is missing required model geometry")
    by_layer: dict[int, list[GGUFTensor]] = {}
    for tensor in gguf.tensor_directory:
        match = re.search(r"(?:blk|layers)\.(\d+)\.", tensor.name)
        if match:
            by_layer.setdefault(int(match.group(1)), []).append(tensor)
    if not set(range(gguf.n_layer)).issubset(by_layer):
        raise GGUFError("GGUF tensor directory does not contain every decoder layer")

    def binding(t: GGUFTensor) -> dict[str, Any]:
        return {"name": t.name, "shape": list(t.shape), "type": t.type_name,
                "block_size": t.block_size, "n_bytes": t.n_bytes, "offset": t.offset}
    layers = []
    # Qwen3.5/3.8 GGUFs encode a 3-linear/1-full hybrid sequence.  The
    # metadata also carries the final ``nextn`` block; llama.cpp reports it in
    # block_count, so retain it as an executable full-attention layer for
    # geometry parity while recording the auxiliary tensors in bindings.
    is_qwen35 = gguf.architecture == "qwen35"
    linear_attention = None
    if is_qwen35:
        from .ir import LinearAttentionSpec
        md = gguf.metadata
        linear_attention = LinearAttentionSpec(
            key_heads=int(md.get("qwen35.ssm.group_count") or 16),
            value_heads=max(1, int((md.get("qwen35.ssm.inner_size") or 6144) // (md.get("qwen35.ssm.state_size") or 128))),
            key_head_dim=128,
            value_head_dim=int(md.get("qwen35.ssm.state_size") or 128),
            conv_kernel_size=int(md.get("qwen35.ssm.conv_kernel") or 4),
            state_dtype="fp32",
            output_gate=True,
            gate_activation="silu",
        )
    for index in range(gguf.n_layer):
        tensors = by_layer[index]
        names = {t.name.lower(): t for t in tensors}
        def find(*parts):
            # Match the projection component as a path component.  A loose
            # substring search makes ``attn_q`` accidentally select
            # ``attn_q_norm`` on directory order changes.
            for part in parts:
                suffix = part.lower()
                for t in tensors:
                    name = t.name.lower()
                    if name.endswith(suffix + ".weight") or name.endswith(suffix):
                        return t
            return None
        q, k, v = find("attn_q", "attention.wq"), find("attn_k", "attention.wk"), find("attn_v", "attention.wv")
        qkv = find("attn_qkv")
        o = find("attn_output", "attention.wo", "self_attn.o_proj")
        ssm_out = find("ssm_out")
        gate, up, down = find("ffn_gate", "mlp.gate_proj"), find("ffn_up", "mlp.up_proj"), find("ffn_down", "mlp.down_proj")
        if is_qwen35 and qkv is not None and ssm_out is not None:
            # Linear-attention blocks use a fused QKV input projection and an
            # SSM output projection instead of separate Q/K/V tensors.
            q = k = v = qkv
            o = ssm_out
        if not all((q, k, v, o, down)) or (gate is None and up is None):
            raise GGUFError(f"layer {index} is missing QKV/O/FFN tensors")
        def extent(t):
            if len(t.shape) != 2: raise GGUFError(f"tensor {t.name} is not a matrix")
            # GGUF stores matrix dimensions as [input(K), output(N)].
            return int(t.shape[0]), int(t.shape[1])
        hidden = int(gguf.n_embd)
        # GGUF matrices are stored as [input (K), output (N)].  The gated/up
        # projection therefore exposes the FFN width on dimension 1; using
        # dimension 0 silently collapsed every imported model's MLP to the
        # hidden size and made simulator timings dramatically too small.
        intermediate = int((gate or up).shape[1])
        metadata: dict[str, Any] = {"gguf_tensor_bindings": [binding(t) for t in tensors], "gguf_physical_weight_bytes": sum(t.n_bytes for t in tensors)}
        projections: dict[str, Any] = {}
        def add_projection(pid, ts, shard_axis="n", segment_ids=None):
            segments = []
            for j, t in enumerate(ts):
                if t.type_name not in {"Q4_K", "Q5_K", "Q6_K", "Q4_0", "Q5_0", "Q8_0", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS", "IQ2_S", "IQ1_S", "IQ1_M"}:
                    return
                kk, nn = extent(t)
                segment_id = (tuple(segment_ids)[j] if segment_ids is not None else f"{pid.replace('.', '-')}-{j}")
                segments.append({"segment_id": segment_id, "physical_tensor_name": t.name, "k": kk, "n": nn, "format": t.type_name, "physical_bytes": t.n_bytes, "tp_shard_axis": shard_axis})
            if segments: projections[pid] = {"segments": segments}
        if is_qwen35 and q is k is v:
            add_projection("linear_attention.qkv", (q,))
            add_projection("linear_attention.output", (o,), "k")
            gate_tensor = find("attn_gate")
            if gate_tensor is not None:
                add_projection("linear_attention.output_gate", (gate_tensor,))
        else:
            add_projection("attention.qkv", (q, k, v), segment_ids=("q", "k", "v")); add_projection("attention.output", (o,), "k", segment_ids=("output",))
            if is_qwen35:
                # llama.cpp's Qwen3.5 full-attention graph uses q/k RMS norms,
                # a sigmoid query gate, and a 64-d rotary section.
                q_width = int(gguf.n_head) * 256
                metadata["attention_execution_descriptor"] = {
                    "schema_version": "heterollm.attention-execution/v1",
                    "query_heads": int(gguf.n_head), "kv_heads": int(gguf.n_head_kv),
                    "head_dim": 256, "query_width": q_width, "gate_width": q_width,
                    "q_projection_width": 2 * q_width, "rotary_dim": int(gguf.metadata.get("qwen35.rope.dimension_count") or 64),
                    "qk_scale": 1.0 / (256.0 ** 0.5), "qk_norm": True,
                    "gate_activation": "sigmoid",
                }
        if is_qwen35 and q is k is v:
            # Gated-delta-net alpha/beta projections are real matrix multiplies;
            # omit scalar state vectors (ssm_a/dt) from weight projections.
            alpha = find("ssm_alpha")
            beta = find("ssm_beta")
            if alpha is not None: add_projection("linear_attention.alpha", (alpha,))
            if beta is not None: add_projection("linear_attention.beta", (beta,))
        if gate and up:
            # Keep both views: ``mlp.up_gate`` describes a fused logical
            # workload, while llama.cpp's CPU graph executes the two physical
            # matrices as separate MUL_MAT invocations.  Exposing the
            # single-segment descriptors lets a runtime with explicit
            # physical-call evidence lower those invocations without guessing
            # or splitting a byte count heuristically.
            add_projection("mlp.gate", (gate,), segment_ids=("mlp-up_gate-0",))
            add_projection("mlp.up", (up,), segment_ids=("mlp-up_gate-1",))
            add_projection("mlp.up_gate", (gate, up),
                           segment_ids=("mlp-up_gate-0", "mlp-up_gate-1"))
        elif up: add_projection("mlp.up", (up,))
        add_projection("mlp.down", (down,))
        if projections:
            metadata["weight_projection_descriptors"] = {"schema_version": "heterollm.weight-projections/v1", "projections": projections}
        if is_qwen35:
            mixer = "full_attention" if qkv is None else "linear_attention"
            la = linear_attention if mixer == "linear_attention" else None
            metadata["qwen35_block_kind"] = mixer
            metadata["qwen35_full_attention_interval"] = gguf.metadata.get("qwen35.full_attention_interval")
        else:
            mixer, la = "full_attention", None
        layers.append(LayerSpec(layer_id=f"layer-{index:03d}", kind="dense", hidden_size=hidden, intermediate_size=intermediate,
                                attention_heads=int(gguf.n_head), kv_heads=int(gguf.n_head_kv), attention_head_dim=(256 if is_qwen35 else 0),
                                sequence_mixer=mixer, linear_attention=la, dtype="fp16", weight_bytes=sum(t.n_bytes for t in tensors), metadata=metadata))
    embedding = next((t for t in gguf.tensor_directory if "token_embd.weight" in t.name or "embed_tokens.weight" in t.name), None)
    if embedding is None: raise GGUFError("GGUF embedding tensor is missing")
    output = next((t for t in gguf.tensor_directory if t.name.lower() in {"output.weight", "lm_head.weight"}), None)
    # Several tied-embedding exports (including Qwen3.5 GGUFs from Unsloth)
    # omit a separate output/lm_head tensor.  llama.cpp reuses token_embd for
    # the final projection in that case, so bind the same physical tensor
    # instead of rejecting an otherwise valid model.
    output_tied_to_embedding = output is None
    if output_tied_to_embedding:
        output = embedding
    graph_architecture = "qwen3_5_hybrid_transformer" if is_qwen35 else gguf.architecture
    graph = build_model_graph_from_layer_specs("GGUF-" + gguf.architecture, layers, architecture=graph_architecture,
        vocabulary_size=int(gguf.vocab_size), max_sequence_length=int(gguf.context_length), embedding_weight_bytes=embedding.n_bytes,
        output_weight_bytes=output.n_bytes,
        # Keep the file-level label under an audit-only key.  The planner's
        # artifact dispatcher consumes per-segment formats from
        # weight_projection_descriptors; feeding a mixed file label such as
        # Q4_K_M into its single-format registry would incorrectly reject the
        # otherwise valid mixed tensor graph.
        metadata={"gguf_sha256": gguf.sha256, "gguf_file_quantization": gguf.quantization, "gguf_tensor_count": len(gguf.tensor_directory),
                  "gguf_output_tied_to_embedding": output_tied_to_embedding,
                  # Keep graph-level bindings compact; tensor directory owns
                  # block geometry and the planner must not interpret these
                  # audit fields as one global artifact format.
                  "gguf_embedding_binding": {k: v for k, v in binding(embedding).items() if k != "block_size"},
                  "gguf_output_binding": {k: v for k, v in binding(output).items() if k != "block_size"}})
    return ModelSpec(name="GGUF-" + gguf.architecture, graph=graph, metadata=graph.attributes)


__all__ = ["GGUFError", "GGUFTensor", "GGUFMetadata", "read_gguf_metadata", "compare_gguf_to_model", "assert_gguf_parity", "build_model_from_gguf"]
