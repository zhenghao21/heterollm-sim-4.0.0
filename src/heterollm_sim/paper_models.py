"""Text-model work for the Exploring HBF comparison, independent of native IR.

These descriptors are consumed by the paper's operation-granularity scheduler.
They are not an assertion that the old native Maverick graph is implemented.
FMA counts as two FLOPs. ``context_tokens`` always means cached tokens BEFORE
the current decode token. Chunked attention is fixed, aligned chunks, not a
sliding window. Routing is an explicit uniform, independent top-k expectation,
not a checkpoint trace and not an instruction to load every expert per token.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, expm1, log1p
from typing import Mapping


@dataclass(frozen=True)
class PaperPrecision:
    """Paper does not identify dtypes: BF16 is a disclosed research choice.

    Quantization scale traffic is deliberately unsupported. An FP8 checkpoint
    cannot be represented merely by reducing this object's element size.
    """

    weight_dtype: str = "BF16"
    kv_dtype: str = "BF16"
    activation_dtype: str = "BF16"
    accumulator_dtype: str = "FP32"
    basis: str = "Explicit BF16 research choice; not a dtype reported by the paper."

    def __post_init__(self) -> None:
        for value in (self.weight_dtype, self.kv_dtype, self.activation_dtype):
            if value not in {"BF16", "FP16", "FP32"}:
                raise ValueError("Only unquantized BF16/FP16/FP32 are modeled; scale traffic is required for FP8/INT.")
        if self.accumulator_dtype != "FP32":
            raise ValueError("Only explicit FP32 accumulation is supported.")

    @property
    def weight_bytes(self) -> int:
        return 4 if self.weight_dtype == "FP32" else 2

    @property
    def kv_bytes(self) -> int:
        return 4 if self.kv_dtype == "FP32" else 2

    @property
    def activation_bytes(self) -> int:
        return 4 if self.activation_dtype == "FP32" else 2


def expected_experts_touched(tokens: int, experts: int, top_k: int = 1) -> float:
    """Distinct experts used by a batch; top-k draws within a token are unique."""
    if tokens < 0 or experts < 1 or not 1 <= top_k <= experts:
        raise ValueError("tokens >= 0 and 1 <= top_k <= experts are required")
    if tokens == 0:
        return 0.0
    if top_k == experts:
        return float(experts)
    return -experts * expm1(tokens * log1p(-top_k / experts))


@dataclass(frozen=True)
class LayerWork:
    """Per physical GPU work for one layer in one EP group's decode step.

    Attention and dense/shared FFN are replicated over EP, tensor-sharded over
    TP. Routed experts are sharded over EP and then tensor-sharded over TP.
    ``batch_size`` is the whole EP-group batch, hence each EP rank originates
    ``local_batch_size`` tokens. PP selects layers; it never divides a layer.
    Matrix FLOPs exclude implementation-dependent pointwise/softmax operations.
    ``live_buffers`` names logical tensors for a fused attention schedule;
    ``intermediate_live_bytes`` applies attention/FFN phase lifetimes. No full
    sequence-by-head score matrix is materialized.
    """

    layer_index: int
    attention_kind: str
    ffn_kind: str
    batch_size: int
    local_batch_size: int
    context_tokens: int
    attended_tokens: int
    local_query_heads: int
    local_kv_heads: int
    expected_local_experts_touched: float
    weights_read_bytes: float
    attention_weight_bytes: int
    dense_weight_bytes: int
    shared_weight_bytes: int
    routed_weight_bytes: float
    router_weight_bytes: int
    norm_weight_bytes: int
    kv_read_bytes: int
    kv_write_bytes: int
    kv_resident_bytes: int
    compute_flops: float
    attention_projection_flops: int
    attention_flops: int
    dense_flops: int
    shared_flops: int
    routed_flops: float
    router_flops: int
    live_buffers: tuple[tuple[str, int], ...]
    expert_dispatch_bytes: float

    @property
    def intermediate_live_bytes(self) -> int:
        sizes = dict(self.live_buffers)
        common = sizes["residual"] + sizes["normalized_input"] + sizes["attention_output"]
        attention = sizes["query"] + sizes["new_key_value"] + sizes["attention_online_stats"]
        feed_forward = sizes["ffn_gate_up"] + sizes["router_logits"]
        return common + max(attention, feed_forward)

    @property
    def attention_compute_flops(self) -> int:
        return self.attention_projection_flops + self.attention_flops

    @property
    def ffn_compute_flops(self) -> float:
        return self.dense_flops + self.shared_flops + self.routed_flops + self.router_flops


@dataclass(frozen=True)
class PaperModelSpec:
    model_id: str
    name: str
    layers: int
    hidden_size: int
    intermediate_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    vocabulary_size: int
    max_context_tokens: int
    tie_word_embeddings: bool
    expert_intermediate_size: int = 0
    routed_experts: int = 0
    top_k: int = 0
    shared_experts: int = 0
    moe_layer_step: int = 0
    full_attention_interval: int = 1
    attention_chunk_size: int = 0
    official_total_parameters_rounded: int = 0
    official_active_parameters_rounded: int = 0
    sources: tuple[str, ...] = ()
    geometry_verification: str = "official_implementation"
    checkpoint_geometry_verified: bool = False
    limitations: tuple[str, ...] = ()
    paper_reported_weight_capacity_gb: int | None = None

    def is_moe(self, layer_index: int) -> bool:
        self._check_layer(layer_index)
        return bool(self.moe_layer_step and (layer_index + 1) % self.moe_layer_step == 0)

    def is_full_attention(self, layer_index: int) -> bool:
        self._check_layer(layer_index)
        return not self.attention_chunk_size or (layer_index + 1) % self.full_attention_interval == 0

    def _check_layer(self, layer_index: int) -> None:
        if not isinstance(layer_index, int) or not 0 <= layer_index < self.layers:
            raise ValueError("layer_index outside model")

    def validate_parallelism(self, tp: int = 1, ep: int = 1) -> None:
        if tp < 1 or ep < 1 or self.query_heads % tp:
            raise ValueError("TP must divide query heads; EP and TP must be positive")
        if tp <= self.kv_heads:
            if self.kv_heads % tp:
                raise ValueError("TP must divide KV heads when sharding GQA")
        elif tp % self.kv_heads:
            raise ValueError("TP > KV heads must be an integral KV replication factor")
        if self.intermediate_size % tp or (self.expert_intermediate_size and self.expert_intermediate_size % tp):
            raise ValueError("TP must divide dense/expert FFN width")
        if self.vocabulary_size % tp:
            raise ValueError("TP must divide vocabulary for exact vocab sharding")
        if self.routed_experts and self.routed_experts % ep:
            raise ValueError("EP must divide routed experts")
        if not self.routed_experts and ep != 1:
            raise ValueError("Dense models use DP, not EP")

    def local_kv_head_count(self, tp: int = 1) -> int:
        self.validate_parallelism(tp)
        return max(1, self.kv_heads // tp)

    @property
    def attention_parameters_per_layer(self) -> int:
        h, q, k = self.hidden_size, self.query_heads * self.head_dim, self.kv_heads * self.head_dim
        return 2 * h * q + 2 * h * k

    @property
    def dense_ffn_parameters(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def expert_parameters(self) -> int:
        return 3 * self.hidden_size * self.expert_intermediate_size

    def layer_parameters(self, layer_index: int, *, active: bool = False) -> int:
        base = self.attention_parameters_per_layer + 2 * self.hidden_size
        if not self.is_moe(layer_index):
            return base + self.dense_ffn_parameters
        expert_count = self.top_k if active else self.routed_experts
        return base + (expert_count + self.shared_experts) * self.expert_parameters + self.hidden_size * self.routed_experts

    @property
    def total_parameters(self) -> int:
        """Exact text-only tensor count. Vision/projector are intentionally absent."""
        embeddings = self.hidden_size * self.vocabulary_size * (1 if self.tie_word_embeddings else 2)
        return embeddings + self.hidden_size + sum(self.layer_parameters(i) for i in range(self.layers))

    @property
    def active_parameters(self) -> int:
        """Conventional active count includes full embedding/head resident tables."""
        embeddings = self.hidden_size * self.vocabulary_size * (1 if self.tie_word_embeddings else 2)
        return embeddings + self.hidden_size + sum(self.layer_parameters(i, active=True) for i in range(self.layers))

    def layer_indices(self, pp: int = 1, stage: int = 0) -> range:
        if not 1 <= pp <= self.layers or not 0 <= stage < pp:
            raise ValueError("PP must partition the model into nonempty stages")
        return range(stage * self.layers // pp, (stage + 1) * self.layers // pp)

    def stage_weight_bytes(self, *, tp: int = 1, ep: int = 1, pp: int = 1,
                           stage: int = 0, precision: PaperPrecision | None = None) -> int:
        """Resident weights per GPU, including unused local experts and norms.

        A tied table is shared only when embedding/head live on the same PP
        stage. Distinct first/last stages each require a private physical copy.
        Router and layer norm parameters are replicated over TP and EP.
        """
        self.validate_parallelism(tp, ep)
        precision = precision or PaperPrecision()
        h = self.hidden_size
        local_attn = 2 * h * self.query_heads * self.head_dim // tp + 2 * h * self.local_kv_head_count(tp) * self.head_dim
        parameters = 0
        for layer in self.layer_indices(pp, stage):
            parameters += local_attn + 2 * h
            if self.is_moe(layer):
                parameters += self.expert_parameters * (self.routed_experts // ep + self.shared_experts) // tp
                parameters += h * self.routed_experts
            else:
                parameters += self.dense_ffn_parameters // tp
        if stage == 0:
            parameters += h * self.vocabulary_size // tp
        if stage == pp - 1:
            if not (pp == 1 and self.tie_word_embeddings):
                parameters += h * self.vocabulary_size // tp
            parameters += h
        return parameters * precision.weight_bytes

    def kv_bytes_per_token(self, *, tp: int = 1, pp: int = 1, stage: int = 0,
                           precision: PaperPrecision | None = None) -> int:
        """Uncompressed K+V size before chunk eviction, for one sequence/GPU."""
        precision = precision or PaperPrecision()
        return len(self.layer_indices(pp, stage)) * 2 * self.local_kv_head_count(tp) * self.head_dim * precision.kv_bytes

    def layer_work(self, layer_index: int, batch_size: int, context_tokens: int, *,
                   tp: int = 1, ep: int = 1, precision: PaperPrecision | None = None) -> LayerWork:
        self._check_layer(layer_index)
        self.validate_parallelism(tp, ep)
        if batch_size < 1 or batch_size % ep:
            raise ValueError("The EP-group batch must be positive and divisible by EP")
        if context_tokens < 0 or context_tokens >= self.max_context_tokens:
            raise ValueError("Context must leave space for the current token")
        p = precision or PaperPrecision()
        h, local_batch = self.hidden_size, batch_size // ep
        qheads, kvheads = self.query_heads // tp, self.local_kv_head_count(tp)
        full = self.is_full_attention(layer_index)
        old_tokens = context_tokens if full else context_tokens % self.attention_chunk_size
        attended = old_tokens + 1
        attention_params = 2 * h * qheads * self.head_dim + 2 * h * kvheads * self.head_dim
        dense_params = shared_params = router_params = 0
        routed_params = touched = 0.0
        dense_flops = shared_flops = router_flops = 0
        routed_flops = dispatch = 0.0
        if self.is_moe(layer_index):
            touched = expected_experts_touched(batch_size, self.routed_experts, self.top_k) / ep
            routed_params = self.expert_parameters * touched / tp
            shared_params = self.expert_parameters * self.shared_experts // tp
            router_params = h * self.routed_experts
            shared_flops = 2 * local_batch * shared_params
            routed_flops = 2 * batch_size * self.top_k * self.expert_parameters / (ep * tp)
            router_flops = 2 * local_batch * router_params
            # Each EP origin sends top-k activations and receives expert outputs.
            # Balanced independent expert ownership => (EP-1)/EP are remote.
            dispatch = 2 * local_batch * self.top_k * h * p.activation_bytes * (ep - 1) / ep
        else:
            dense_params = self.dense_ffn_parameters // tp
            dense_flops = 2 * local_batch * dense_params
        attn_projection = 2 * local_batch * attention_params
        attention_flops = 4 * local_batch * qheads * self.head_dim * attended
        kv_token_bytes = 2 * local_batch * kvheads * self.head_dim * p.kv_bytes
        # Live buffers are a fused-attention / sequential FFN tile, not KV
        # residency or model weights. FFN gate+up coexist before SwiGLU; routed
        # and shared paths are scheduled serially to reuse the same workspace.
        ffn_width = self.expert_intermediate_size if self.is_moe(layer_index) else self.intermediate_size
        buffers = (("residual", local_batch * h * p.activation_bytes),
                   ("normalized_input", local_batch * h * p.activation_bytes),
                   ("query", local_batch * qheads * self.head_dim * p.activation_bytes),
                   ("new_key_value", kv_token_bytes),
                   ("attention_output", local_batch * h * p.activation_bytes),
                   ("ffn_gate_up", 2 * local_batch * ffn_width // tp * p.activation_bytes),
                   ("router_logits", local_batch * self.routed_experts * 4 if self.is_moe(layer_index) else 0),
                   ("attention_online_stats", 2 * local_batch * qheads * 4))
        weight_bytes = (attention_params + dense_params + shared_params + routed_params + router_params + 2 * h) * p.weight_bytes
        return LayerWork(layer_index, "full" if full else "chunked", "moe" if self.is_moe(layer_index) else "dense",
                         batch_size, local_batch, context_tokens, attended, qheads, kvheads, touched,
                         weight_bytes, attention_params * p.weight_bytes, dense_params * p.weight_bytes,
                         shared_params * p.weight_bytes, routed_params * p.weight_bytes,
                         router_params * p.weight_bytes, 2 * h * p.weight_bytes,
                         old_tokens * kv_token_bytes, kv_token_bytes, attended * kv_token_bytes,
                         attn_projection + attention_flops + dense_flops + shared_flops + routed_flops + router_flops,
                         attn_projection, attention_flops, dense_flops, shared_flops, routed_flops, router_flops,
                         buffers, dispatch)

    def endpoint_work(self, batch_size: int, *, tp: int = 1, ep: int = 1,
                      precision: PaperPrecision | None = None) -> dict[str, int]:
        """Embedding lookup, final RMSNorm and complete vocabulary projection."""
        self.validate_parallelism(tp, ep)
        if batch_size < 1 or batch_size % ep:
            raise ValueError("The EP-group batch must be positive and divisible by EP")
        p, b = precision or PaperPrecision(), batch_size // ep
        # An independent uniform token assumption only reduces lookup row IO;
        # output projection still reads the complete vocab shard per step.
        rows = expected_experts_touched(b, self.vocabulary_size)
        return {"embedding_read_bytes": ceil(rows * self.hidden_size * p.weight_bytes / tp),
                "head_read_bytes": self.hidden_size * self.vocabulary_size * p.weight_bytes // tp,
                "final_norm_read_bytes": self.hidden_size * p.weight_bytes,
                "head_flops": 2 * b * self.hidden_size * self.vocabulary_size // tp,
                "logits_bytes": b * self.vocabulary_size * 4 // tp}

    def metadata(self, precision: PaperPrecision | None = None) -> dict:
        p = precision or PaperPrecision()
        weight_bytes = self.total_parameters * p.weight_bytes
        data = {**asdict(self), "precision": asdict(p),
                "text_total_parameters": self.total_parameters,
                "text_active_parameters": self.active_parameters,
                "text_weight_bytes": weight_bytes,
                "text_weight_decimal_gb": weight_bytes / 1_000_000_000,
                "text_weight_gib": weight_bytes / 1024**3,
                "routing_basis": "Uniform independent token routing, unique top-k within each token; expected distinct experts per batch.",
                "execution_backend": "paper_operation_event_model_not_native_ir",
                "compute_basis": "Matrix and attention multiply-accumulate FLOPs only; FMA=2; fused attention does not materialize full scores.",
                "parallelism_basis": "Independent TP*EP*PP axes; dense/attention/shared replicated over EP; router/norm replicated over TP."}
        if self.paper_reported_weight_capacity_gb is not None:
            reported = self.paper_reported_weight_capacity_gb
            data["paper_capacity_unit_sensitivity"] = {
                "reported_value": reported, "reported_unit": "GB",
                "explicit_options": {"decimal_GB_bytes": reported * 1_000_000_000,
                                     "binary_GiB_bytes": reported * 1024**3},
                "default_weight_bytes_changed": False,
                "basis": "Reported rounded capacity is ambiguous; GiB proximity is a consistency clue, not geometry verification. Tensor-derived bytes remain authoritative; unit interpretation must be a separately labeled sensitivity run."}
        return data


_META = "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/"
_HF = "https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/llama4/"
_TORCHTUNE = "https://raw.githubusercontent.com/meta-pytorch/torchtune/main/torchtune/models/llama4/"

LLAMA3_405B = PaperModelSpec(
    "llama3_405b", "Llama 3.1 405B", 126, 16384, 53248, 128, 8, 128, 128256, 131072, False,
    official_total_parameters_rounded=405_000_000_000, official_active_parameters_rounded=405_000_000_000,
    sources=(_META + "sku_list.py", _META + "llama3/model.py",
             "https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct"),
    geometry_verification="Official Meta 405B bf16-mp8 SKU and FFN rounding formula; not mp16 duplicated KV.",
    limitations=("Paper calls this Llama3 405B; released Llama3.1 405B BF16 mp8 geometry is used explicitly.",
                 "Checkpoint config is gated and was not independently downloaded; Meta official SKU provides dimensions."))

LLAMA4_MAVERICK = PaperModelSpec(
    "llama4_maverick", "Llama 4 Maverick 17B 128E (text)", 48, 5120, 16384, 40, 8, 128, 202048, 1048576, False,
    expert_intermediate_size=8192, routed_experts=128, top_k=1, shared_experts=1,
    moe_layer_step=2, full_attention_interval=4, attention_chunk_size=8192,
    official_total_parameters_rounded=400_000_000_000, official_active_parameters_rounded=17_000_000_000,
    sources=("https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct",
             _TORCHTUNE + "_model_builders.py", _TORCHTUNE + "_component_builders.py",
             _META + "llama4/model.py", _META + "llama4/moe.py", _HF + "configuration_llama4.py", _HF + "modeling_llama4.py"),
    geometry_verification="Official Meta torchtune llama4_maverick_17b_128e builder and component implementation verified: Maverick dimensions and schedules, not Scout defaults; gated checkpoint itself remains unverified.",
    limitations=("Not a Scout preset alias: 128 routed experts, 24 MoE and 24 dense layers, 36 chunked and 12 full layers.",
                 "Official Maverick HF config is gated; checkpoint tensors were not downloaded or independently verified. Geometry is verified against Meta's named Maverick implementation.",
                 "Vision encoder and projector are excluded for text-only serving; 400B/17B official figures are rounded model-card values.",
                 "Uniform routing and fused attention workspace are explicit research scheduling choices, not measured router traces."),
    paper_reported_weight_capacity_gb=746)


def get_paper_model(model_id: str) -> PaperModelSpec:
    models = {LLAMA3_405B.model_id: LLAMA3_405B, LLAMA4_MAVERICK.model_id: LLAMA4_MAVERICK}
    try:
        return models[model_id]
    except KeyError:
        raise ValueError(f"Unknown paper model: {model_id}") from None


def verify_hf_geometry(spec: PaperModelSpec, config: Mapping) -> None:
    """Validate a caller-supplied checkpoint config; never invent missing fields."""
    text = config.get("text_config", config)
    expected = {"hidden_size": spec.hidden_size, "num_hidden_layers": spec.layers,
                "intermediate_size": spec.expert_intermediate_size or spec.intermediate_size,
                "num_attention_heads": spec.query_heads, "num_key_value_heads": spec.kv_heads,
                "vocab_size": spec.vocabulary_size, "max_position_embeddings": spec.max_context_tokens}
    if spec.routed_experts:
        expected.update(num_local_experts=spec.routed_experts, num_experts_per_tok=spec.top_k,
                        intermediate_size_mlp=spec.intermediate_size, attention_chunk_size=spec.attention_chunk_size)
        expected_moe = [i for i in range(spec.layers) if spec.is_moe(i)]
        actual_moe = text.get("moe_layers")
        if actual_moe is None:
            step = text.get("interleave_moe_layer_step")
            if not isinstance(step, int) or step < 1:
                raise ValueError("Missing explicit Maverick MoE layer schedule")
            actual_moe = list(range(step - 1, spec.layers, step))
        if actual_moe != expected_moe:
            raise ValueError("Maverick MoE layer schedule differs")
        if "layer_types" in text:
            expected_types = ["full_attention" if spec.is_full_attention(i) else "chunked_attention" for i in range(spec.layers)]
            if text["layer_types"] != expected_types:
                raise ValueError("Maverick attention layer schedule differs")
        elif "no_rope_layers" in text:
            expected_rope = [int(not spec.is_full_attention(i)) for i in range(spec.layers)]
            if text["no_rope_layers"] != expected_rope:
                raise ValueError("Maverick attention layer schedule differs")
        elif text.get("nope_layer_interval", text.get("no_rope_layer_interval")) != spec.full_attention_interval:
            raise ValueError("Missing explicit Maverick full/chunked attention schedule")
    for field, wanted in expected.items():
        if text.get(field) != wanted:
            raise ValueError(f"Checkpoint {field}: expected {wanted!r}, got {text.get(field)!r}")
    tied = text.get("tie_word_embeddings", config.get("tie_word_embeddings"))
    if tied is not spec.tie_word_embeddings:
        raise ValueError("Checkpoint must declare exact embedding/head tying")
