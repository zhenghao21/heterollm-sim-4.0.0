"""Explicit operation/tile-event backend for the Exploring HBF experiments.

The paper models bulk read/write bandwidth, not OCP transactions. This backend
selects that abstraction explicitly. It keeps private GPU memories, finite
SRAM/read buffers, network reservations and PD admission; it never replaces a
failed NandCore run, reuses a single-request fast forecast, or fits paper bars.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections import OrderedDict
from math import ceil, isfinite, ulp
from typing import Iterable

from .paper_memory import PaperMemoryAssumptions, PaperMemoryCapacityError, PaperMemorySystem
from .paper_models import PaperModelSpec, PaperPrecision
from .paper_pd import PaperPDHostAssumptions, PaperPDImporter
from .paper_kv_capacity import staggered_kv_token_series
from .paper_serving_control import DecodeExecution, PendingAdmission, ServingConfig, ServingRequest, run_serving


def prepared_slot_count(batch, output_tokens, lookahead):
    """Worst simultaneous successors for the explicitly staggered full batch."""
    return min(batch, max(1, ceil(batch*lookahead/(output_tokens-1))+1))


@dataclass(frozen=True)
class PaperExecutionAssumptions:
    gpu_dense_bf16_flops_s: float = 4e15
    matrix_efficiency: float = 1.
    scalar_flops_s: float = 130e12
    scalar_efficiency: float = 1.
    kernel_launch_ns: float = 500.
    nvlink_gb_s: float = 1800.
    infiniband_gb_s: float = 100.
    network_latency_ns: float = 500.
    nic_scope: str = "gpu"
    sram_gb_s_per_stack: float = 10000.
    sram_access_ns: float = 1.
    capacity_unit_bytes: int = 1_000_000_000
    prepared_lookahead: int = 2
    pd_host_buffer_bytes: int = 64*1024**2
    pd_host_gb_s: float = 200.
    use_period_compression: bool = True
    use_decode_transition_cache: bool = True
    basis: str = (
        "Paper Table I bandwidth/capacity and 1800/100 GB/s fabrics; current NVIDIA HGX Rubin "
        "dense BF16 4 PFLOPS and FP32 130 TFLOPS supplement undisclosed paper compute. "
        "Author SAFARI talk slide 19 explicitly assumes specified peak performance for GPU cores, "
        "memory and interconnects; matrix/scalar efficiencies are therefore 100%. "
        "500 ns launch/link startup, 10 TB/s SRAM per stack, "
        "1 ns SRAM, dedicated per-GPU NICs, uniform expert routing and balanced striped buffers "
        "are research assumptions, not paper measurements. No paper throughput is fitted.")

    def __post_init__(self):
        for key in ("gpu_dense_bf16_flops_s", "matrix_efficiency", "scalar_flops_s",
                    "scalar_efficiency", "nvlink_gb_s", "infiniband_gb_s", "sram_gb_s_per_stack", "pd_host_gb_s"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(key + " must be positive and finite")
        if self.matrix_efficiency > 1 or self.scalar_efficiency > 1:
            raise ValueError("efficiency cannot exceed one")
        for key in ("kernel_launch_ns", "network_latency_ns", "sram_access_ns"):
            if not isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(key + " must be non-negative and finite")
        if self.capacity_unit_bytes not in {10**9, 2**30}:
            raise ValueError("capacity units must explicitly select decimal GB or binary GiB")
        if isinstance(self.prepared_lookahead, bool) or not isinstance(self.prepared_lookahead, int) or self.prepared_lookahead < 1:
            raise ValueError("prepared_lookahead must be a positive integer")
        if not isinstance(self.use_period_compression,bool) or not isinstance(self.use_decode_transition_cache,bool):
            raise ValueError("event compression controls must be boolean")
        if self.nic_scope not in {"gpu", "node"}:
            raise ValueError("NIC scope must explicitly be gpu or eight-GPU node")
        if isinstance(self.pd_host_buffer_bytes, bool) or not isinstance(self.pd_host_buffer_bytes, int) or self.pd_host_buffer_bytes<2 or self.pd_host_buffer_bytes%2:
            raise ValueError("PD HOST DRAM capacity must contain two equal nonempty slots")


@dataclass(frozen=True)
class PaperParallelism:
    dp: int = 1
    pp: int = 1
    tp: int = 1
    ep: int = 1
    microbatches: int = 1

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in asdict(self).values()):
            raise ValueError("parallel degrees and microbatches must be positive integers")

    @property
    def gpus_per_replica(self):
        return self.pp * self.tp * self.ep

    @property
    def gpu_count(self):
        return self.dp * self.gpus_per_replica


def enumerate_parallelism(model: PaperModelSpec, gpu_count: int):
    if gpu_count not in (1, 2, 4, 8, 16):
        raise ValueError("paper GPU counts are 1, 2, 4, 8 and 16")
    result = []
    for dp in (1, 2, 4, 8, 16):
        for pp in (1, 2, 4, 8, 16):
            for tp in (1, 2, 4, 8, 16):
                for ep in (1, 2, 4, 8, 16):
                    if dp * pp * tp * ep != gpu_count or pp > model.layers:
                        continue
                    try:
                        model.validate_parallelism(tp, ep)
                    except ValueError:
                        continue
                    # Explicit GPipe-style microbatching is a legal serving
                    # dimension for multi-stage pipelines. Single-stage
                    # microbatching is canonicalized to M1 in this study;
                    # multi-stage layouts keep powers of two up to PP so
                    # PP2_M1/M2 and intermediate PP4_M2 are distinct.
                    micro_limit = 1 if pp == 1 else pp
                    micros = tuple(1 << power for power in range(micro_limit.bit_length())
                                   if (1 << power) <= micro_limit)
                    for micro in micros:
                        result.append(PaperParallelism(dp, pp, tp, ep, micro))
    return tuple(result)


class PaperNetwork:
    """Directed GPU injection/receive calendars; NVLink within eight-GPU nodes."""
    def __init__(self, assumptions):
        self.assumptions = assumptions
        self.calendars = {}
        self.busy = {}
        self.bytes = 0
        self.operations = 0

    def transfer(self, source, target, size, ready_ns):
        if size <= 0 or source == target:
            return float(ready_ns)
        a = self.assumptions
        remote = source // 8 != target // 8
        rate = a.infiniband_gb_s if remote else a.nvlink_gb_s
        family = "ib" if remote else "nvlink"
        source_endpoint = source//8 if remote and a.nic_scope == "node" else source
        target_endpoint = target//8 if remote and a.nic_scope == "node" else target
        keys = (f"{family}:tx:{source_endpoint}", f"{family}:rx:{target_endpoint}")
        start = max(ready_ns, *(self.calendars.get(key, 0.) for key in keys))
        duration = a.network_latency_ns + size / rate
        for key in keys:
            self.calendars[key] = start + duration
            self.busy[key] = self.busy.get(key, 0.) + duration
        self.bytes += ceil(size)
        self.operations += 1
        return start + duration

    def collective(self, ranks: Iterable[int], size, ready_ns, *, kind="all_reduce"):
        ranks = tuple(ranks)
        if len(ranks) < 2 or not size:
            return float(ready_ns)
        if kind not in {"all_reduce", "all_to_all", "all_gather"}:
            raise ValueError("unknown collective")
        n = len(ranks)
        rounds = 2 * (n-1) if kind == "all_reduce" else n-1
        now = ready_ns
        for step in range(rounds):
            distance = step+1 if kind == "all_to_all" else 1
            ends = [self.transfer(rank, ranks[(i+distance) % n], size/n, now)
                    for i, rank in enumerate(ranks)]
            now = max(ends)
        return now

    def receive_pd(self, rank, size, ready_ns):
        """External prefill node uses the same IB receive port as collectives."""
        key, latency, bandwidth = self.pd_receive_parameters(rank)
        start = max(ready_ns, self.calendars.get(key, 0.))
        duration = latency + size/bandwidth
        self.calendars[key] = start+duration
        self.busy[key] = self.busy.get(key, 0.)+duration
        self.bytes += ceil(size)
        self.operations += 1
        return start+duration

    def pd_receive_parameters(self, rank):
        return (f"ib:rx:{rank//8 if self.assumptions.nic_scope == 'node' else rank}",
                self.assumptions.network_latency_ns, self.assumptions.infiniband_gb_s)


class PaperInferenceEngine:
    """One independent DP replica with private TP/PP/EP GPU actors.

    DP copies use identical arrivals and dedicated memory/NIC resources. Their
    symmetry is explicit; only independent replicas are multiplied. Tensor,
    pipeline and expert parallelism run their actual per-rank work/collectives.
    """
    def __init__(self, model, hardware, parallel, *, batch_size, input_tokens, output_tokens,
                 assumptions=None, precision=None):
        self.model, self.hardware, self.parallel = model, hardware, parallel
        self.assumptions = assumptions or PaperExecutionAssumptions()
        self.precision = precision or PaperPrecision()
        for name,value,minimum in (("batch",batch_size,1),("input_tokens",input_tokens,1),("output_tokens",output_tokens,2)):
            if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
                raise ValueError(name+" must be an integer at least "+str(minimum))
        self.batch_size, self.input_tokens, self.output_tokens = batch_size, input_tokens, output_tokens
        self._expected_slots = frozenset(range(batch_size))
        model.validate_parallelism(parallel.tp, parallel.ep)
        if batch_size < parallel.ep * parallel.microbatches or batch_size % (parallel.ep * parallel.microbatches):
            raise ValueError("replica batch must divide EP and microbatch degrees")
        if input_tokens + output_tokens > model.max_context_tokens:
            raise ValueError("paper request exceeds model context")
        self.prepared_slots = prepared_slot_count(batch_size, output_tokens, self.assumptions.prepared_lookahead)
        self.memory = [PaperMemorySystem(hardware, assumptions=self._memory_assumptions())
                       for _ in range(parallel.gpus_per_replica)]
        self.node_nic_replica_shares = (min(parallel.dp, 8//parallel.gpus_per_replica)
            if self.assumptions.nic_scope == "node" and parallel.gpus_per_replica < 8 else 1)
        network_assumptions = replace(self.assumptions,
            infiniband_gb_s=self.assumptions.infiniband_gb_s/self.node_nic_replica_shares)
        self.network = PaperNetwork(network_assumptions)
        self.pd_importers = [PaperPDImporter(self.network, memory, rank,
            assumptions=PaperPDHostAssumptions(buffer_bytes=self.assumptions.pd_host_buffer_bytes,
                read_gb_s=self.assumptions.pd_host_gb_s, write_gb_s=self.assumptions.pd_host_gb_s,
                basis="Explicit per-GPU reserved HOST DRAM ports/buffer for a streamed bounce path; not GPU HBM and not disclosed paper hardware. HOST bandwidth/path is a sensitivity input."))
            for rank,memory in enumerate(self.memory)]
        self.category_ns = {key: 0. for key in ("Attention", "FFN", "Communication", "Others")}
        self.compressed_periods = 0
        self.layer_execution_count = 0
        self._base_work = {}
        self._decode_cache = OrderedDict()
        self.decode_cache_hits = self.decode_cache_misses = 0
        self.observed_kv_peak_bytes_per_gpu = [0]*parallel.gpus_per_replica
        self.capacity = self.capacity_requirements()
        if not self.capacity["accepted"]:
            raise PaperMemoryCapacityError(self.capacity["reason"])

    def _memory_assumptions(self):
        a, h = self.assumptions, self.hardware
        return PaperMemoryAssumptions(
            sram_read_gb_s_per_stack=a.sram_gb_s_per_stack,
            sram_write_gb_s_per_stack=a.sram_gb_s_per_stack, sram_access_ns=a.sram_access_ns,
            host_read_gb_s=h.hbm_read_gb_s+h.hbf_read_gb_s,
            # HBF program throughput is not its TSV/input-link bandwidth.
            # Select symmetric read/write input links explicitly rather than
            # throttling that link a second time to the program-array limit.
            host_write_gb_s=h.hbm_write_gb_s+h.hbf_read_gb_s,
            host_full_duplex=True, memory_full_duplex=True,
            shared_io_gb_s=100_000., shared_io_full_duplex=True, hbm_access_ns=20.,
            hbm_read_buffer_bytes_per_stack=2*1024**2,
            basis=a.basis + " GPU HBM read pipeline has an explicit 2 MiB double buffer per HBM stack; 20 ns HBM startup; symmetric stack input/output link rates at the table read-link rate, independently of NAND program-array throughput; 100 TB/s shared ingress.")

    def rank(self, stage, expert, tensor):
        return (stage * self.parallel.ep + expert) * self.parallel.tp + tensor

    def _kv_stage_bytes(self, stage, context):
        m, p = self.model, self.parallel
        unit = 2 * m.local_kv_head_count(p.tp) * m.head_dim * self.precision.kv_bytes
        return sum(unit * (context if m.is_full_attention(layer) else context % m.attention_chunk_size)
                   for layer in m.layer_indices(p.pp, stage))

    def capacity_requirements(self):
        m, p, a, h = self.model, self.parallel, self.assumptions, self.hardware
        unit = a.capacity_unit_bytes
        hbm_capacity, hbf_capacity = h.hbm_stacks*36*unit, h.hbf_stacks*512*unit
        local_prepared = ceil(self.prepared_slots / p.ep)
        occupancy = staggered_kv_token_series(self.batch_size, p.ep, self.input_tokens,
                                             self.output_tokens, m.attention_chunk_size or 0)
        rows = []
        for stage in range(p.pp):
            weights = m.stage_weight_bytes(tp=p.tp, ep=p.ep, pp=p.pp, stage=stage, precision=self.precision)
            if h.hbf_stacks:
                weights = ceil(weights/4096)*4096
            per_token = 2*m.local_kv_head_count(p.tp)*m.head_dim*self.precision.kv_bytes
            full_layers = sum(m.is_full_attention(layer) for layer in m.layer_indices(p.pp, stage))
            chunk_layers = len(m.layer_indices(p.pp, stage))-full_layers
            def pool_bytes(tokens):
                logical = tokens*per_token
                return ceil(logical/4096)*4096 if h.hbf_stacks else logical
            active_by_ep = [max(full_layers*pool_bytes(full)+chunk_layers*pool_bytes(chunk)
                for full,chunk in zip(full_phases,chunk_phases)) for full_phases,chunk_phases
                in zip(occupancy.full_by_ep_phase,occupancy.chunk_by_ep_phase)]
            prepared = (full_layers*pool_bytes(local_prepared*self.input_tokens)
                +chunk_layers*pool_bytes(local_prepared*(self.input_tokens % m.attention_chunk_size
                                                        if m.attention_chunk_size else 0)))
            kv = max(active_by_ep)+prepared
            micro_batch = self.batch_size // p.microbatches
            workspace = max(m.layer_work(layer, micro_batch, self.input_tokens, tp=p.tp, ep=p.ep,
                precision=self.precision).intermediate_live_bytes for layer in m.layer_indices(p.pp, stage))
            if stage == p.pp-1:
                workspace = max(workspace, m.endpoint_work(micro_batch, tp=p.tp, ep=p.ep,
                    precision=self.precision)["logits_bytes"] + 2*(micro_batch//p.ep)*m.hidden_size*self.precision.activation_bytes)
            device = weights+kv+(workspace if not h.hbf_stacks else 0)
            workspace_capacity = h.hbf_stacks*40_000_000 if h.intermediate_storage == "logic_die_SRAM" else hbm_capacity
            accepted = (device <= (hbf_capacity if h.hbf_stacks else hbm_capacity)
                        and workspace <= workspace_capacity)
            rows.append({"stage": stage, "resident_weight_bytes_per_gpu": weights,
                         "active_and_prepared_kv_bytes_per_gpu": kv,
                         "active_paged_kv_peak_bytes_by_ep": active_by_ep,
                         "prepared_kv_pool_bytes_per_gpu": prepared,
                         "workspace_peak_bytes_per_gpu": workspace,
                         "device_capacity_bytes": hbf_capacity if h.hbf_stacks else hbm_capacity,
                         "workspace_capacity_bytes": workspace_capacity, "accepted": accepted})
        return {"accepted": all(row["accepted"] for row in rows), "stages": rows,
                "prepared_slots_per_replica": self.prepared_slots,
                "kv_allocation_policy": "paged pool with exact peak over the complete stable-slot age cycle, including the current write before completion release; finite prepared pool is additional",
                "reason": "weights/KV/prepared imports or intermediate workspace exceed a private GPU capacity"}

    def _access(self, rank, tier, direction, byte_count, ready, label):
        if byte_count <= 0:
            return ready
        memory = self.memory[rank]
        count = memory.hbf_stacks if tier in {"hbf", "sram"} else memory.hbm_stacks
        quotient, remainder = divmod(ceil(byte_count), count)
        sizes = [quotient+(stack < remainder) for stack in range(count)]
        if tier == "hbf" and direction == "write":
            # A packed write owns whole pages, page-interleaved across stacks.
            # Do not scatter one small logical record to every stack and then
            # invent eight separate padded page programs for that record.
            page_q, page_r = divmod(ceil(byte_count/4096), count)
            sizes = [(page_q+(stack < page_r))*4096 for stack in range(count)]
        return max(memory.access(tier, direction, sizes[stack], stack=stack,
            ready_ns=ready, label=label).end_ns for stack in range(count) if sizes[stack])

    def _stream(self, rank, byte_count, flops, ready, label):
        compute = flops / (self.assumptions.gpu_dense_bf16_flops_s * self.assumptions.matrix_efficiency) * 1e9
        ready = self.memory[rank].compute(self.assumptions.kernel_launch_ns, ready_ns=ready, label="dispatch:"+label).end_ns
        if byte_count:
            end = self.memory[rank].stream_read_compute(ceil(byte_count), compute,
                ready_ns=ready, label=label).end_ns
        else:
            end = self.memory[rank].compute(compute, ready_ns=ready, label=label).end_ns
        return end

    def _intermediates(self, rank, size, ready, label):
        memory = self.memory[rank]
        if self.hardware.intermediate_storage != "logic_die_SRAM":
            return []
        q, remainder = divmod(size, memory.hbf_stacks)
        return [memory.allocate_intermediate(label, q+(i < remainder), stack=i, ready_ns=ready)
                for i in range(memory.hbf_stacks) if q+(i < remainder)]

    def _workspace_io(self, rank, size, direction, ready, label):
        tier = "sram" if self.hardware.intermediate_storage == "logic_die_SRAM" else "hbm"
        return self._access(rank, tier, direction, size, ready, label)

    def _work(self, layer, group_contexts, expert):
        p, m = self.parallel, self.model
        local = group_contexts[expert::p.ep]
        key = (m.is_moe(layer), m.is_full_attention(layer), len(group_contexts))
        if key not in self._base_work:
            self._base_work[key] = m.layer_work(layer, len(group_contexts), 0, tp=p.tp, ep=p.ep, precision=self.precision)
        work = self._base_work[key]
        old = sum(context if m.is_full_attention(layer) else context % m.attention_chunk_size for context in local)
        unit = 2*m.local_kv_head_count(p.tp)*m.head_dim*self.precision.kv_bytes
        attention = 4 * (m.query_heads//p.tp) * m.head_dim * (old+len(local))
        return replace(work, kv_read_bytes=old*unit, kv_resident_bytes=(old+len(local))*unit,
                       attention_flops=attention, compute_flops=work.compute_flops-work.attention_flops+attention)

    def _layer(self, stage, layer, contexts, ready):
        p, m = self.parallel, self.model
        ends, writes = {}, []
        works, leases = {}, {}
        for expert in range(p.ep):
            work = self._work(layer, contexts, expert)
            for tensor in range(p.tp):
                rank = self.rank(stage, expert, tensor)
                works[rank] = work
                leases[rank] = self._intermediates(rank, work.intermediate_live_bytes, ready, "layer-workspace")
                scalar = 10*work.local_batch_size*m.hidden_size / (self.assumptions.scalar_flops_s*self.assumptions.scalar_efficiency)*1e9
                start = self._workspace_io(rank, dict(work.live_buffers)["residual"]*2, "read", ready, "norm-input")
                start = self._stream(rank, work.norm_weight_bytes//2, 0, start, "attention-norm-weight")
                start = self.memory[rank].compute(scalar, ready_ns=start, label="norm-rope").end_ns
                o_parameters = m.hidden_size*(m.query_heads//p.tp)*m.head_dim
                o_bytes = o_parameters*self.precision.weight_bytes
                projection = self._stream(rank, work.attention_weight_bytes-o_bytes,
                    work.attention_projection_flops-2*work.local_batch_size*o_parameters, start, "qkv-projection")
                kv_tier = "hbf" if self.hardware.hbf_stacks else "hbm"
                writes.append(self._access(rank, kv_tier, "write", work.kv_write_bytes, projection, "decode-kv"))
                # New K/V remain in the live workspace; writing them can overlap
                # attention on existing KV, as stated in the paper's footnote.
                attention = self._stream(rank, work.kv_read_bytes, work.attention_flops, projection, "attention-kv")
                attended = sum((context if m.is_full_attention(layer) else context % m.attention_chunk_size)+1
                               for context in contexts[expert::p.ep])
                softmax_ops = 5*attended*(m.query_heads//p.tp)
                attention = self.memory[rank].compute(softmax_ops / (self.assumptions.scalar_flops_s*
                    self.assumptions.scalar_efficiency)*1e9, ready_ns=attention, label="softmax").end_ns
                ends[rank] = self._stream(rank, o_bytes, 2*work.local_batch_size*o_parameters,
                                         attention, "attention-output-projection")
        attention_end = max(ends.values())
        self.category_ns["Attention"] += attention_end-ready
        communication_start = attention_end
        for expert in range(p.ep):
            ranks = [self.rank(stage, expert, tensor) for tensor in range(p.tp)]
            payload = len(contexts)//p.ep * m.hidden_size * self.precision.activation_bytes
            collective = self.network.collective(ranks, payload, max(ends[rank] for rank in ranks))
            for rank in ranks:
                ends[rank] = collective
        # Normalize and route before any expert dispatch, including EP=1.
        for rank, work in works.items():
            ends[rank] = self._stream(rank, work.norm_weight_bytes//2, 0, ends[rank], "ffn-norm-weight")
            ends[rank] = self.memory[rank].compute(5*work.local_batch_size*m.hidden_size /
                (self.assumptions.scalar_flops_s*self.assumptions.scalar_efficiency)*1e9,
                ready_ns=ends[rank], label="ffn-norm").end_ns
        if m.is_moe(layer):
            for rank, work in works.items():
                ends[rank] = self._stream(rank, work.router_weight_bytes,
                                         work.router_flops, ends[rank], "expert-router")
                route_bytes = dict(work.live_buffers)["router_logits"]
                ends[rank] = self._workspace_io(rank, route_bytes, "write", ends[rank], "router-logits")
                ends[rank] = self._workspace_io(rank, route_bytes, "read", ends[rank], "router-top-k")
                ends[rank] = self.memory[rank].compute(work.local_batch_size*m.routed_experts /
                    (self.assumptions.scalar_flops_s*self.assumptions.scalar_efficiency)*1e9,
                    ready_ns=ends[rank], label="router-select").end_ns
        if m.is_moe(layer) and p.ep > 1:
            for tensor in range(p.tp):
                ranks = [self.rank(stage, expert, tensor) for expert in range(p.ep)]
                payload = works[ranks[0]].expert_dispatch_bytes/2 * p.ep/(p.ep-1)
                collective = self.network.collective(ranks, payload, max(ends[rank] for rank in ranks), kind="all_to_all")
                for rank in ranks:
                    ends[rank] = collective
        ffn_start = max(ends.values())
        self.category_ns["Communication"] += ffn_start-communication_start
        for rank, work in works.items():
            temp = 3*work.local_batch_size*m.hidden_size*self.precision.activation_bytes
            start = self._workspace_io(rank, temp, "write", ends[rank], "attention-output-and-ffn-input")
            start = self._workspace_io(rank, temp, "read", start, "ffn-input")
            end = start
            gate_up = dict(work.live_buffers)["ffn_gate_up"]
            # Shared and routed experts reuse the workspace serially. Within
            # each path, down-projection waits for gate/up, SwiGLU and its IO.
            for name, weight, flops in (("dense", work.dense_weight_bytes, work.dense_flops),
                    ("shared", work.shared_weight_bytes, work.shared_flops),
                    ("routed", work.routed_weight_bytes, work.routed_flops)):
                if not weight:
                    continue
                end = self._stream(rank, weight*2/3, flops*2/3, end, name+"-gate-up")
                end = self._workspace_io(rank, gate_up, "write", end, name+"-gate-up-output")
                end = self._workspace_io(rank, gate_up, "read", end, name+"-swiglu-input")
                end = self.memory[rank].compute(3*gate_up/self.precision.activation_bytes /
                    (self.assumptions.scalar_flops_s*self.assumptions.scalar_efficiency)*1e9,
                    ready_ns=end, label=name+"-swiglu").end_ns
                end = self._stream(rank, weight/3, flops/3, end, name+"-down")
            ends[rank] = end
        ffn_end = max(ends.values())
        self.category_ns["FFN"] += ffn_end-ffn_start
        communication_start = ffn_end
        if m.is_moe(layer) and p.ep > 1:
            for tensor in range(p.tp):
                ranks = [self.rank(stage, expert, tensor) for expert in range(p.ep)]
                payload = works[ranks[0]].expert_dispatch_bytes/2*p.ep/(p.ep-1)
                collective = self.network.collective(ranks, payload, max(ends[rank] for rank in ranks), kind="all_to_all")
                for rank in ranks:
                    ends[rank] = collective
        for expert in range(p.ep):
            ranks = [self.rank(stage, expert, tensor) for tensor in range(p.tp)]
            collective = self.network.collective(ranks, len(contexts)//p.ep*m.hidden_size*self.precision.activation_bytes,
                                                  max(ends[rank] for rank in ranks))
            for rank in ranks:
                ends[rank] = collective
        end = max([*ends.values(), *writes])
        network_end = max(ends.values())
        self.category_ns["Communication"] += network_end-communication_start
        self.category_ns["Others"] += end-network_end
        for rank, items in leases.items():
            for lease in items:
                self.memory[rank].release_intermediate(lease, at_ns=end)
        self.layer_execution_count += 1
        return end

    def _stage(self, stage, contexts, ready):
        p, m = self.parallel, self.model
        now = ready
        if stage == 0:
            endpoint = m.endpoint_work(len(contexts), tp=p.tp, ep=p.ep, precision=self.precision)
            lookup_ends = {self.rank(stage,e,t): self._stream(self.rank(stage,e,t),
                endpoint["embedding_read_bytes"], 0., now, "embedding") for e in range(p.ep) for t in range(p.tp)}
            gathered = []
            for expert in range(p.ep):
                ranks = [self.rank(stage,expert,t) for t in range(p.tp)]
                gathered.append(self.network.collective(ranks,
                    len(contexts)//p.ep*m.hidden_size*self.precision.activation_bytes,
                    max(lookup_ends[rank] for rank in ranks), kind="all_gather"))
            now = max(gathered)
            self.category_ns["Others"] += now-ready
        layers = list(m.layer_indices(p.pp, stage))
        # Two dense layers or eight Maverick layers restore the finite-buffer
        # slot ordering. Compression additionally requires real quiescence.
        period = 8 if m.attention_chunk_size else 2
        cursor = 0
        stage_ranks = [self.rank(stage, expert, tensor) for expert in range(p.ep) for tensor in range(p.tp)]
        while cursor < len(layers):
            size = min(period, len(layers)-cursor)
            signature = tuple((m.is_moe(layer), m.is_full_attention(layer)) for layer in layers[cursor:cursor+size]) + tuple(contexts)
            snapshots = [self.memory[rank].snapshot_counters(signature=signature) for rank in stage_ranks]
            network_before = (dict(self.network.calendars), dict(self.network.busy), self.network.bytes, self.network.operations)
            categories_before = dict(self.category_ns)
            start = now
            for layer in layers[cursor:cursor+size]:
                now = self._layer(stage, layer, contexts, now)
            remaining = (len(layers)-cursor-size)//size
            if remaining and self.assumptions.use_period_compression:
                expected = tuple((m.is_moe(layer), m.is_full_attention(layer)) for layer in layers[cursor:cursor+size])
                same = all(tuple((m.is_moe(layer), m.is_full_attention(layer))
                    for layer in layers[cursor+(repeat+1)*size:cursor+(repeat+2)*size]) == expected
                    for repeat in range(remaining))
                used_keys = [key for key, value in self.network.busy.items() if value != network_before[1].get(key, 0.)]
                idle = (all(network_before[0].get(key, 0.) <= start for key in used_keys)
                        and all(self.network.calendars[key] <= now for key in used_keys))
                # All private actors must prove the period before any replay.
                qualified = same and idle and all(self.memory[rank].can_repeat_quiescent_period(
                    snapshot, start, now, signature=signature)
                    for rank, snapshot in zip(stage_ranks, snapshots))
                if qualified:
                    for rank, snapshot in zip(stage_ranks, snapshots):
                        if not self.memory[rank].repeat_quiescent_period(snapshot, start, now,
                                repetitions=remaining, signature=signature):
                            raise RuntimeError("period proof changed before replay")
                    shift = remaining*(now-start)
                    for key in used_keys:
                        self.network.calendars[key] += shift
                        self.network.busy[key] += remaining*(self.network.busy[key]-network_before[1].get(key, 0.))
                    self.network.bytes += remaining*(self.network.bytes-network_before[2])
                    self.network.operations += remaining*(self.network.operations-network_before[3])
                    for key in self.category_ns:
                        self.category_ns[key] += remaining*(self.category_ns[key]-categories_before[key])
                    self.layer_execution_count += remaining*size
                    self.compressed_periods += remaining
                    now += shift
                    cursor += remaining*size
            cursor += size
        if stage == p.pp-1:
            head_start = now
            endpoint = m.endpoint_work(len(contexts), tp=p.tp, ep=p.ep, precision=self.precision)
            ends = []
            for expert in range(p.ep):
                head_ends = []
                ranks = [self.rank(stage, expert, tensor) for tensor in range(p.tp)]
                for rank in ranks:
                    leases = self._intermediates(rank, endpoint["logits_bytes"], now, "logits")
                    end = self._stream(rank, endpoint["head_read_bytes"]+endpoint["final_norm_read_bytes"],
                                       endpoint["head_flops"], now, "lm-head")
                    end = self._workspace_io(rank, endpoint["logits_bytes"], "write", end, "logits")
                    for lease in leases:
                        self.memory[rank].release_intermediate(lease, at_ns=end)
                    head_ends.append(end)
                end = self.network.collective(ranks, len(contexts)//p.ep*16, max(head_ends))
                ends.append(end)
            now = max(ends)
            self.category_ns["Others"] += now-head_start
        return now

    def _decode_signature_and_capacity(self, requests, pending):
        p,m = self.parallel,self.model
        ordered = sorted(requests,key=lambda request:request.slot_id)
        micro = self.batch_size//p.microbatches
        signature = []
        for index in range(p.microbatches):
            group = ordered[index*micro:(index+1)*micro]
            for expert in range(p.ep):
                local = group[expert::p.ep]
                signature += [sum(request.context_tokens for request in local),
                    sum(request.context_tokens % m.attention_chunk_size for request in local) if m.attention_chunk_size else 0]
        for expert in range(p.ep):
            active = [request for request in requests if request.slot_id % p.ep == expert]
            successors = [item.request for item in pending if item.request.slot_id % p.ep == expert]
            full_tokens = sum(request.context_tokens+1 for request in active)+sum(request.context_tokens for request in successors)
            chunk_tokens = (sum(request.context_tokens % m.attention_chunk_size+1 for request in active)
                +sum(request.context_tokens % m.attention_chunk_size for request in successors)) if m.attention_chunk_size else 0
            for stage in range(p.pp):
                full_layers = sum(m.is_full_attention(layer) for layer in m.layer_indices(p.pp,stage))
                chunk_layers = len(m.layer_indices(p.pp,stage))-full_layers
                unit = 2*m.local_kv_head_count(p.tp)*m.head_dim*self.precision.kv_bytes
                sizes = [full_tokens*unit,chunk_tokens*unit]
                if self.hardware.hbf_stacks:
                    sizes = [ceil(value/4096)*4096 for value in sizes]
                size = full_layers*sizes[0]+chunk_layers*sizes[1]
                for tensor in range(p.tp):
                    rank = self.rank(stage,expert,tensor)
                    self.observed_kv_peak_bytes_per_gpu[rank] = max(size,self.observed_kv_peak_bytes_per_gpu[rank])
                    plan = self.capacity["stages"][stage]
                    if size > plan["active_and_prepared_kv_bytes_per_gpu"]:
                        raise PaperMemoryCapacityError("actual active/prepared paged KV exceeds its planned private pool")
        return tuple(signature)

    def execute_decode(self, requests, pending, now_ns):
        if len(requests) != self.batch_size:
            raise ValueError("configured full-batch execution requires every admitted slot")
        if frozenset(request.slot_id for request in requests) != self._expected_slots:
            raise ValueError("full-batch requests must own each configured stable slot exactly once")
        if any(request.input_tokens != self.input_tokens or request.output_tokens != self.output_tokens for request in requests):
            raise ValueError("request lengths differ from the configured homogeneous paper workload")
        if len({request.request_id for request in requests}) != self.batch_size or any(not request.prefill_first_output_token for request in requests):
            raise ValueError("decode slots require distinct requests with their first output supplied by prefill")
        if (len({item.request.slot_id for item in pending}) != len(pending)
            or any(item.request.slot_id not in self._expected_slots or item.request.input_tokens != self.input_tokens
                   or item.request.output_tokens != self.output_tokens for item in pending)):
            raise ValueError("prepared successors must use distinct configured slots and the same workload")
        key = self._decode_signature_and_capacity(requests,pending)
        cached = self._decode_cache.get(key) if self.assumptions.use_decode_transition_cache else None
        if cached is not None:
            templates, network_state, execution, category_delta, layers, periods = cached
            def matching_network_input(item):
                name,_,_,source_offset,source_roundoff = item
                current = self.network.calendars.get(name,0.)
                if source_offset == 0:
                    return current <= now_ns
                return current > now_ns and abs((current-now_ns)-source_offset) <= source_roundoff+4*(ulp(current)+ulp(now_ns))
            network_ok = all(matching_network_input(item) for item in network_state[0])
            if network_ok and all(memory.can_apply_quiescent_transition(template,now_ns,signature=key)
                                 for memory,template in zip(self.memory,templates)):
                for memory,template in zip(self.memory,templates):
                    if memory.apply_quiescent_transition(template,now_ns,signature=key) is None:
                        raise RuntimeError("decode transition proof changed during replay")
                for name,offset,busy,_,_ in network_state[0]:
                    self.network.calendars[name] = now_ns+offset
                    self.network.busy[name] = self.network.busy.get(name,0.)+busy
                self.network.bytes += network_state[1]
                self.network.operations += network_state[2]
                for name,value in category_delta.items():
                    self.category_ns[name] += value
                self.layer_execution_count += layers
                self.compressed_periods += periods
                self.decode_cache_hits += 1
                self._decode_cache.move_to_end(key)
                return replace(execution,end_ns=now_ns+execution.end_ns)
        self.decode_cache_misses += 1
        snapshots = [memory.snapshot_counters(signature=key) for memory in self.memory]
        net_before = (dict(self.network.calendars),dict(self.network.busy),self.network.bytes,self.network.operations)
        category_before = dict(self.category_ns)
        layer_before,period_before = self.layer_execution_count,self.compressed_periods
        execution = self._execute_decode_events(requests,pending,now_ns)
        if self.assumptions.use_decode_transition_cache:
            used = [name for name,value in self.network.busy.items() if value != net_before[1].get(name,0.)]
            net_complete = all(self.network.calendars[name] <= execution.end_ns for name in used)
            templates = tuple(memory.make_quiescent_transition(before,now_ns,execution.end_ns,signature=key,
                              allow_reserved_frontiers=True)
                              for memory,before in zip(self.memory,snapshots))
            if net_complete and all(template is not None for template in templates):
                network = (tuple((name,self.network.calendars[name]-now_ns,
                    self.network.busy[name]-net_before[1].get(name,0.),
                    max(0,net_before[0].get(name,0.)-now_ns),
                    4*(ulp(net_before[0].get(name,0.))+ulp(now_ns))) for name in used),
                    self.network.bytes-net_before[2],self.network.operations-net_before[3])
                self._decode_cache[key] = (templates,network,replace(execution,end_ns=execution.end_ns-now_ns),
                    {name:self.category_ns[name]-category_before[name] for name in category_before},
                    self.layer_execution_count-layer_before,self.compressed_periods-period_before)
                self._decode_cache.move_to_end(key)
                while len(self._decode_cache) > max(128,self.output_tokens-1):
                    self._decode_cache.popitem(last=False)
        return execution

    def _execute_decode_events(self, requests, pending, now_ns):
        p = self.parallel
        contexts = [request.context_tokens for request in sorted(requests, key=lambda request: request.slot_id)]
        if len(contexts) != self.batch_size:
            raise ValueError("configured full-batch execution requires every admitted slot")
        before = dict(self.category_ns)
        before_program = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in self.memory)
        micro = self.batch_size//p.microbatches
        completed = []
        stage_available = [now_ns]*p.pp
        for index in range(p.microbatches):
            group = contexts[index*micro:(index+1)*micro]
            ready = now_ns
            for stage in range(p.pp):
                ready = self._stage(stage, group, max(ready, stage_available[stage]))
                stage_available[stage] = ready
                if stage+1 < p.pp:
                    transfer_start = ready
                    transfers = [self.network.transfer(self.rank(stage,e,t), self.rank(stage+1,e,t),
                        len(group)//p.ep*self.model.hidden_size*self.precision.activation_bytes,
                        ready) for e in range(p.ep) for t in range(p.tp)]
                    ready = max(transfers)
                    self.category_ns["Communication"] += ready-transfer_start
            completed.append(ready)
        end = max(completed)
        # Return selected tokens to the first PP stage before its next step.
        if p.pp > 1:
            feedback_start = end
            end = max(self.network.transfer(self.rank(p.pp-1,e,t), self.rank(0,e,t),
                self.batch_size//p.ep*4, end) for e in range(p.ep) for t in range(p.tp))
            self.category_ns["Communication"] += end-feedback_start
        bytes_written = sum(self.model.kv_bytes_per_token(tp=p.tp, pp=p.pp, stage=stage,
            precision=self.precision)*(self.batch_size//p.ep)*p.ep*p.tp for stage in range(p.pp))
        # Bulk paper writes use complete NAND pages but have no invented FTL/GC.
        program = sum(memory.bytes_by_operation.get("hbf:write", 0) for memory in self.memory)-before_program
        return DecodeExecution(end, program, bytes_written, 0,
            {key: self.category_ns[key]-before[key] for key in before})

    def import_kv(self, request, now_ns):
        p = self.parallel
        expert = request.slot_id % p.ep
        ends = []
        transfer = 0
        program = 0
        host_busy = 0.
        network_busy = 0.
        program_busy = 0.
        for stage in range(p.pp):
            size = self._kv_stage_bytes(stage, request.context_tokens)
            for tensor in range(p.tp):
                rank = self.rank(stage, expert, tensor)
                nic_key = self.network.pd_receive_parameters(rank)[0]
                nic_before = self.network.busy.get(nic_key, 0.)
                media_before = {key:value.busy_ns for key,value in self.memory[rank]._calendars.items()
                    if key.startswith(("hbf:","hbf_array:","hbm:")) and key.endswith((":write",":shared"))}
                imported = self.pd_importers[rank].import_kv(size, now_ns=now_ns,
                    tier="hbf" if self.hardware.hbf_stacks else "hbm", label="pd-import")
                ends.append(imported.end_ns)
                transfer += imported.transfer_bytes
                program += imported.physical_program_bytes
                host_busy += imported.counters["host_write_busy_ns"]+imported.counters["host_read_busy_ns"]
                network_busy += self.network.busy.get(nic_key, 0.)-nic_before
                program_busy += sum(value.busy_ns-media_before.get(key, 0.)
                    for key,value in self.memory[rank]._calendars.items()
                    if key.startswith(("hbf:","hbf_array:","hbm:")) and key.endswith((":write",":shared")))
        return PendingAdmission(request, max(ends), transfer, program, transfer,
                                {"PD HOST DRAM service": host_busy, "PD IB receive service":network_busy,
                                 "PD memory program/write service":program_busy})

    def run(self, *, warmup_steps, measurement_steps):
        p = self.parallel
        result = run_serving(ServingConfig(self.batch_size, self.input_tokens, self.output_tokens,
            warmup_steps=warmup_steps, measurement_steps=measurement_steps,
            prefetch_slots=self.prepared_slots, import_lookahead_tokens=self.assumptions.prepared_lookahead,
            retain_step_records=False, hbf_capacity_bytes=self.hardware.hbf_stacks*512*
                self.assumptions.capacity_unit_bytes*p.gpus_per_replica), self.execute_decode, self.import_kv)
        output = result.as_dict()
        window_seconds = (result.measurement_end_ns-result.measurement_start_ns)/1e9
        whole_group_program = result.decode_program_bytes+result.pd_program_completed_bytes
        physical_lower = max(0,whole_group_program-result.pending_unfinished_program_start_bytes)
        physical_upper = whole_group_program+result.pending_unfinished_program_end_bytes
        hbf_capacity = self.hardware.hbf_stacks*512*self.assumptions.capacity_unit_bytes*p.gpus_per_replica
        factor = 3*365*24*3600/window_seconds/hbf_capacity if hbf_capacity else None
        output.update(backend="paper_bulk_tile_events", model=self.model.metadata(self.precision),
            hardware=asdict(self.hardware), parallelism=asdict(p), research_assumptions=asdict(self.assumptions),
            capacity=self.capacity, gpu_memories=[memory.summary() for memory in self.memory],
            pd_host_staging=[importer.summary() for importer in self.pd_importers],
            pd_completion_accounting="whole-import completion groups; finite partial-program boundary uncertainty is reported, not claimed as per-page completion",
            measurement_program_bytes_lower_bound=physical_lower,
            measurement_program_bytes_upper_bound=physical_upper,
            projected_three_year_pec_lower_bound=physical_lower*factor if factor is not None else None,
            projected_three_year_pec_upper_bound=physical_upper*factor if factor is not None else None,
            network={"bytes": self.network.bytes, "operations": self.network.operations,
                     "nic_scope": self.assumptions.nic_scope,
                     "node_nic_replica_shares": self.node_nic_replica_shares,
                     "effective_ib_gb_s_per_replica_node": self.network.assumptions.infiniband_gb_s,
                     "node_partition_policy": "fixed equal bandwidth partitions among colocated DP replicas; no borrowed idle share",
                     "resources": {key:{"available_ns":value,"busy_ns":self.network.busy[key]}
                                   for key,value in self.network.calendars.items()}},
            compressed_layer_periods=self.compressed_periods, executed_model_layers=self.layer_execution_count,
            decode_transition_cache_hits=self.decode_cache_hits,
            decode_transition_cache_misses=self.decode_cache_misses,
            observed_kv_peak_bytes_per_gpu=self.observed_kv_peak_bytes_per_gpu,
            dp_symmetry_reduction="identical independent replicas with private memory; dedicated GPU NICs or explicit fixed shares of the node NIC",
            cluster_decode_tokens_per_second=result.decode_tokens_per_second*p.dp,
            per_gpu_decode_tokens_per_second=result.decode_tokens_per_second/p.gpus_per_replica,
            per_gpu_batch_size=self.batch_size/p.gpus_per_replica,
            old_single_request_fast_mode_used=False,
            kv_program_layout="packed write groups, whole 4KiB pages interleaved over stacks from stack zero; no FTL/RMW/erase inferred",
            category_semantics="phase spans per pipeline stage/microbatch; overlapping stages and PD service are non-additive, not a critical-path Figure 5 decomposition",
            accuracy="operation/tile model assumptions; not an OCP transaction reproduction or a 10% accuracy guarantee")
        return output
