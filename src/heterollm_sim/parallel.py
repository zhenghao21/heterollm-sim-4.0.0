"""Logical TP/PP/EP planning independent of model-name adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .config import ScenarioConfig
from .ir import (
    ACTIVE_MEMORY_COMPONENT_KINDS,
    LayerSpec,
    ModelGraphExecutionView,
    ParallelSpec,
    RankMappingSpec,
    model_graph_execution_view,
    normalize_component_kind,
)


@dataclass(frozen=True)
class LogicalRank:
    rank: int
    component_id: str
    tp_rank: int
    pp_rank: int
    ep_rank: int
    memory_component_id: Optional[str] = None
    cim_component_id: Optional[str] = None

    @property
    def coordinates(self) -> Tuple[int, int, int]:
        return (self.tp_rank, self.pp_rank, self.ep_rank)


@dataclass(frozen=True)
class ShardExtent:
    global_size: int
    degree: int
    rank: int
    padded_size: int
    local_size: int
    padding: int


@dataclass(frozen=True)
class ParallelPlan:
    tp_degree: int
    pp_degree: int
    ep_degree: int
    ranks: Tuple[LogicalRank, ...]
    layer_to_stage: Mapping[str, int]
    collective_algorithm: str
    routing_policy: str
    allow_padding: bool

    @property
    def world_size(self) -> int:
        return self.tp_degree * self.pp_degree * self.ep_degree

    def stage_for_layer(self, layer: LayerSpec) -> int:
        return int(self.layer_to_stage[layer.layer_id])

    def ranks_for_stage(self, stage: int) -> Tuple[LogicalRank, ...]:
        if not 0 <= stage < self.pp_degree:
            raise ValueError("invalid PP stage {}".format(stage))
        return tuple(rank for rank in self.ranks if rank.pp_rank == stage)

    def ranks_for_layer(self, layer: LayerSpec) -> Tuple[LogicalRank, ...]:
        """Return every TP×EP rank that owns ``layer``'s contiguous PP stage."""

        return self.ranks_for_stage(self.stage_for_layer(layer))

    def tp_group(self, stage: int, ep_rank: int = 0) -> Tuple[LogicalRank, ...]:
        return tuple(
            sorted(
                (
                    rank
                    for rank in self.ranks
                    if rank.pp_rank == stage and rank.ep_rank == ep_rank
                ),
                key=lambda rank: rank.tp_rank,
            )
        )

    def ep_group(self, stage: int, tp_rank: int = 0) -> Tuple[LogicalRank, ...]:
        return tuple(
            sorted(
                (
                    rank
                    for rank in self.ranks
                    if rank.pp_rank == stage and rank.tp_rank == tp_rank
                ),
                key=lambda rank: rank.ep_rank,
            )
        )

    def rank_at(self, tp_rank: int, pp_rank: int, ep_rank: int) -> LogicalRank:
        for rank in self.ranks:
            if rank.coordinates == (tp_rank, pp_rank, ep_rank):
                return rank
        raise KeyError(
            "missing rank at tp={}, pp={}, ep={}".format(tp_rank, pp_rank, ep_rank)
        )


def shard_extent(
    global_size: int, degree: int, rank: int, *, allow_padding: bool = True
) -> ShardExtent:
    if global_size <= 0:
        raise ValueError("global_size must be positive")
    if degree <= 0 or not 0 <= rank < degree:
        raise ValueError("invalid shard degree or rank")
    if global_size % degree and not allow_padding:
        raise ValueError(
            "dimension {} is not divisible by degree {}".format(global_size, degree)
        )
    local = int(math.ceil(global_size / float(degree)))
    padded = local * degree
    return ShardExtent(global_size, degree, rank, padded, local, padded - global_size)


def _auto_layer_stages(
    layers: Sequence[LayerSpec], pp_degree: int
) -> Dict[str, int]:
    if pp_degree > len(layers):
        raise ValueError(
            "pp_degree {} exceeds model layer count {}".format(pp_degree, len(layers))
        )
    # Contiguous cost-weighted stages.  Hybrid blocks and high-expert MoE
    # layers can differ by orders of magnitude, so count balancing is not a
    # conservative automatic PP policy.
    costs = [_layer_stage_cost(layer) for layer in layers]
    remaining_cost = float(sum(costs))
    mapping: Dict[str, int] = {}
    cursor = 0
    for stage in range(pp_degree):
        stages_left = pp_degree - stage
        layers_left = len(layers) - cursor
        if stages_left == 1:
            stop = len(layers)
        else:
            target = remaining_cost / stages_left
            stop = cursor
            accumulated = 0.0
            max_stop = len(layers) - (stages_left - 1)
            while stop < max_stop:
                next_cost = float(costs[stop])
                if stop > cursor and accumulated + next_cost > target:
                    break
                accumulated += next_cost
                stop += 1
            if stop == cursor:
                stop += 1
        for index in range(cursor, stop):
            mapping[layers[index].layer_id] = stage
            remaining_cost -= float(costs[index])
        cursor = stop
    return mapping


def _layer_stage_cost(layer: LayerSpec) -> int:
    """Return a deterministic parameter/operation proxy for PP balancing."""

    hidden = layer.hidden_size
    if layer.is_linear_attention:
        geometry = layer.linear_attention
        if geometry is None:  # guarded by LayerSpec validation
            raise ValueError("linear layer is missing geometry")
        mixer = hidden * (
            geometry.query_width + geometry.key_width + geometry.value_width
        )
        mixer += geometry.value_width * hidden
        if geometry.output_gate:
            mixer += hidden * geometry.value_width
        mixer += geometry.conv_kernel_size * (
            geometry.query_width + geometry.key_width + geometry.value_width
        )
        mixer += geometry.recurrent_state_elements
    else:
        head_dim = layer.effective_attention_head_dim
        qkv_width = hidden + 2 * layer.effective_kv_heads * head_dim
        mixer = hidden * qkv_width + hidden * hidden
    if layer.is_moe:
        ffn = hidden * layer.num_experts * (3 * layer.intermediate_size)
        ffn += hidden * layer.num_experts
        if layer.has_shared_expert:
            ffn += hidden * (3 * layer.shared_expert_intermediate_size)
            if layer.shared_expert_gate:
                ffn += hidden
    else:
        ffn = hidden * (3 * layer.intermediate_size)
    return max(1, mixer + ffn)


def _logical_rank(mapping: RankMappingSpec) -> LogicalRank:
    return LogicalRank(
        rank=mapping.rank,
        component_id=mapping.component_id,
        tp_rank=mapping.tp_rank,
        pp_rank=mapping.pp_rank,
        ep_rank=mapping.ep_rank,
        memory_component_id=mapping.memory_component_id,
        cim_component_id=mapping.cim_component_id,
    )


def _auto_ranks(
    scenario: ScenarioConfig, tp_degree: int, pp_degree: int, ep_degree: int
) -> Tuple[LogicalRank, ...]:
    compute_components = sorted(
        (
            component.component_id
            for component in scenario.hardware.components
            if component.kind.strip().lower().replace("-", "_") == "gpu"
        )
    )
    world_size = tp_degree * pp_degree * ep_degree
    if len(compute_components) < world_size:
        raise ValueError(
            "parallel world requires {} compute components, found {}; provide an explicit "
            "rank_mapping to model colocated logical ranks".format(
                world_size, len(compute_components)
            )
        )
    ranks: List[LogicalRank] = []
    flat_rank = 0
    for pp_rank, ep_rank, tp_rank in product(
        range(pp_degree), range(ep_degree), range(tp_degree)
    ):
        ranks.append(
            LogicalRank(
                rank=flat_rank,
                component_id=compute_components[flat_rank],
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                ep_rank=ep_rank,
            )
        )
        flat_rank += 1
    return tuple(ranks)


def build_parallel_plan(
    scenario: ScenarioConfig,
    execution_view: Optional[ModelGraphExecutionView] = None,
) -> ParallelPlan:
    execution_view = execution_view or model_graph_execution_view(
        scenario.model.graph,
        schema_version=scenario.model.schema_version,
    )
    execution_layers = tuple(
        descriptor.layer for descriptor in execution_view.layer_instances
    )

    source: ParallelSpec = scenario.placement.parallel
    tp_degree = source.tp_degree
    pp_degree = source.pp_degree
    ep_degree = source.ep_degree
    if min(tp_degree, pp_degree, ep_degree) <= 0:
        raise ValueError("parallel degrees must be positive")
    allow_padding = source.allow_padding
    rank_sources = source.rank_mapping
    ranks = (
        tuple(_logical_rank(mapping) for mapping in rank_sources)
        if rank_sources
        else _auto_ranks(scenario, tp_degree, pp_degree, ep_degree)
    )
    expected_world = tp_degree * pp_degree * ep_degree
    if len(ranks) != expected_world:
        raise ValueError(
            "rank_mapping must contain exactly {} ranks".format(expected_world)
        )
    if len({rank.rank for rank in ranks}) != len(ranks):
        raise ValueError("rank_mapping rank values must be unique")
    if {rank.rank for rank in ranks} != set(range(expected_world)):
        raise ValueError(
            "rank_mapping rank values must cover [0, {})".format(
                expected_world
            )
        )
    expected_coordinates = set(
        product(range(tp_degree), range(pp_degree), range(ep_degree))
    )
    actual_coordinates = {rank.coordinates for rank in ranks}
    if actual_coordinates != expected_coordinates:
        raise ValueError("rank_mapping must cover every TP/PP/EP coordinate exactly once")
    components = scenario.hardware.component_map()
    for rank in ranks:
        component = components.get(rank.component_id)
        if component is None:
            raise ValueError(
                "rank {} references unknown component {}".format(
                    rank.rank, rank.component_id
                )
            )
        kind = component.kind.strip().lower().replace("-", "_")
        if kind != "gpu":
            raise ValueError(
                "rank {} must map to a GPU component for the reference cost provider, not {}".format(
                    rank.rank, component.kind
                )
            )
        for optional_id, label in (
            (rank.memory_component_id, "memory"),
            (rank.cim_component_id, "CIM"),
        ):
            if optional_id and optional_id not in components:
                raise ValueError(
                    "rank {} references unknown {} component {}".format(
                        rank.rank, label, optional_id
                    )
                )
        if rank.memory_component_id:
            memory_component = components[rank.memory_component_id]
            memory_kind = normalize_component_kind(memory_component.kind)
            if (
                not memory_component.is_active_memory
                or not memory_component.is_writable
            ):
                reason = (
                    memory_kind
                    if not memory_component.is_active_memory
                    else "read-only {}".format(memory_kind)
                )
                raise ValueError(
                    "rank {} memory component {} must be writable active memory, not {}".format(
                        rank.rank, rank.memory_component_id, reason
                    )
                )
        if rank.cim_component_id:
            cim_kind = components[
                rank.cim_component_id
            ].kind.strip().lower().replace("-", "_")
            if "cim" not in cim_kind and "compute_in_memory" not in cim_kind:
                raise ValueError(
                    "rank {} CIM component {} has non-CIM kind {}".format(
                        rank.rank, rank.cim_component_id, cim_kind
                    )
                )

    layer_to_stage = _auto_layer_stages(execution_layers, pp_degree)
    explicit = dict(source.layer_to_stage)
    known_layers = {layer.layer_id for layer in execution_layers}
    unknown = sorted(set(explicit) - known_layers)
    if unknown:
        raise ValueError(
            "layer_to_stage references unknown layers: {}".format(", ".join(unknown))
        )
    for layer_id, stage in explicit.items():
        stage_number = int(stage)
        if not 0 <= stage_number < pp_degree:
            raise ValueError(
                "layer {} has invalid PP stage {}".format(layer_id, stage_number)
            )
        layer_to_stage[layer_id] = stage_number
    occupied = set(layer_to_stage.values())
    if occupied != set(range(pp_degree)):
        raise ValueError("every PP stage must own at least one layer")
    stage_sequence = tuple(
        layer_to_stage[layer.layer_id] for layer in execution_layers
    )
    if any(
        next_stage < stage
        for stage, next_stage in zip(stage_sequence, stage_sequence[1:])
    ):
        raise ValueError(
            "layer_to_stage must preserve decoder layer order as contiguous "
            "PP stage segments"
        )

    collective_algorithm = source.collective_algorithm
    if collective_algorithm not in {"auto", "ring", "tree"}:
        raise ValueError("collective_algorithm must be auto, ring, or tree")
    routing_policy = source.routing_policy
    if routing_policy != "lowest_latency":
        raise ValueError("unsupported parallel routing policy {}".format(routing_policy))
    return ParallelPlan(
        tp_degree=tp_degree,
        pp_degree=pp_degree,
        ep_degree=ep_degree,
        ranks=tuple(sorted(ranks, key=lambda rank: rank.rank)),
        layer_to_stage={
            layer.layer_id: layer_to_stage[layer.layer_id]
            for layer in execution_layers
        },
        collective_algorithm=collective_algorithm,
        routing_policy=routing_policy,
        allow_padding=allow_padding,
    )


__all__ = [
    "LogicalRank",
    "ParallelPlan",
    "ShardExtent",
    "build_parallel_plan",
    "shard_extent",
]
