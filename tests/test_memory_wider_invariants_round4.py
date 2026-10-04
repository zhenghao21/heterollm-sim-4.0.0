"""Additional passing core/state checks; synthetic finite geometries only."""
from dataclasses import replace
from itertools import product
import pytest
from heterollm_sim.memory_mapping import map_dram_address, map_nand_address
from heterollm_sim.memory_types import AccessRequest, DramConfig, NandConfig
from heterollm_sim.dram_core import DramCore
from heterollm_sim.nand_core import NandCore
from heterollm_sim.data_motion import PhysicalRuntimeContext


def test_dram_geometry_grid_is_bijective():
    for channels,banks,ranks,stacks,dies,interleave in product((1,2),(1,2,4),(1,2),(1,2),(1,2),(64,128,256)):
        cfg=DramConfig(channels=channels,banks_per_group=banks,ranks_per_channel=ranks,
                       stacks=stacks,dies_per_stack=dies,interleave_bytes=interleave,
                       rows_per_bank=4,row_bytes=256,burst_bytes=64)
        positions=set()
        for a in range(0,cfg.capacity_bytes,64):
            m=map_dram_address(cfg,a)
            pos=tuple(getattr(m,k) for k in ('stack','die','lane','rank','bank_group','bank','row','column'))
            assert m.stack < stacks and m.die < dies and m.row < cfg.rows_per_bank
            assert m.lane < cfg.lane_count and m.rank < ranks and m.bank < banks
            assert pos not in positions, (cfg,a,m)
            positions.add(pos)
        assert len(positions)==cfg.capacity_bytes//64


def test_nand_geometry_grid_is_bijective():
    for channels,targets,dies,luns,planes,independent in product((1,2),(1,2),(1,2),(1,2),(1,2),(False,True)):
        cfg=NandConfig(channels=channels,targets_per_channel=targets,dies_per_target=dies,
                       luns_per_die=luns,planes_per_lun=planes,planes_independent=independent,
                       blocks_per_plane=2,pages_per_block=2,page_bytes=512,
                       host_granularity_bytes=512)
        positions=set()
        for a in range(0,cfg.capacity_bytes,512):
            m=map_nand_address(cfg,a)
            pos=tuple(getattr(m,k) for k in ('channel','target','die','lun','plane','block','page'))
            assert m.channel < channels and m.target < targets and m.die < dies
            assert m.lun < luns and m.plane < planes and m.block < 2 and m.page < 2
            assert pos not in positions, (cfg,a,m)
            positions.add(pos)
        assert len(positions)==cfg.capacity_bytes//512


@pytest.mark.parametrize('kind',('dram','nand'))
def test_detail_retention_does_not_change_core_completion(kind):
    cfg=(DramConfig(channels=2,banks_per_group=2,max_expanded_segments=2)
         if kind=='dram' else NandConfig(channels=2,page_bytes=512,
              host_granularity_bytes=512,max_expanded_segments=2))
    cls=DramCore if kind=='dram' else NandCore
    req=AccessRequest('big','read',0,4096)
    bounded=cls(cfg).execute(req)
    full=cls(replace(cfg,max_expanded_segments=10000)).execute(req)
    assert bounded.counters['details_truncated'] is True
    assert not bounded.stages and not bounded.mapping
    assert bounded.completion_ns==full.completion_ns
    assert bounded.counters['resource_busy_ns']==full.counters['resource_busy_ns']
    assert bounded.counters['resource_last_intervals']==full.counters['resource_last_intervals']


def test_separate_run_contexts_keep_separate_state():
    cfg=DramConfig(banks_per_group=1,row_bytes=128)
    a,b=PhysicalRuntimeContext(),PhysicalRuntimeContext()
    ca,cb=a.runtime(cfg,'dram').core,b.runtime(cfg,'dram').core
    r0=ca.execute(AccessRequest('first','read',0,64))
    r1=ca.execute(AccessRequest('again','read',64,64,r0.completion_ns))
    r2=cb.execute(AccessRequest('other-run','read',64,64))
    assert r1.counters['row_hits']==1
    assert r2.counters['row_misses']==1
    assert ca is not cb
