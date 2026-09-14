"""Offline, source-backed presets for HardwareIR communication protocols.

The catalog deliberately separates three bandwidth scopes:

* ``raw_gbps`` is the physical/signalling calculation before line or packet
  overhead where the public source exposes enough information;
* ``effective_one_way_gbps`` is the usable one-direction value only when it
  can be derived without inventing a workload-dependent payload efficiency;
* ``aggregate_bidirectional_gbps`` is the sum of both directions and must not
  be copied into :class:`~heterollm_sim.ir.LinkSpec.bandwidth_gbps`.

``simulation_defaults`` uses the simulator's internal rate boundary and represents
one direction.  Every field remains a starting point: the Web UI exposes it as
a manual override before creating a port/link pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


CATALOG_VERSION = "1.0.0"
CATALOG_CUTOFF_AT = "2026-08-22T00:00:00Z"

S1_STANDARD = "S1_STANDARD"
S2_VENDOR_DECLARED = "S2_VENDOR_DECLARED"
A_ANALYTICAL = "A_ANALYTICAL"


@dataclass(frozen=True)
class ProtocolSource:
    title: str
    url: str
    publisher: str
    evidence_level: str
    accessed_at: str = "2026-08-22"

    def to_metadata(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "publisher": self.publisher,
            "evidence_level": self.evidence_level,
            "accessed_at": self.accessed_at,
        }


@dataclass(frozen=True)
class ProtocolPresetDefinition:
    preset_id: str
    name: str
    protocol: str
    version: str
    organization: str
    transfer_unit: str
    transfer_unit_count: int
    transfer_unit_semantics: str
    io_speed_value: float
    io_speed_unit: str
    raw_gbps: Optional[float]
    effective_one_way_gbps: Optional[float]
    aggregate_bidirectional_gbps: Optional[float]
    displayed_bandwidth_scope: str
    simulation_bandwidth_gbps: float
    simulation_latency_ns: float
    port_roles: Tuple[str, str]
    payload: Optional[str]
    derivation: str
    limitations: Tuple[str, ...]
    sources: Tuple[ProtocolSource, ...]
    evidence_level: str = S1_STANDARD


def _source(
    title: str,
    url: str,
    publisher: str,
    evidence_level: str = S1_STANDARD,
) -> ProtocolSource:
    return ProtocolSource(title, url, publisher, evidence_level)


JEDEC_HBM3 = _source(
    "JEDEC JESD238B.01 High Bandwidth Memory DRAM (HBM3)",
    "https://www.jedec.org/standards-documents/docs/jesd238b01",
    "JEDEC",
)
JEDEC_HBM4 = _source(
    "JEDEC JESD270-4A High Bandwidth Memory DRAM (HBM4)",
    "https://www.jedec.org/standards-documents/docs/jesd270-4a",
    "JEDEC",
)
NVIDIA_H200 = _source(
    "NVIDIA H200 Tensor Core GPU specifications",
    "https://www.nvidia.com/en-us/data-center/h200/",
    "NVIDIA",
    S2_VENDOR_DECLARED,
)
PCI_SIG_PCIE5 = _source(
    "PCI Express 5.0 Specification",
    "https://pcisig.com/pci-express-50-specification",
    "PCI-SIG",
)
CXL_3 = _source(
    "Compute Express Link 3.0 Specification",
    "https://computeexpresslink.org/cxl-specification/",
    "CXL Consortium",
)
UCIE_2 = _source(
    "UCIe 2.0 Specification",
    "https://www.uciexpress.org/specifications",
    "UCIe Consortium",
)
NVIDIA_H100 = _source(
    "NVIDIA H100 Tensor Core GPU specifications",
    "https://www.nvidia.com/en-us/data-center/h100/",
    "NVIDIA",
    S2_VENDOR_DECLARED,
)
JEDEC_HBM2E = _source(
    "JEDEC JESD235D High Bandwidth Memory DRAM",
    "https://www.jedec.org/standards-documents/docs/jesd235d",
    "JEDEC",
)
NVIDIA_GH200 = _source(
    "NVIDIA Grace Hopper Superchip specifications",
    "https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/",
    "NVIDIA",
    S2_VENDOR_DECLARED,
)
AMD_MI300X = _source(
    "AMD Instinct MI300X accelerator specifications",
    "https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html",
    "AMD",
    S2_VENDOR_DECLARED,
)
INTEL_GAUDI3 = _source(
    "Intel Gaudi 3 AI accelerator specifications",
    "https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi.html",
    "Intel",
    S2_VENDOR_DECLARED,
)


_PRESETS: Tuple[ProtocolPresetDefinition, ...] = (
    ProtocolPresetDefinition(
        preset_id="hbm2e-3_2-1024",
        name="HBM2E 3.2 GT/s · 1024 DQ",
        protocol="HBM",
        version="HBM2E",
        organization="JEDEC",
        transfer_unit="channel",
        transfer_unit_count=8,
        transfer_unit_semantics="PortSpec.lanes 表示八个独立 HBM 通道，不表示 1024 根 DQ 引脚。",
        io_speed_value=3.2,
        io_speed_unit="GT/s per DQ pin",
        raw_gbps=3276.8,
        effective_one_way_gbps=3276.8,
        aggregate_bidirectional_gbps=None,
        displayed_bandwidth_scope="单个 HBM2E 堆叠的一向峰值为 409.6 GB/s；读写共享接口，不能相加。",
        simulation_bandwidth_gbps=3276.8,
        simulation_latency_ns=45.0,
        port_roles=("controller", "device"),
        payload=None,
        derivation="3.2 GT/s × 1024 DQ = 409.6 GB/s；HBM 内存接口不采用 PCIe 式线路编码扣减。",
        limitations=(
            "未扣除刷新、ECC、命令调度、控制器、封装与热限制。",
            "lanes 字段保存独立通道数；DQ 宽度保留在目录说明中。",
        ),
        sources=(JEDEC_HBM2E,),
    ),
    ProtocolPresetDefinition(
        preset_id="hbm3-6_4-1024",
        name="HBM3 6.4 GT/s · 1024 DQ",
        protocol="HBM",
        version="HBM3",
        organization="JEDEC",
        transfer_unit="channel",
        transfer_unit_count=16,
        transfer_unit_semantics="PortSpec.lanes 在 HBM 中表示 16 个独立通道，不表示 1024 根 DQ 引脚。",
        io_speed_value=6.4,
        io_speed_unit="GT/s per DQ pin",
        raw_gbps=6553.6,
        effective_one_way_gbps=6553.6,
        aggregate_bidirectional_gbps=None,
        displayed_bandwidth_scope="单个 HBM3 堆叠的单向峰值；读写共用双向 DQ，不把读带宽与写带宽相加。",
        simulation_bandwidth_gbps=6553.6,
        simulation_latency_ns=40.0,
        port_roles=("controller", "device"),
        payload=None,
        derivation="6.4 GT/s × 1024 DQ = 819.2 GB/s。HBM 不使用 PCIe 式线路编码扣减。",
        limitations=(
            "未扣除刷新、ECC、命令总线、控制器调度、封装和热限制。",
            "HBM 通道由两个伪通道共享命令与数据资源；仿真中的 lanes 字段只保存独立通道数。",
        ),
        sources=(JEDEC_HBM3,),
    ),
    ProtocolPresetDefinition(
        preset_id="hbm3e-h200-stack-slice",
        name="H200 HBM3E · 六堆叠均分切片",
        protocol="HBM",
        version="HBM3E",
        organization="NVIDIA",
        transfer_unit="channel",
        transfer_unit_count=16,
        transfer_unit_semantics="一个 GPU 控制器端口对应一个 HBM3E 堆叠；lanes 仍按 16 个 HBM 通道建模。",
        io_speed_value=800.0,
        io_speed_unit="GB/s per stack slice (derived)",
        raw_gbps=None,
        effective_one_way_gbps=6400.0,
        aggregate_bidirectional_gbps=None,
        displayed_bandwidth_scope="H200 官方 4.8 TB/s 总显存带宽按六个堆叠等分后的单堆叠分析值。",
        simulation_bandwidth_gbps=6400.0,
        simulation_latency_ns=40.0,
        port_roles=("controller", "device"),
        payload=None,
        derivation="4.8 TB/s ÷ 6 = 单堆叠 0.8 TB/s = 800 GB/s。",
        limitations=(
            "NVIDIA 未在产品页公开每个物理堆叠的独立带宽，均分仅用于系统级拓扑建模。",
            "141 GB 可见容量不等于六个 DRAM 堆叠的原始物理容量，组合预设按可见容量等分。",
        ),
        sources=(NVIDIA_H200,),
        evidence_level=S2_VENDOR_DECLARED,
    ),
    ProtocolPresetDefinition(
        preset_id="hbm4-8_0-2048",
        name="HBM4 8.0 GT/s · 2048 DQ",
        protocol="HBM",
        version="HBM4",
        organization="JEDEC",
        transfer_unit="channel",
        transfer_unit_count=32,
        transfer_unit_semantics="PortSpec.lanes 表示 32 个独立通道；HBM4 的数据接口宽度为 2048 DQ。",
        io_speed_value=8.0,
        io_speed_unit="GT/s per DQ pin",
        raw_gbps=16384.0,
        effective_one_way_gbps=16384.0,
        aggregate_bidirectional_gbps=None,
        displayed_bandwidth_scope="单个 HBM4 堆叠的 JEDEC 基准单向峰值，读写不可同时相加。",
        simulation_bandwidth_gbps=16384.0,
        simulation_latency_ns=40.0,
        port_roles=("controller", "device"),
        payload=None,
        derivation="8.0 GT/s × 2048 DQ = 2.048 TB/s。",
        limitations=("未扣除刷新、ECC、控制器、封装和热限制。",),
        sources=(JEDEC_HBM4,),
    ),
    ProtocolPresetDefinition(
        preset_id="pcie-5_0-x16",
        name="PCIe 5.0 ×16",
        protocol="PCIe",
        version="5.0",
        organization="PCI-SIG",
        transfer_unit="lane",
        transfer_unit_count=16,
        transfer_unit_semantics="一条 lane 含一对发送差分线与一对接收差分线；×16 在两个方向各有 16 lane。",
        io_speed_value=32.0,
        io_speed_unit="GT/s per lane per direction",
        raw_gbps=512.0,
        effective_one_way_gbps=504.12307692307695,
        aggregate_bidirectional_gbps=1008.2461538461539,
        displayed_bandwidth_scope="×16 单向线路容量，已扣除 128b/130b 线码；未扣除 TLP/DLLP、流控和软件开销。",
        simulation_bandwidth_gbps=504.12307692307695,
        simulation_latency_ns=150.0,
        port_roles=("root", "endpoint"),
        payload=None,
        derivation="32 GT/s × 16 lane = 原始一向 64 GB/s；×128/130 = 一向 63.015 GB/s；双向聚合为 126.031 GB/s。",
        limitations=("协议有效负载取决于 TLP 尺寸、包头、重放、流控、拓扑和软件栈。",),
        sources=(PCI_SIG_PCIE5,),
    ),
    ProtocolPresetDefinition(
        preset_id="cxl-3_0-x16",
        name="CXL 3.0 ×16 / 64 GT/s",
        protocol="CXL",
        version="3.0",
        organization="CXL Consortium",
        transfer_unit="lane",
        transfer_unit_count=16,
        transfer_unit_semantics="复用 PCIe 6.0 物理层；×16 表示每方向各 16 lane，CXL.io、CXL.cache 与 CXL.mem 共享链路。",
        io_speed_value=64.0,
        io_speed_unit="GT/s per lane per direction",
        raw_gbps=1024.0,
        effective_one_way_gbps=None,
        aggregate_bidirectional_gbps=2048.0,
        displayed_bandwidth_scope="×16 物理层单向理论值；双向聚合只用于展示，CXL 有效负载因 FLIT 与事务类型而异。",
        simulation_bandwidth_gbps=1024.0,
        simulation_latency_ns=180.0,
        port_roles=("host", "device"),
        payload=None,
        derivation="64 GT/s × 16 lane = 一向物理层 128 GB/s；双向物理层聚合为 256 GB/s。",
        limitations=(
            "没有用一个固定比例伪造 CXL.io/cache/mem 的通用有效负载；simulation_defaults 因而是物理层上限。",
            "实际时延与带宽取决于设备类型、交换层级、一致性流量、FLIT 占用和内存介质。",
        ),
        sources=(CXL_3,),
    ),
    ProtocolPresetDefinition(
        preset_id="ucie-2_0-standard-x64",
        name="UCIe 2.0 标准封装 ×64",
        protocol="UCIe",
        version="2.0",
        organization="UCIe Consortium",
        transfer_unit="lane",
        transfer_unit_count=64,
        transfer_unit_semantics="×64 表示一个标准封装模块每方向 64 条主带数据 lane；发送与接收资源分离。",
        io_speed_value=32.0,
        io_speed_unit="GT/s per lane per direction",
        raw_gbps=2048.0,
        effective_one_way_gbps=None,
        aggregate_bidirectional_gbps=4096.0,
        displayed_bandwidth_scope="标准封装 ×64 的单向物理层理论值；协议/Raw/Streaming 映射的有效负载分别计算。",
        simulation_bandwidth_gbps=2048.0,
        simulation_latency_ns=20.0,
        port_roles=("endpoint", "endpoint"),
        payload="streaming",
        derivation="UCIe 2.0 的 32 GT/s × 64 lane = 一向物理层 256 GB/s；双向聚合为 512 GB/s。",
        limitations=(
            "simulation_defaults 采用 streaming payload；选择 PCIe/CXL 或 Raw 映射时必须手动覆盖 payload。",
            "未扣除 FLIT、CRC、重试、训练、降宽和封装实现开销。",
        ),
        sources=(UCIE_2,),
    ),
    ProtocolPresetDefinition(
        preset_id="nvlink-4-h100-18",
        name="NVLink 4 · H100 18-link",
        protocol="NVLink",
        version="4.0",
        organization="NVIDIA",
        transfer_unit="link",
        transfer_unit_count=18,
        transfer_unit_semantics="PortSpec.lanes 在该厂商聚合预设中表示 18 条 NVLink，而不是协议未公开的内部物理 lane 数。",
        io_speed_value=25.0,
        io_speed_unit="GB/s per link per direction (derived)",
        raw_gbps=None,
        effective_one_way_gbps=3600.0,
        aggregate_bidirectional_gbps=7200.0,
        displayed_bandwidth_scope="单个 H100 GPU 的 18-link 聚合：450 GB/s 单向、900 GB/s 双向聚合。",
        simulation_bandwidth_gbps=3600.0,
        simulation_latency_ns=100.0,
        port_roles=("endpoint", "endpoint"),
        payload=None,
        derivation="NVIDIA 公布 900 GB/s 双向聚合；÷2 得 450 GB/s 单向，÷18 得每 link 每方向 25 GB/s。",
        limitations=(
            "NVIDIA 产品页未公开足以独立计算线路编码效率的物理 lane 细节，因此 raw_gbps 留空。",
            "该预设代表单 GPU 的全部 18-link 聚合，不代表任意两个 GPU 之间必然同时获得全部带宽。",
        ),
        sources=(NVIDIA_H100,),
        evidence_level=S2_VENDOR_DECLARED,
    ),
    ProtocolPresetDefinition(
        preset_id="nvlink-c2c-gh200",
        name="GH200 NVLink-C2C coherent link",
        protocol="NVLink-C2C",
        version="GH200",
        organization="NVIDIA",
        transfer_unit="coherent_chip_to_chip_link",
        transfer_unit_count=1,
        transfer_unit_semantics="一个逻辑端口表示 Grace CPU 与 Hopper GPU 之间完整的一致性芯片互连。",
        io_speed_value=900.0,
        io_speed_unit="GB/s aggregate bidirectional vendor value",
        raw_gbps=None,
        effective_one_way_gbps=3600.0,
        aggregate_bidirectional_gbps=7200.0,
        displayed_bandwidth_scope="公开值为双向聚合 900 GB/s；仿真边界按对称方向假设采用一向 450 GB/s。",
        simulation_bandwidth_gbps=3600.0,
        simulation_latency_ns=50.0,
        port_roles=("endpoint", "endpoint"),
        payload=None,
        derivation="公开双向聚合 900 GB/s ÷ 2 = 一向 450 GB/s；换算后写入 IR 内部单位。",
        limitations=(
            "一向 450 GB/s 来自双向聚合值的对称方向分析假设，调用方可按实测通信方向覆盖。",
            "未公开的线路编码、封装通道、协议开销与一致性流量竞争不做反向推断。",
        ),
        sources=(NVIDIA_GH200,),
        evidence_level=S2_VENDOR_DECLARED,
    ),
    ProtocolPresetDefinition(
        preset_id="infinity-fabric-mi300x-envelope",
        name="MI300X Infinity Fabric analysis envelope",
        protocol="InfinityFabric",
        version="MI300X",
        organization="AMD",
        transfer_unit="accelerator_endpoint_aggregate",
        transfer_unit_count=8,
        transfer_unit_semantics="八个公开 Infinity Fabric 端点折叠为一个加速器级逻辑端口，不表示封装内部 lane。",
        io_speed_value=896.0,
        io_speed_unit="GB/s aggregate bidirectional vendor value",
        raw_gbps=None,
        effective_one_way_gbps=3584.0,
        aggregate_bidirectional_gbps=7168.0,
        displayed_bandwidth_scope="厂商公开双向聚合 896 GB/s；仿真默认按对称方向分析为一向 448 GB/s。",
        simulation_bandwidth_gbps=3584.0,
        simulation_latency_ns=100.0,
        port_roles=("endpoint", "endpoint"),
        payload=None,
        derivation="公开双向聚合 896 GB/s ÷ 2 = 一向 448 GB/s；该推导不是 Infinity Fabric 标准线路值。",
        limitations=(
            "AMD 未公开足以还原端点分配、线路编码与任意两卡并发带宽的完整物理参数。",
            "一向值、时延与非阻塞行为都是可覆盖分析默认，不得视为统一协议标准。",
        ),
        sources=(AMD_MI300X,),
        evidence_level=A_ANALYTICAL,
    ),
    ProtocolPresetDefinition(
        preset_id="roce-v2-gaudi3-8x200gbe-envelope",
        name="Gaudi 3 RoCEv2 · 8-port endpoint envelope",
        protocol="RoCE",
        version="v2",
        organization="Intel",
        transfer_unit="ethernet_port",
        transfer_unit_count=8,
        transfer_unit_semantics="一个逻辑端点聚合八个以太网端口；每端口每方向标称 25 GB/s。",
        io_speed_value=25.0,
        io_speed_unit="GB/s per Ethernet port per direction",
        raw_gbps=1600.0,
        effective_one_way_gbps=None,
        aggregate_bidirectional_gbps=3200.0,
        displayed_bandwidth_scope="八端口物理层聚合的一向上限为 200 GB/s；未伪造 RoCE 有效负载效率。",
        simulation_bandwidth_gbps=1600.0,
        simulation_latency_ns=800.0,
        port_roles=("endpoint", "switch"),
        payload=None,
        derivation="8 × 每端口每方向 25 GB/s = 聚合一向 200 GB/s；双向物理层合计 400 GB/s。",
        limitations=(
            "八端口分配是八卡参考底板的分析 envelope，不代表 Intel 规定所有 24 个集成端口的唯一分组。",
            "RoCE 有效吞吐取决于以太网帧、拥塞控制、交换网络、RDMA 消息尺寸与软件栈。",
        ),
        sources=(INTEL_GAUDI3,),
        evidence_level=A_ANALYTICAL,
    ),
    ProtocolPresetDefinition(
        preset_id="lpddr5x-gh200-aggregate",
        name="GH200 LPDDR5X aggregate · 500 GB/s",
        protocol="LPDDR5X",
        version="GH200",
        organization="NVIDIA",
        transfer_unit="superchip_memory_subsystem",
        transfer_unit_count=1,
        transfer_unit_semantics="一个端口表示 Grace CPU 的完整 LPDDR5X 内存子系统，不表示单颗 DRAM 或物理通道。",
        io_speed_value=500.0,
        io_speed_unit="GB/s aggregate memory bandwidth",
        raw_gbps=None,
        effective_one_way_gbps=4000.0,
        aggregate_bidirectional_gbps=None,
        displayed_bandwidth_scope="GH200 产品级 LPDDR5X 内存总带宽 500 GB/s；读写共享内存控制器资源。",
        simulation_bandwidth_gbps=4000.0,
        simulation_latency_ns=80.0,
        port_roles=("controller", "device"),
        payload=None,
        derivation="直接采用厂商公布的产品级 500 GB/s 内存带宽；换算后写入 IR 内部单位。",
        limitations=(
            "该 envelope 只适用于 GH200 的聚合内存子系统，不是任意 LPDDR5X 器件或通道的标准预设。",
            "未扣除刷新、ECC、一致性流量、控制器调度、NUMA 与热限制。",
        ),
        sources=(NVIDIA_GH200,),
        evidence_level=S2_VENDOR_DECLARED,
    ),
)


_BY_ID: Mapping[str, ProtocolPresetDefinition] = {
    item.preset_id: item for item in _PRESETS
}
if len(_BY_ID) != len(_PRESETS):
    raise RuntimeError("通信协议预设 ID 重复")


def _display_gbs(value_gbps: Optional[float]) -> Optional[float]:
    return None if value_gbps is None else value_gbps / 8.0


def _metadata(item: ProtocolPresetDefinition) -> Dict[str, Any]:
    return {
        "id": item.preset_id,
        "name": item.name,
        "protocol": item.protocol,
        "version": item.version,
        "organization": item.organization,
        "transfer_unit": item.transfer_unit,
        "transfer_unit_count": item.transfer_unit_count,
        "transfer_unit_semantics": item.transfer_unit_semantics,
        "io_speed": {"value": item.io_speed_value, "unit": item.io_speed_unit},
        "bandwidth": {
            "raw_gbps": item.raw_gbps,
            "raw_gbs": _display_gbs(item.raw_gbps),
            "effective_one_way_gbps": item.effective_one_way_gbps,
            "effective_one_way_gbs": _display_gbs(item.effective_one_way_gbps),
            "aggregate_bidirectional_gbps": item.aggregate_bidirectional_gbps,
            "aggregate_bidirectional_gbs": _display_gbs(item.aggregate_bidirectional_gbps),
        },
        "displayed_bandwidth_scope": item.displayed_bandwidth_scope,
        "evidence_level": item.evidence_level,
        "sources": [source.to_metadata() for source in item.sources],
        "limitations": list(item.limitations),
        "catalog_version": CATALOG_VERSION,
    }


def _simulation_defaults(item: ProtocolPresetDefinition) -> Dict[str, Any]:
    source_role, target_role = item.port_roles
    common = {
        "protocol": item.protocol,
        "version": item.version,
        "lanes": item.transfer_unit_count,
        "bandwidth_gbps": item.simulation_bandwidth_gbps,
    }
    if item.payload:
        common["payload"] = item.payload
    metadata = {
        "protocol_preset_id": item.preset_id,
        "displayed_bandwidth_scope": item.displayed_bandwidth_scope,
        "bandwidth_semantics": "one_way_capacity",
        "manual_override_allowed": True,
    }
    return {
        "source_port": {
            **common,
            "role": source_role,
            "direction": "bidirectional",
            "max_links": 1,
            "metadata": metadata,
        },
        "target_port": {
            **common,
            "role": target_role,
            "direction": "bidirectional",
            "max_links": 1,
            "metadata": metadata,
        },
        "link": {
            **common,
            "latency_ns": item.simulation_latency_ns,
            "bidirectional": True,
            "metadata": metadata,
        },
        "manual_override": {
            "allowed": True,
            "fields": ["version", "lanes", "bandwidth_gbps", "latency_ns", "payload"],
            "note": "目录值是公开规格的系统级起点；可按具体器件、拓扑、降宽、协议开销或测量结果覆盖。",
        },
    }


def list_protocol_presets() -> Tuple[Dict[str, Any], ...]:
    """Return stable, compact catalog metadata ordered by preset ID."""

    return tuple(_metadata(item) for item in sorted(_PRESETS, key=lambda value: value.preset_id))


def protocol_preset_filters() -> Dict[str, Sequence[str]]:
    items = list_protocol_presets()
    return {
        "protocol": sorted({item["protocol"] for item in items}),
        "organization": sorted({item["organization"] for item in items}),
        "version": sorted({item["version"] for item in items}),
    }


def protocol_preset_page(
    *,
    query: str = "",
    protocol: str = "",
    organization: str = "",
) -> Dict[str, Any]:
    normalized_query = query.strip().lower()
    normalized_protocol = protocol.strip().lower()
    normalized_organization = organization.strip().lower()
    items = []
    for item in list_protocol_presets():
        if normalized_protocol and item["protocol"].lower() != normalized_protocol:
            continue
        if normalized_organization and item["organization"].lower() != normalized_organization:
            continue
        haystack = " ".join(
            [
                item["id"],
                item["name"],
                item["protocol"],
                item["version"],
                item["organization"],
                item["transfer_unit_semantics"],
            ]
        ).lower()
        if normalized_query and normalized_query not in haystack:
            continue
        items.append(item)
    return {
        "items": items,
        "total": len(items),
        "filters": protocol_preset_filters(),
        "catalog": {"version": CATALOG_VERSION, "cutoff_at": CATALOG_CUTOFF_AT},
    }


def get_protocol_preset(preset_id: str) -> ProtocolPresetDefinition:
    return _BY_ID[preset_id]


def protocol_preset_detail(preset_id: str) -> Dict[str, Any]:
    item = get_protocol_preset(preset_id)
    return {
        "preset": _metadata(item),
        "derivation": item.derivation,
        "simulation_defaults": _simulation_defaults(item),
        "catalog": {"version": CATALOG_VERSION, "cutoff_at": CATALOG_CUTOFF_AT},
    }


__all__ = [
    "A_ANALYTICAL",
    "CATALOG_CUTOFF_AT",
    "CATALOG_VERSION",
    "ProtocolPresetDefinition",
    "ProtocolSource",
    "get_protocol_preset",
    "list_protocol_presets",
    "protocol_preset_detail",
    "protocol_preset_filters",
    "protocol_preset_page",
]
