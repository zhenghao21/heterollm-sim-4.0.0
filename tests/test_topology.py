import unittest

from heterollm_sim.ir import ComponentSpec, HardwareSpec, LinkSpec, PortSpec
from heterollm_sim.topology import (
    TopologyValidationError,
    assert_valid_topology,
    validate_topology,
)


def _port(
    port_id,
    protocol,
    role,
    *,
    direction="bidirectional",
    version="1.0",
    lanes=16,
    bandwidth_gbps=64.0,
    max_links=1,
    payload=None
):
    return PortSpec(
        port_id=port_id,
        protocol=protocol,
        role=role,
        direction=direction,
        version=version,
        lanes=lanes,
        bandwidth_gbps=bandwidth_gbps,
        max_links=max_links,
        payload=payload,
    )


def _two_component_hardware(protocol, left_role, right_role, **link_overrides):
    left = ComponentSpec(
        "left",
        "gpu",
        (_port("p", protocol, left_role, version="5.0"),),
        package_id="pkg0",
        die_id="die0",
    )
    right = ComponentSpec(
        "right",
        "accelerator",
        (_port("p", protocol, right_role, version="5.0"),),
        package_id="pkg0",
        die_id="die1",
    )
    link_values = dict(
        link_id="link",
        source_component="left",
        source_port="p",
        target_component="right",
        target_port="p",
        protocol=protocol,
        version="5.0",
        lanes=16,
        bandwidth_gbps=64.0,
        latency_ns=10.0,
        bidirectional=True,
    )
    link_values.update(link_overrides)
    return HardwareSpec("pair", (left, right), (LinkSpec(**link_values),))


class ValidTopologyTests(unittest.TestCase):
    def test_gpu_eight_hbm_and_ucie_cim(self):
        gpu_ports = [
            _port(
                "hbm{}".format(index),
                "HBM",
                "controller",
                version="3.0",
                bandwidth_gbps=819.0,
            )
            for index in range(8)
        ]
        gpu_ports.append(
            _port(
                "ucie0",
                "UCIe",
                "endpoint",
                version="1.1",
                bandwidth_gbps=256.0,
                payload="streaming",
            )
        )
        components = [
            ComponentSpec(
                component_id="gpu0",
                kind="gpu",
                ports=tuple(gpu_ports),
                package_id="package0",
                die_id="gpu_die",
                capacity_bytes=64 * 1024 * 1024,
            )
        ]
        links = []
        for index in range(8):
            hbm_id = "hbm{}".format(index)
            components.append(
                ComponentSpec(
                    component_id=hbm_id,
                    kind="hbm",
                    ports=(
                        _port(
                            "host",
                            "HBM",
                            "device",
                            version="3.0",
                            bandwidth_gbps=819.0,
                        ),
                    ),
                    package_id="package0",
                    die_id="{}_die".format(hbm_id),
                    capacity_bytes=16 * 1024**3,
                )
            )
            links.append(
                LinkSpec(
                    link_id="gpu_{}".format(hbm_id),
                    source_component="gpu0",
                    source_port=hbm_id,
                    target_component=hbm_id,
                    target_port="host",
                    protocol="HBM",
                    version="3.0",
                    lanes=16,
                    bandwidth_gbps=819.0,
                    latency_ns=40.0,
                )
            )

        components.append(
            ComponentSpec(
                component_id="cim0",
                kind="sram_cim",
                ports=(
                    _port(
                        "ucie0",
                        "UCIe",
                        "endpoint",
                        version="1.1",
                        bandwidth_gbps=256.0,
                        payload="streaming",
                    ),
                ),
                package_id="package0",
                die_id="cim_die",
                capacity_bytes=256 * 1024 * 1024,
            )
        )
        links.append(
            LinkSpec(
                link_id="gpu_cim",
                source_component="gpu0",
                source_port="ucie0",
                target_component="cim0",
                target_port="ucie0",
                protocol="UCIe",
                version="1.1",
                lanes=16,
                bandwidth_gbps=256.0,
                latency_ns=15.0,
                payload="streaming",
            )
        )
        hardware = HardwareSpec("gpu-hbm-cim", tuple(components), tuple(links))

        report = validate_topology(hardware)

        self.assertTrue(report.is_valid, report.format())
        assert_valid_topology(hardware)


class InvalidTopologyTests(unittest.TestCase):
    def assert_has_code(self, hardware, expected_code):
        report = validate_topology(hardware)
        self.assertFalse(report.is_valid)
        self.assertIn(expected_code, report.error_codes, report.format())
        issue = next(item for item in report.errors if item.code == expected_code)
        self.assertRegex(issue.message, r"[\u3400-\u9fff]")
        self.assertTrue(issue.message_en)
        self.assertIn("拓扑校验失败", report.format())
        self.assertIn("topology validation failed", report.format_en())
        with self.assertRaises(TopologyValidationError):
            assert_valid_topology(hardware)

    def test_protocol_roles(self):
        cases = (
            ("pcie", "endpoint", "endpoint", "pcie_role_pair"),
            ("cxl", "host", "host", "cxl_role_pair"),
            ("nvlink", "host", "device", "nvlink_endpoint_role"),
        )
        for protocol, left_role, right_role, expected in cases:
            with self.subTest(protocol=protocol):
                self.assert_has_code(
                    _two_component_hardware(protocol, left_role, right_role), expected
                )

    def test_hbm_must_use_dedicated_protocol(self):
        hardware = _two_component_hardware("pcie", "root", "endpoint")
        hbm = ComponentSpec(
            "right",
            "hbm",
            hardware.components[1].ports,
            package_id="pkg0",
            die_id="die1",
            capacity_bytes=1024,
        )
        hardware = HardwareSpec("bad-hbm", (hardware.components[0], hbm), hardware.links)
        self.assert_has_code(hardware, "hbm_non_dedicated_protocol")

    def test_ucie_requires_same_package_different_die_and_payload(self):
        hardware = _two_component_hardware("ucie", "endpoint", "endpoint")
        right = ComponentSpec(
            "right",
            "accelerator",
            hardware.components[1].ports,
            package_id="pkg1",
            die_id="die0",
        )
        hardware = HardwareSpec("bad-ucie", (hardware.components[0], right), hardware.links)
        report = validate_topology(hardware)
        self.assertIn("ucie_cross_package", report.error_codes)
        self.assertIn("ucie_same_die", report.error_codes)
        self.assertIn("ucie_payload_missing", report.error_codes)

    def test_lane_direction_and_version_checks(self):
        hardware = _two_component_hardware(
            "pcie", "root", "endpoint", lanes=32, version="6.0"
        )
        left = hardware.components[0]
        right_port = _port(
            "p",
            "pcie",
            "endpoint",
            direction="output",
            version="5.0",
            lanes=16,
        )
        right = ComponentSpec(
            "right",
            "accelerator",
            (right_port,),
            package_id="pkg0",
            die_id="die1",
        )
        hardware = HardwareSpec("bad-link", (left, right), hardware.links)
        report = validate_topology(hardware)
        self.assertIn("lanes_exceed_port", report.error_codes)
        self.assertIn("version_unsupported", report.error_codes)
        self.assertIn("target_direction", report.error_codes)
        self.assertIn("bidirectional_port_required", report.error_codes)

    def test_link_bandwidth_may_not_exceed_port(self):
        hardware = _two_component_hardware(
            "pcie", "root", "endpoint", bandwidth_gbps=128.0
        )
        self.assert_has_code(hardware, "bandwidth_exceeds_port")

    def test_port_occupancy_connectivity_and_capacity(self):
        shared = _port("p", "nvlink", "endpoint", max_links=1)
        components = (
            ComponentSpec("a", "gpu", (shared,), capacity_bytes=-1),
            ComponentSpec("b", "gpu", (_port("p", "nvlink", "endpoint"),)),
            ComponentSpec("c", "gpu", (_port("p", "nvlink", "endpoint"),)),
            ComponentSpec("isolated", "gpu"),
        )
        links = (
            LinkSpec("ab", "a", "p", "b", "p", "nvlink"),
            LinkSpec("ac", "a", "p", "c", "p", "nvlink"),
        )
        report = validate_topology(HardwareSpec("bad-resources", components, links))
        self.assertIn("negative_capacity", report.error_codes)
        self.assertIn("port_overoccupied", report.error_codes)
        self.assertIn("disconnected_topology", report.error_codes)


if __name__ == "__main__":
    unittest.main()
