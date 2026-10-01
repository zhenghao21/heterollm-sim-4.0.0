from tools.extract_cuda_resource_usage import parse_resource_usage


def test_resource_parser_preserves_arch_and_specializations():
    raw = """arch = sm_120a
 Function q4_m1:
  REG:53 STACK:16 SHARED:1408 LOCAL:0 CONSTANT[0]:1056
 Function q4_m2:
  REG:80 STACK:16 SHARED:2560 LOCAL:0 CONSTANT[0]:1056
arch = sm_90
 Function another:
  REG:32 STACK:0 SHARED:0 LOCAL:0
"""
    rows = parse_resource_usage(raw)
    assert [r['registers_per_thread'] for r in rows] == [53, 80, 32]
    assert [r['shared_bytes'] for r in rows] == [1408, 2560, 0]
    assert [r['arch'] for r in rows] == ['sm_120a', 'sm_120a', 'sm_90']
    assert rows[0]['function'] == 'q4_m1'
