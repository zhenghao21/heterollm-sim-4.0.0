"""Packed-row references used by the independent CPU GET_ROWS probe."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
from probe_synthetic_get_rows import decode_rows, synthetic_rows


def test_q4k_reference_decodes_scale_min_and_both_nibbles():
    packed=np.zeros((1,144),np.uint8)
    packed[0,:2]=np.frombuffer(np.float16(1).tobytes(),np.uint8)
    packed[0,2:4]=np.frombuffer(np.float16(1).tobytes(),np.uint8)
    packed[0,4:8]=1; packed[0,8:12]=2; packed[0,12:16]=0x21
    packed[0,16:]=0x43
    out=decode_rows('Q4_K',packed,256)[0]
    for group in range(8):
        np.testing.assert_array_equal(out[group*32:(group+1)*32],1 if group%2==0 else 2)


def test_q6k_reference_decodes_signed_values_and_scale_groups():
    packed=np.zeros((1,210),np.uint8)
    packed[0,:128]=0x21; packed[0,128:192]=0xE4
    packed[0,192:208]=1
    packed[0,208:]=np.frombuffer(np.float16(1).tobytes(),np.uint8)
    out=decode_rows('Q6_K',packed,256)[0]
    for half in range(2):
        for group,value in enumerate((-31,-15,2,18)):
            np.testing.assert_array_equal(out[half*128+group*32:half*128+(group+1)*32],value)


@pytest.mark.parametrize('fmt',['F16','F32','Q4_K','Q6_K'])
def test_probe_generated_rows_are_finite_and_repeatable(fmt):
    data=synthetic_rows(fmt,1024,4,np.random.default_rng(1))
    again=synthetic_rows(fmt,1024,4,np.random.default_rng(1))
    np.testing.assert_array_equal(data,again)
    selected=decode_rows(fmt,data[[2,0,2]],1024)
    assert np.isfinite(selected).all()
    np.testing.assert_array_equal(selected[0],selected[2])
