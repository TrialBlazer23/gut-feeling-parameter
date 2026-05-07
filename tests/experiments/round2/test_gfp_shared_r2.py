import numpy as np
import pytest
from experiments.round2.gfp_shared_r2 import smooth, last_n

def test_smooth_shorter_than_window():
    arr = [1, 2, 3]
    w = 4
    result = smooth(arr, w)
    assert result == [1, 2, 3]
    assert isinstance(result, list)

def test_smooth_equal_to_window():
    arr = [1, 2, 3, 4]
    w = 4
    result = smooth(arr, w)
    assert len(result) == 1
    assert pytest.approx(result[0]) == 2.5  # (1+2+3+4)/4

def test_smooth_longer_than_window():
    arr = [1, 2, 3, 4, 5]
    w = 3
    # windows: [1,2,3], [2,3,4], [3,4,5]
    # means: 2.0, 3.0, 4.0
    result = smooth(arr, w)
    assert len(result) == 3
    assert pytest.approx(result) == [2.0, 3.0, 4.0]

def test_smooth_empty():
    assert smooth([], w=4) == []

def test_smooth_w1():
    arr = [1, 2, 3]
    result = smooth(arr, w=1)
    assert pytest.approx(result) == [1.0, 2.0, 3.0]

def test_smooth_constant():
    arr = [10] * 10
    w = 5
    result = smooth(arr, w)
    assert len(result) == 6
    assert all(pytest.approx(x) == 10.0 for x in result)

def test_last_n_empty():
    assert last_n([]) == 0.0

def test_last_n_short_list():
    ep_list = [{"prec_det": 0.5}, {"prec_det": 1.0}]
    # n defaults to 8
    assert last_n(ep_list) == 0.75

def test_last_n_long_list():
    ep_list = [{"prec_det": float(i)} for i in range(10)]
    # last 8: 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
    # sum = 2+3+4+5+6+7+8+9 = 44
    # mean = 44 / 8 = 5.5
    assert last_n(ep_list, n=8) == 5.5

def test_last_n_custom_key():
    ep_list = [{"other": 10.0}, {"other": 20.0}]
    assert last_n(ep_list, n=2, key="other") == 15.0
