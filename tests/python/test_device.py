import binstatcuda as bsc


def test_device_count_returns_int():
    count = bsc.device_count()
    assert isinstance(count, int)
