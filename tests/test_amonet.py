import os

import pytest

from aik_py.patches import amonet

SIZES = (0, 1, 512, 1023, 1024, 1025, 2047, 2048, 2049, 3072, 4096, 100_000)


def test_round_trip() -> None:
    for size in SIZES:
        data = os.urandom(size)
        stripped, micro = amonet.strip(data)
        assert amonet.apply(stripped, micro) == data, f"failed at size {size}"


def test_strip_matches_bash_dd_pipeline() -> None:
    data = bytes(range(256)) * 16
    # Bash: first 1 KiB is the microloader, everything from 1024 on is the head
    # stub (1024..2047) followed by the tail (2048..).
    microloader = data[:1024]
    head_stub = data[1024:2048]
    tail = data[2048:]
    stripped, micro = amonet.strip(data)
    assert micro == microloader
    assert stripped == head_stub + tail


def test_microloader_is_fully_reclaimed() -> None:
    data = os.urandom(4096)
    _, micro = amonet.strip(data)
    assert len(micro) == amonet.BLOCK
    assert micro not in data[amonet.BLOCK :]


def test_apply_prepends_microloader() -> None:
    body = os.urandom(2048)
    micro = os.urandom(1024)
    assert amonet.apply(body, micro) == micro + body


@pytest.mark.parametrize("size", [0, 1, 512, 1023, 1024, 2047])
def test_short_input_raises_in_strict_mode(size: int) -> None:
    with pytest.raises(ValueError):
        amonet.strip(os.urandom(size), strict=True)


def test_oversized_microloader_raises() -> None:
    with pytest.raises(ValueError):
        amonet.apply(os.urandom(2048), os.urandom(1025))
