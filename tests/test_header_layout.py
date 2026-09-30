"""Validate the offset constants against images packed by the bundled mkbootimg.

These are the ground truth for the layout. If mkbootimg is unavailable the whole
module skips, so it is safe to run in a checkout without ``bin/``.
"""

import os
import struct
import subprocess
from pathlib import Path

import pytest

from aik_py.types import (
    BOOT_HEADER_V1_SIZE,
    BOOT_HEADER_V2_SIZE,
    BOOT_HEADER_V3_SIZE,
    BOOT_V1_DTB_ADDR,
    BOOT_V1_DTB_SIZE,
    BOOT_V1_HEADER_SIZE,
    BOOT_V1_RECOVERY_DTBO_SIZE,
    BOOT_V3_HEADER_SIZE,
    BOOT_V3_KERNEL_SIZE,
    BOOT_V3_RAMDISK_SIZE,
    MAGIC_BOOT,
    MAGIC_VENDOR,
    VENDOR_CMDLINE_OFFSET,
    VENDOR_DTB_ADDR,
    VENDOR_DTB_SIZE,
    VENDOR_HEADER_SIZE,
    VENDOR_HEADER_SIZE_VALUE,
    VENDOR_KERNEL_ADDR,
    VENDOR_PAGE_SIZE,
    VENDOR_RAMDISK_ADDR,
    VENDOR_VENDOR_RAMDISK_SIZE,
)

pytestmark = pytest.mark.binary


def _mkbootimg() -> Path | None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("bin/linux/x86_64/mkbootimg", "bin/linux/aarch64/mkbootimg"):
        candidate = root / rel
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


MKBOOTIMG = _mkbootimg()
requires_mkbootimg = pytest.mark.skipif(
    MKBOOTIMG is None, reason="bundled mkbootimg not present"
)

PAGE = 2048
KERNEL_SIZE = 1024
RAMDISK_SIZE = 512
DTB_SIZE = 700
RECOVERY_DTBO_SIZE = 1012
DTB_OFFSET = 0x1A000000
BASE = 0x10000000


@pytest.fixture(scope="module")
def blobs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("blobs")
    (d / "kernel").write_bytes(bytes(range(256)) * (KERNEL_SIZE // 256))
    (d / "ramdisk").write_bytes(b"R" * RAMDISK_SIZE)
    (d / "recovery_dtbo").write_bytes(b"D" * RECOVERY_DTBO_SIZE)
    (d / "dtb").write_bytes(b"D" * DTB_SIZE)
    (d / "vendor_ramdisk").write_bytes(b"V" * 300)
    return d


def _pack(tmp_path: Path, name: str, *args: str, out_flag: str = "-o") -> bytes:
    out = tmp_path / name
    subprocess.run(
        [str(MKBOOTIMG), *args, out_flag, str(out)],
        capture_output=True,
        check=True,
    )
    return out.read_bytes()


def _u32(image: bytes, off: int) -> int:
    return struct.unpack_from("<I", image, off)[0]


@requires_mkbootimg
def test_v1_header_offsets(blobs: Path, tmp_path: Path):
    img = _pack(
        tmp_path,
        "b1.img",
        "--kernel",
        str(blobs / "kernel"),
        "--ramdisk",
        str(blobs / "ramdisk"),
        "--header_version",
        "1",
        "--pagesize",
        str(PAGE),
    )
    assert img[:8] == MAGIC_BOOT
    assert _u32(img, 0x08) == KERNEL_SIZE
    assert _u32(img, 0x10) == RAMDISK_SIZE
    assert _u32(img, 0x24) == PAGE
    assert _u32(img, 0x28) == 1
    assert _u32(img, BOOT_V1_HEADER_SIZE) == BOOT_HEADER_V1_SIZE


@requires_mkbootimg
def test_v2_tail_offsets(blobs: Path, tmp_path: Path):
    img = _pack(
        tmp_path,
        "b2.img",
        "--kernel",
        str(blobs / "kernel"),
        "--ramdisk",
        str(blobs / "ramdisk"),
        "--dtb",
        str(blobs / "dtb"),
        "--recovery_dtbo",
        str(blobs / "recovery_dtbo"),
        "--dtb_offset",
        hex(DTB_OFFSET),
        "--header_version",
        "2",
        "--pagesize",
        str(PAGE),
    )
    assert _u32(img, BOOT_V1_RECOVERY_DTBO_SIZE) == RECOVERY_DTBO_SIZE
    assert _u32(img, BOOT_V1_DTB_SIZE) == DTB_SIZE
    assert _u32(img, BOOT_V1_DTB_ADDR) == BASE + DTB_OFFSET
    assert _u32(img, BOOT_V1_HEADER_SIZE) == BOOT_HEADER_V2_SIZE


@requires_mkbootimg
def test_v3_sparse_layout(blobs: Path, tmp_path: Path):
    img = _pack(
        tmp_path,
        "b3.img",
        "--kernel",
        str(blobs / "kernel"),
        "--ramdisk",
        str(blobs / "ramdisk"),
        "--header_version",
        "3",
        "--pagesize",
        str(PAGE),
    )
    assert _u32(img, BOOT_V3_KERNEL_SIZE) == KERNEL_SIZE
    assert _u32(img, BOOT_V3_RAMDISK_SIZE) == RAMDISK_SIZE
    assert _u32(img, BOOT_V3_HEADER_SIZE) == BOOT_HEADER_V3_SIZE


@requires_mkbootimg
def test_vendor_header_offsets(blobs: Path, tmp_path: Path):
    img = _pack(
        tmp_path,
        "v.img",
        "--kernel",
        str(blobs / "kernel"),
        "--vendor_ramdisk",
        str(blobs / "vendor_ramdisk"),
        "--vendor_cmdline",
        "vendorcmdline",
        "--dtb",
        str(blobs / "dtb"),
        "--header_version",
        "3",
        "--pagesize",
        str(PAGE),
        out_flag="--vendor_boot",
    )
    assert img[:8] == MAGIC_VENDOR
    assert _u32(img, VENDOR_PAGE_SIZE) == PAGE
    assert _u32(img, VENDOR_KERNEL_ADDR) == BASE + 0x8000
    assert _u32(img, VENDOR_RAMDISK_ADDR) == BASE + 0x1000000
    assert _u32(img, VENDOR_VENDOR_RAMDISK_SIZE) == 300
    assert img[VENDOR_CMDLINE_OFFSET : VENDOR_CMDLINE_OFFSET + 14] == (
        b"vendorcmdline\x00"
    )
    assert _u32(img, VENDOR_HEADER_SIZE) == VENDOR_HEADER_SIZE_VALUE
    assert _u32(img, VENDOR_DTB_SIZE) == DTB_SIZE
    assert _u32(img, VENDOR_DTB_ADDR) == BASE + 0x1F00000
