"""Tests for AOSP boot image header_version 0-2.

The byte offsets asserted here were derived empirically from images packed by the
AIK-bundled ``bin/linux/x86_64/mkbootimg`` and cross-checked with its
``unpackbootimg``. They match upstream AOSP ``mkbootimg.py``, including the
16-byte name / 512-byte cmdline pairing that older AOSP docs get wrong.
"""

from __future__ import annotations

import hashlib
import pathlib
import random
import struct
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aik_py.formats.aosp import (
    BootImage,
    BootImageError,
    BootImageHeader,
    OsVersion,
    pack,
    parse,
)

KERNEL = bytes(range(256)) * 8  # 2048 bytes
RAMDISK = b"ramdisk-payload" * 7  # 105 bytes, not a page multiple
SECOND = b"second!" * 6  # 42 bytes
RDTBO = b"dtbo" * 19  # 76 bytes
DTB = b"d" * 30


def make_image(
    header_version: int = 2,
    page_size: int = 4096,
    hashtype: str = "sha1",
    kernel: bytes = KERNEL,
    ramdisk: bytes = RAMDISK,
    second: bytes | None = None,
    recovery_dtbo: bytes | None = None,
    dtb: bytes | None = None,
    **header_kwargs,
) -> BootImage:
    kwargs = {
        "kernel_addr": 0x80008000,
        "ramdisk_addr": 0x81000000,
        "second_addr": 0x80F00000,
        "tags_addr": 0x80000100,
        "name": b"testboard",
        "cmdline": b"console=ttyS0",
        "os_version": OsVersion(11, 0, 0, 21, 8),
    }
    kwargs.update(header_kwargs)
    return BootImage(
        header=BootImageHeader(
            page_size=page_size,
            header_version=header_version,
            hashtype=hashtype,
            dtb_addr=0x01F00000,
            recovery_dtbo_offset=0x4000,
            **kwargs,
        ),
        kernel=kernel,
        ramdisk=ramdisk,
        second=second,
        recovery_dtbo=recovery_dtbo,
        dtb=dtb,
    )


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hv", (0, 1, 2))
@pytest.mark.parametrize("page_size", (2048, 4096, 8192, 16384))
def test_roundtrip_all_header_fields(hv: int, page_size: int) -> None:
    img = make_image(
        header_version=hv,
        page_size=page_size,
        second=SECOND if hv >= 1 else None,
        recovery_dtbo=RDTBO if hv >= 1 else None,
        dtb=DTB if hv >= 2 else None,
    )
    out = parse(pack(img))

    assert out.header.header_version == hv
    assert out.header.page_size == page_size
    assert out.header.kernel_addr == img.header.kernel_addr
    assert out.header.ramdisk_addr == img.header.ramdisk_addr
    assert out.header.second_addr == img.header.second_addr
    assert out.header.tags_addr == img.header.tags_addr
    assert out.header.name == img.header.name
    assert out.header.cmdline == img.header.cmdline
    assert out.header.extra_cmdline == img.header.extra_cmdline
    assert out.header.os_version == img.header.os_version
    assert out.header.hashtype == img.header.hashtype
    # An unset id is computed during pack, so compare against a repack instead.
    assert pack(out) == pack(img)
    # dtb_addr only exists on disk from v2 on, so only round-trips there.
    if hv >= 2:
        assert out.header.dtb_addr == img.header.dtb_addr
    if hv >= 1:
        assert out.header.recovery_dtbo_offset == img.header.recovery_dtbo_offset

    assert out.kernel == img.kernel
    assert out.ramdisk == img.ramdisk
    assert out.second == img.second
    assert out.recovery_dtbo == img.recovery_dtbo
    assert out.dtb == img.dtb


@pytest.mark.parametrize("hv", (0, 1, 2))
def test_pack_is_byte_stable(hv: int) -> None:
    """Packing a parsed image reproduces the original bytes exactly."""
    img = make_image(
        header_version=hv,
        second=SECOND if hv >= 1 else None,
        recovery_dtbo=RDTBO if hv >= 1 else None,
        dtb=DTB if hv >= 2 else None,
    )
    first = pack(img)
    assert pack(parse(first)) == first


@pytest.mark.parametrize("hv", (0, 1, 2))
def test_absent_optional_payloads_roundtrip_as_none(hv: int) -> None:
    img = make_image(header_version=hv, second=None, recovery_dtbo=None, dtb=None)
    out = parse(pack(img))
    assert out.second is None
    assert out.recovery_dtbo is None
    assert out.dtb is None
    assert out.kernel == KERNEL
    assert out.ramdisk == RAMDISK


# --------------------------------------------------------------------------
# exact byte offsets
# --------------------------------------------------------------------------


def test_header_fields_land_at_verified_offsets() -> None:
    img = make_image(
        header_version=2,
        page_size=4096,
        hashtype="sha256",
        second=SECOND,
        recovery_dtbo=RDTBO,
        dtb=DTB,
    )
    data = pack(img)

    assert data[0x00:0x08] == b"ANDROID!"
    assert struct.unpack_from("<I", data, 0x08)[0] == len(KERNEL)
    assert struct.unpack_from("<I", data, 0x0C)[0] == 0x80008000
    assert struct.unpack_from("<I", data, 0x10)[0] == len(RAMDISK)
    assert struct.unpack_from("<I", data, 0x14)[0] == 0x81000000
    assert struct.unpack_from("<I", data, 0x18)[0] == len(SECOND)
    assert struct.unpack_from("<I", data, 0x1C)[0] == 0x80F00000
    assert struct.unpack_from("<I", data, 0x20)[0] == 0x80000100
    assert struct.unpack_from("<I", data, 0x24)[0] == 4096
    assert struct.unpack_from("<I", data, 0x28)[0] == 2
    assert struct.unpack_from("<I", data, 0x2C)[0] == OsVersion(11, 0, 0, 21, 8).pack()

    # name is 16 bytes, cmdline 512 -- not the 512/1024 older docs describe
    assert data[0x30:0x40] == b"testboard".ljust(16, b"\x00")
    assert data[0x40:0x240] == b"console=ttyS0".ljust(512, b"\x00")
    assert data[0x260:0x660] == b"\x00" * 1024

    # id occupies all 32 bytes under sha256
    assert parse(data).header.hashtype == "sha256"
    assert len(data[0x240:0x260]) == 32
    assert data[0x240:0x260] != b"\x00" * 32

    # v1/v2 extension block
    assert struct.unpack_from("<I", data, 0x660)[0] == len(RDTBO)
    assert struct.unpack_from("<Q", data, 0x664)[0] == 0x4000
    assert struct.unpack_from("<I", data, 0x66C)[0] == 1660
    assert struct.unpack_from("<I", data, 0x670)[0] == len(DTB)
    assert struct.unpack_from("<Q", data, 0x674)[0] == 0x01F00000


@pytest.mark.parametrize(
    ("hv", "expected"),
    [(0, 1592), (1, 1648), (2, 1660)],
)
def test_header_size_per_version(hv: int, expected: int) -> None:
    img = make_image(header_version=hv, second=SECOND, dtb=DTB if hv >= 2 else None)
    data = pack(img)
    if hv >= 1:
        assert struct.unpack_from("<I", data, 0x66C)[0] == expected
    # Everything from the reported size to the end of page 0 is zero padding.
    assert data[expected:4096] == b"\x00" * (4096 - expected)


def test_v0_writes_no_header_size_field() -> None:
    """v0 predates header_size; upstream leaves the slot zero."""
    data = pack(make_image(header_version=0))
    assert struct.unpack_from("<I", data, 0x66C)[0] == 0


# --------------------------------------------------------------------------
# payload ordering and page alignment
# --------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", (2048, 4096))
@pytest.mark.parametrize("hv", (0, 1, 2))
def test_payload_order_and_page_alignment(hv: int, page_size: int) -> None:
    img = make_image(
        header_version=hv,
        page_size=page_size,
        second=SECOND if hv >= 1 else None,
        recovery_dtbo=RDTBO if hv >= 1 else None,
        dtb=DTB if hv >= 2 else None,
    )
    data = pack(img)

    def align(n: int) -> int:
        return -(-n // page_size) * page_size

    expected_order = [KERNEL, RAMDISK]
    if hv >= 1:
        expected_order.append(SECOND)
        expected_order.append(RDTBO)
    if hv >= 2:
        expected_order.append(DTB)

    cursor = page_size
    for payload in expected_order:
        assert data[cursor : cursor + len(payload)] == payload
        cursor += len(payload)
        # any padding up to the next page boundary is zero
        padded = align(cursor)
        assert data[cursor:padded] == b"\x00" * (padded - cursor)
        cursor = padded

    # the file ends exactly at the last payload's page boundary
    assert len(data) == cursor


@pytest.mark.parametrize("page_size", (2048, 4096))
@pytest.mark.parametrize("size", (1, 100, 2047, 2048, 2049, 4095, 4096, 4097))
def test_payload_sizes_align_independently(page_size: int, size: int) -> None:
    img = make_image(header_version=1, page_size=page_size, kernel=b"\xa5" * size)
    out = parse(pack(img))
    assert out.kernel == b"\xa5" * size
    assert out.ramdisk == RAMDISK


def test_payload_order_kernel_ramdisk_second_dtb() -> None:
    img = make_image(header_version=2, second=SECOND, recovery_dtbo=RDTBO, dtb=DTB)
    data = pack(img)
    assert data.index(KERNEL) == 4096
    assert data.index(RAMDISK) == 8192
    assert data.index(SECOND) == 12288
    assert data.index(RDTBO) == 16384
    assert data.index(DTB) == 20480


# --------------------------------------------------------------------------
# hashtype
# --------------------------------------------------------------------------


def test_sha1_id_is_20_bytes_with_zero_padding() -> None:
    data = pack(make_image(header_version=2, hashtype="sha1", second=SECOND, dtb=DTB))
    raw = data[0x240:0x260]
    assert raw[:20] != b"\x00" * 20
    assert raw[20:32] == b"\x00" * 12

    img = parse(data)
    assert img.header.hashtype == "sha1"
    assert len(img.header.id) == 20


def test_sha256_id_occupies_all_32_bytes() -> None:
    data = pack(make_image(header_version=2, hashtype="sha256", second=SECOND, dtb=DTB))
    raw = data[0x240:0x260]
    assert any(b != 0 for b in raw[20:32])

    img = parse(data)
    assert img.header.hashtype == "sha256"
    assert len(img.header.id) == 32


def test_caller_supplied_id_is_used_verbatim() -> None:
    supplied = bytes(range(32))
    img = make_image(header_version=2, second=SECOND, dtb=DTB, id=supplied)
    assert pack(img)[0x240:0x260] == supplied


def test_computed_id_matches_aosp_payload_digest() -> None:
    """The v0-v2 id is the AOSP payload digest: each component's bytes then a u32 size.

    Slots are conditional on header_version, which is what the bundled tools do.
    """
    img = make_image(
        header_version=2, hashtype="sha256", second=SECOND, recovery_dtbo=RDTBO, dtb=DTB
    )
    data = pack(img)

    expected = hashlib.sha256()
    for component in (KERNEL, RAMDISK, SECOND, RDTBO, DTB):
        expected.update(component)
        expected.update(struct.pack("<I", len(component)))

    assert data[0x240:0x260] == expected.digest()


# --------------------------------------------------------------------------
# base + offset decomposition
# --------------------------------------------------------------------------


def test_base_plus_offset_resolves_to_addresses() -> None:
    base = 0x80000000
    img = make_image(
        header_version=1,
        kernel_addr=base + 0x00008000,
        ramdisk_addr=base + 0x01000000,
        second_addr=base + 0x00F00000,
        tags_addr=base + 0x00000100,
        second=SECOND,
    )
    out = parse(pack(img))
    assert out.header.kernel_addr - base == 0x00008000
    assert out.header.ramdisk_addr - base == 0x01000000
    assert out.header.second_addr - base == 0x00F00000
    assert out.header.tags_addr - base == 0x00000100


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0x16000158, OsVersion(11, 0, 0, 21, 8)),
        (0x1C00097B, OsVersion(14, 0, 1, 23, 11)),
        (0x14081801, OsVersion(10, 2, 3, 0, 1)),
        (0x1E3111EC, OsVersion(15, 12, 34, 30, 12)),
    ],
)
def test_os_version_encoding(raw: int, expected: OsVersion) -> None:
    assert OsVersion.unpack(raw) == expected
    assert expected.pack() == raw


def test_os_version_reports_calendar_values() -> None:
    v = OsVersion.unpack(0x16000158)
    assert (v.major, v.minor, v.patch) == (11, 0, 0)
    assert v.year == 2021
    assert v.month == 8


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_rejects_bad_magic() -> None:
    data = bytearray(pack(make_image()))
    data[0:8] = b"NOTANDRO"
    with pytest.raises(BootImageError, match="bad magic"):
        parse(bytes(data))


def test_rejects_file_too_short_for_magic() -> None:
    with pytest.raises(BootImageError, match="truncated"):
        parse(b"AND")


def test_rejects_truncated_payload() -> None:
    data = pack(make_image(header_version=1, second=SECOND))
    with pytest.raises(BootImageError, match="truncated|runs past"):
        parse(data[: len(data) - 8192])


def test_rejects_truncated_header() -> None:
    data = pack(make_image(header_version=2, second=SECOND, dtb=DTB))
    with pytest.raises(BootImageError, match="header bytes|truncated"):
        parse(data[:1000])


def test_rejects_absurd_page_size() -> None:
    data = bytearray(pack(make_image()))
    struct.pack_into("<I", data, 0x24, 0x7FFFFFFF)
    with pytest.raises(BootImageError, match="truncated|runs past|page_size"):
        parse(bytes(data))


def test_rejects_page_size_not_multiple_of_four() -> None:
    data = bytearray(pack(make_image()))
    struct.pack_into("<I", data, 0x24, 4095)
    with pytest.raises(BootImageError, match="page_size"):
        parse(bytes(data))


def test_rejects_size_exceeding_file() -> None:
    data = bytearray(pack(make_image()))
    struct.pack_into("<I", data, 0x08, 0x7FFFFFFF)  # kernel_size
    with pytest.raises(BootImageError, match="truncated|runs past"):
        parse(bytes(data))


def test_rejects_unsupported_header_version() -> None:
    data = bytearray(pack(make_image(header_version=2)))
    struct.pack_into("<I", data, 0x28, 3)
    with pytest.raises(BootImageError, match="unsupported header_version"):
        parse(bytes(data))


def test_pack_rejects_page_size_smaller_than_the_header() -> None:
    """A 1660-byte header cannot live in a 1024-byte page."""
    with pytest.raises(BootImageError, match="smaller than"):
        pack(make_image(header_version=2, page_size=1024, second=SECOND, dtb=DTB))


def test_pack_accepts_minimum_viable_page_size() -> None:
    """2048 fits the largest v2 header, so it must work."""
    for hv in (0, 1, 2):
        img = make_image(
            header_version=hv,
            page_size=2048,
            second=SECOND if hv >= 1 else None,
            recovery_dtbo=RDTBO if hv >= 1 else None,
            dtb=DTB if hv >= 2 else None,
        )
        out = parse(pack(img))
        assert out.header.page_size == 2048
        assert out.kernel == KERNEL


def test_pack_rejects_oversized_board_name() -> None:
    """The 16-byte name field must keep a NUL; the tool rejects 16 characters."""
    with pytest.raises(BootImageError, match="board name"):
        pack(make_image(name=b"x" * 16))
    # 15 fits
    assert pack(make_image(name=b"x" * 15))


def test_pack_rejects_oversized_cmdline() -> None:
    with pytest.raises(BootImageError, match="cmdline"):
        pack(make_image(cmdline=b"x" * 513))


def test_full_width_cmdline_roundtrips_without_a_terminator() -> None:
    """mkbootimg fills the 512-byte cmdline edge to edge, leaving no NUL."""
    full = b"z" * 512
    data = pack(make_image(cmdline=full))
    assert data[0x40:0x240] == full
    assert parse(data).header.cmdline == full


def test_maximum_length_board_name_roundtrips() -> None:
    data = pack(make_image(name=b"n" * 15))
    assert data[0x30:0x40] == b"n" * 15 + b"\x00"
    assert parse(data).header.name == b"n" * 15


def test_zero_length_payloads_are_handled() -> None:
    data = pack(make_image(kernel=b"", ramdisk=b""))
    assert len(data) == 4096  # just the header page
    out = parse(data)
    assert out.kernel == b""
    assert out.ramdisk == b""


def test_zero_length_optional_payload_reads_back_as_none() -> None:
    out = parse(pack(make_image(header_version=1, second=b"")))
    assert out.second is None


def test_pack_rejects_dtb_below_v2() -> None:
    img = make_image(header_version=1, dtb=DTB)
    with pytest.raises(BootImageError, match="dtb requires header_version 2"):
        pack(img)


def test_pack_rejects_recovery_dtbo_below_v1() -> None:
    img = make_image(header_version=0, recovery_dtbo=RDTBO)
    with pytest.raises(BootImageError, match="recovery_dtbo requires header_version"):
        pack(img)


def test_corrupt_header_never_leaks_a_raw_error() -> None:
    """Every malformed byte must surface as BootImageError, not struct/IndexError."""
    data = bytearray(
        pack(make_image(header_version=2, second=SECOND, recovery_dtbo=RDTBO, dtb=DTB))
    )
    rng = random.Random(20240930)
    for _ in range(2000):
        corrupt = bytearray(data)
        offset = rng.randrange(0, min(0x700, len(corrupt)))
        corrupt[offset] ^= 0xFF
        try:
            parse(bytes(corrupt))
        except BootImageError:
            pass


def test_truncation_mid_payload_is_rejected() -> None:
    """Cutting into any payload must raise, not silently return short data."""
    full = pack(
        make_image(header_version=2, second=SECOND, recovery_dtbo=RDTBO, dtb=DTB)
    )
    # stop short of the last payload, which begins on a page boundary
    last_payload_start = full.rindex(DTB)
    for n in range(0, last_payload_start, 512):
        with pytest.raises(BootImageError):
            parse(full[:n])


# --------------------------------------------------------------------------
# page_size fallback
# --------------------------------------------------------------------------


def test_zero_page_size_field_falls_back_to_4096() -> None:
    """unpackbootimg substitutes 4096 for a zero page_size in the v0-v2 path."""
    data = bytearray(pack(make_image(header_version=1, page_size=4096, second=SECOND)))
    struct.pack_into("<I", data, 0x24, 0)
    img = parse(bytes(data))

    assert img.header.page_size == 4096
    # and the payloads still land where a 4096 grid puts them
    assert img.kernel == KERNEL
    assert img.ramdisk == RAMDISK
    assert img.second == SECOND


def test_non_zero_page_size_is_recorded_verbatim() -> None:
    """The fallback is only for a zero field; 2048 must survive a round trip."""
    data = pack(make_image(header_version=1, page_size=2048, second=SECOND))
    assert struct.unpack_from("<I", data, 0x24)[0] == 2048
    assert parse(data).header.page_size == 2048
    assert parse(data).kernel == KERNEL
    assert parse(data).ramdisk == RAMDISK


def test_pack_defaults_page_size_when_zero() -> None:
    img = make_image(header_version=1, page_size=0, second=SECOND)
    data = pack(img)
    assert struct.unpack_from("<I", data, 0x24)[0] == 4096
    assert parse(data).header.page_size == 4096


# --------------------------------------------------------------------------
# optional oracle cross-check
# --------------------------------------------------------------------------

ORACLE_DIR = pathlib.Path("/tmp/aik-aosp")


@pytest.mark.skipif(
    not ORACLE_DIR.is_dir(), reason="bundled mkbootimg fixtures not built"
)
@pytest.mark.parametrize("hv", (0, 1, 2))
def test_matches_bundled_mkbootimg_output(hv: int) -> None:
    """Images produced by the AIK-bundled tool must parse to the same values."""
    raw = (ORACLE_DIR / f"hv{hv}.img").read_bytes()
    img = parse(raw)

    assert img.kernel == (ORACLE_DIR / "k.bin").read_bytes()
    assert img.ramdisk == (ORACLE_DIR / "rd.bin").read_bytes()
    assert img.second == (ORACLE_DIR / "s.bin").read_bytes()
    assert img.header.name == b"testboard"
    assert img.header.cmdline == b"console=ttyS0"

    # our repack is byte-identical to what the C tool emitted
    assert pack(img) == raw


@pytest.mark.skipif(
    not ORACLE_DIR.is_dir(), reason="bundled mkbootimg fixtures not built"
)
def test_matches_bundled_sha1_output() -> None:
    raw = (ORACLE_DIR / "hv2sha1.img").read_bytes()
    img = parse(raw)
    assert img.header.hashtype == "sha1"
    assert len(img.header.id) == 20
    assert img.recovery_dtbo == (ORACLE_DIR / "rdtb.bin").read_bytes()
    assert img.dtb == (ORACLE_DIR / "d.bin").read_bytes()
    assert pack(img) == raw
