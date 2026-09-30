"""Tests for the AOSP vendor_boot (VNDRBOOT) codec.

Every offset asserted here was verified against real vendor_boot images on this
machine and against images produced by the bundled ``bin/linux/x86_64/mkbootimg``,
not read off a header table. See ``aik_py.formats.vendor`` for the layout notes.
"""

from __future__ import annotations

import pathlib
import struct

import pytest

from aik_py.formats import vendor
from aik_py.formats.vendor import (
    VendorBootImage,
    VendorRamdiskFragment,
    pack,
    parse,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
# Resolve the bundled oracle tools relative to the repo, not an absolute path,
# so these tests actually run in CI instead of silently skipping.
VENDOR_BIN = REPO_ROOT / "bin" / "linux" / "x86_64"
REAL_IMAGES = [
    pathlib.Path.home() / "Downloads" / "vendor_boot.img",
    pathlib.Path.home() / "Downloads" / "vendor_boot(1).img",
]


def _sample(version: int = 3, page_size: int = 4096, **overrides) -> VendorBootImage:
    fields = {
        "header_version": version,
        "page_size": page_size,
        "kernel_addr": 0x40008000,
        "ramdisk_addr": 0x41000000,
        "tags_addr": 0x40000100,
        "dtb_addr": 0x41F00000,
        "vendor_cmdline": b"androidboot.serialconsole=0",
        "name": b"testboard",
        "vendor_ramdisk": bytes(range(256)) * 2,
        "dtb": b"\xd0\xd2\xe2\xef" + bytes(range(64)),
    }
    fields.update(overrides)
    return VendorBootImage(**fields)


# --------------------------------------------------------------------------
# round trips
# --------------------------------------------------------------------------


@pytest.mark.parametrize("version", [3, 4])
def test_round_trip_preserves_every_field(version: int) -> None:
    image = _sample(version)
    parsed = parse(pack(image))

    assert parsed.header_version == version
    assert parsed.page_size == image.page_size
    assert parsed.kernel_addr == image.kernel_addr
    assert parsed.ramdisk_addr == image.ramdisk_addr
    assert parsed.tags_addr == image.tags_addr
    assert parsed.dtb_addr == image.dtb_addr
    assert parsed.vendor_cmdline == image.vendor_cmdline
    assert parsed.name == image.name
    assert parsed.vendor_ramdisk == image.vendor_ramdisk
    assert parsed.dtb == image.dtb


@pytest.mark.parametrize("version", [3, 4])
def test_round_trip_is_byte_identical(version: int) -> None:
    first = pack(_sample(version))
    assert pack(parse(first)) == first


@pytest.mark.parametrize("version", [3, 4])
def test_vendor_boot_has_no_kernel_or_hashtype(version: int) -> None:
    """The Bash unpackbootimg emits no -kernel and no -hashtype for VNDRBOOT."""
    image = parse(pack(_sample(version)))
    assert not hasattr(image, "kernel")
    assert not hasattr(image, "hashtype")
    assert not hasattr(image, "os_version")


# --------------------------------------------------------------------------
# header offsets
# --------------------------------------------------------------------------


def test_header_fields_land_at_verified_offsets() -> None:
    image = _sample(3)
    raw = pack(image)
    u32 = lambda off: struct.unpack_from("<I", raw, off)[0]

    assert raw[0x000:0x008] == b"VNDRBOOT"
    assert u32(0x008) == 3
    assert u32(0x00C) == 4096
    assert u32(0x010) == 0x40008000  # kernel_addr
    assert u32(0x014) == 0x41000000  # ramdisk_addr
    assert u32(0x018) == len(image.vendor_ramdisk)  # vendor_ramdisk_size
    assert raw[0x01C : 0x01C + 27] == b"androidboot.serialconsole=0"
    assert u32(0x81C) == 0x40000100  # tags_addr
    assert raw[0x820:0x829] == b"testboard"
    assert u32(0x830) == vendor.HEADER_SIZE_V3  # header_size
    assert u32(0x834) == len(image.dtb)  # dtb_size
    # dtb_addr is 64-bit, not 32-bit as older header tables suggest
    assert struct.unpack_from("<Q", raw, 0x838)[0] == 0x41F00000


def test_v4_only_fields_sit_after_the_v3_header() -> None:
    raw = pack(_sample(4, fragments=(VendorRamdiskFragment(b"abcd", 1, b"frag"),)))
    u32 = lambda off: struct.unpack_from("<I", raw, off)[0]

    assert u32(0x830) == vendor.HEADER_SIZE_V4  # 2128
    assert u32(0x840) == 108  # vendor_ramdisk_table_size
    assert u32(0x844) == 1  # entry_num
    assert u32(0x848) == 108  # entry_size
    assert u32(0x84C) == 0  # bootconfig_size


def test_fixed_width_fields_are_nul_padded() -> None:
    raw = pack(_sample(3))
    cmdline_len = len(b"androidboot.serialconsole=0")
    assert raw[0x1C + cmdline_len : 0x81C] == b"\x00" * (
        vendor.CMDLINE_SIZE - cmdline_len
    )
    name_len = len(b"testboard")
    assert raw[0x820 + name_len : 0x830] == b"\x00" * (vendor.NAME_SIZE - name_len)


# --------------------------------------------------------------------------
# page alignment
# --------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [2048, 4096])
@pytest.mark.parametrize("ramdisk_len", [1, 100, 4096, 5000])
def test_payloads_start_on_page_boundaries(page_size: int, ramdisk_len: int) -> None:
    image = _sample(3, page_size=page_size, vendor_ramdisk=b"\xa5" * ramdisk_len)
    raw = pack(image)
    parsed = parse(raw)

    assert parsed.vendor_ramdisk == image.vendor_ramdisk
    assert parsed.dtb == image.dtb

    header = raw[: vendor.HEADER_SIZE_V3]
    assert len(header) % 4 == 0
    payload_start = -(-vendor.HEADER_SIZE_V3 // page_size) * page_size
    assert raw[payload_start : payload_start + ramdisk_len] == image.vendor_ramdisk

    dtb_start = -(-(payload_start + ramdisk_len) // page_size) * page_size
    assert raw[dtb_start : dtb_start + len(image.dtb)] == image.dtb


@pytest.mark.parametrize("page_size", [2048, 4096])
def test_total_length_is_page_aligned(page_size: int) -> None:
    image = _sample(
        3, page_size=page_size, vendor_ramdisk=b"\x5a" * 3333, dtb=b"\x11" * 777
    )
    raw = pack(image)
    assert len(raw) % page_size == 0


def test_odd_page_size_is_rejected() -> None:
    with pytest.raises(vendor.ImageValueError, match="power of two"):
        pack(_sample(3, page_size=3000))


# --------------------------------------------------------------------------
# v4 vendor ramdisk fragment table
# --------------------------------------------------------------------------


def test_v4_fragments_round_trip() -> None:
    fragments = (
        VendorRamdiskFragment(b"first-fragment", 1, b"platform"),
        VendorRamdiskFragment(b"second-fragment", 3, b"dlkm", board_id=(0, 1, 0, 2)),
    )
    image = _sample(4, fragments=fragments, dtb=b"\xd0\xd2\xe2\xef" * 40)
    parsed = parse(pack(image))

    assert len(parsed.fragments) == 2
    assert parsed.fragments[0].data == b"first-fragment"
    assert parsed.fragments[0].name == b"platform"
    assert parsed.fragments[0].ramdisk_type == 1
    assert parsed.fragments[1].data == b"second-fragment"
    assert parsed.fragments[1].name == b"dlkm"
    assert parsed.fragments[1].ramdisk_type == 3
    # board_id slots are positional: a 0 means "unused", so it must survive
    assert parsed.fragments[1].board_id == (
        0,
        1,
        0,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert parsed.vendor_ramdisk == b"first-fragmentsecond-fragment"


def test_fragment_table_entry_layout() -> None:
    """size/offset/type uint32, 32-byte name, 16 uint32 board ids == 108 bytes."""
    image = _sample(
        4, fragments=(VendorRamdiskFragment(b"12345678", 2, b"nm", board_id=(7,)),)
    )
    raw = pack(image)
    u32 = lambda off: struct.unpack_from("<I", raw, off)[0]

    header_size = u32(0x830)
    payload = -(-header_size // image.page_size) * image.page_size
    ramdisk_len = len(image.vendor_ramdisk)
    table_start = -(-(payload + ramdisk_len) // image.page_size) * image.page_size
    if image.dtb:
        table_start = (
            -(-(table_start + len(image.dtb)) // image.page_size) * image.page_size
        )

    assert u32(table_start) == 8  # size
    assert u32(table_start + 4) == 0  # offset of the first fragment
    assert u32(table_start + 8) == 2  # type
    assert raw[table_start + 12 : table_start + 14] == b"nm"
    assert u32(table_start + 44) == 7  # first board_id word


def test_fragment_table_needs_a_v4_sized_header() -> None:
    image = _sample(
        4,
        header_size=vendor.HEADER_SIZE_V3,
        fragments=(VendorRamdiskFragment(b"abcd", 1, b"x"),),
    )
    with pytest.raises(vendor.ImageValueError, match="no room"):
        pack(image)


def test_ambiguous_v4_header_size_is_rejected() -> None:
    """2112 is valid (no v4 words); anything strictly between 2112 and 2128 is not."""
    image = _sample(4, header_size=vendor.HEADER_SIZE_V3)
    assert parse(pack(image)).header_size == vendor.HEADER_SIZE_V3

    raw = bytearray(pack(_sample(4)))
    struct.pack_into("<I", raw, 0x830, 2120)
    with pytest.raises(vendor.ImageFormatError, match="ambiguous"):
        parse(bytes(raw))
    with pytest.raises(vendor.ImageValueError, match="ambiguous"):
        pack(_sample(4, header_size=2120))


def test_dtb_addr_is_64_bit() -> None:
    """dtb_addr must round-trip values above 4 GiB, which is why it is a uint64."""
    image = _sample(4, dtb_addr=0x1_0000_0000, dtb=b"\xd0\xd2\xe2\xef" * 32)
    assert parse(pack(image)).dtb_addr == 0x1_0000_0000
    with pytest.raises(vendor.ImageValueError, match="dtb_addr"):
        pack(_sample(3, dtb_addr=0x1_0000_0000_0000_0000))


def test_page_size_must_fit_in_32_bits() -> None:
    with pytest.raises(vendor.ImageValueError, match="power of two"):
        pack(_sample(3, page_size=1 << 32))


def test_board_id_slots_are_preserved() -> None:
    """A 0 board id is an unused slot, not padding to be compacted away."""
    frag = VendorRamdiskFragment(b"abcd", 1, b"x", board_id=(0, 5, 0, 9))
    image = _sample(4, fragments=(frag,), dtb=b"\xd0\xd2\xe2\xef" * 8)
    packed = pack(image)
    parsed = parse(packed)
    assert parsed.fragments[0].board_id == (
        0,
        5,
        0,
        9,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    assert pack(parsed) == packed


# --------------------------------------------------------------------------
# oracle images built by the bundled mkbootimg
# --------------------------------------------------------------------------


def _have_oracle_vendor_tool() -> bool:
    return (VENDOR_BIN / "mkbootimg").exists()


requires_tool = pytest.mark.skipif(
    not _have_oracle_vendor_tool(), reason="bundled mkbootimg not present"
)


def _run_mkbootimg(tmp_path: pathlib.Path, version: int, page_size: int) -> bytes:
    import subprocess

    work = tmp_path / f"v{version}p{page_size}"
    work.mkdir()
    (work / "vrd.bin").write_bytes(bytes(range(100)))
    (work / "dtb.bin").write_bytes(bytes(range(200)))
    out = work / "img"
    subprocess.run(
        [
            str(VENDOR_BIN / "mkbootimg"),
            "--vendor_ramdisk",
            "vrd.bin",
            "--dtb",
            "dtb.bin",
            "--vendor_cmdline",
            "androidboot.serialconsole=0",
            "--board",
            "testboard",
            "--base",
            "0x40000000",
            "--pagesize",
            str(page_size),
            "--header_version",
            str(version),
            "--vendor_boot",
            str(out),
        ],
        cwd=work,
        check=True,
        capture_output=True,
    )
    return out.read_bytes()


@requires_tool
@pytest.mark.parametrize("version", [3, 4])
@pytest.mark.parametrize("page_size", [2048, 4096])
def test_parses_images_built_by_bundled_mkbootimg(
    tmp_path: pathlib.Path, version: int, page_size: int
) -> None:
    raw = _run_mkbootimg(tmp_path, version, page_size)
    image = parse(raw)

    assert image.header_version == version
    assert image.page_size == page_size
    assert image.vendor_cmdline == b"androidboot.serialconsole=0"
    assert image.name == b"testboard"
    assert image.vendor_ramdisk == bytes(range(100))
    assert image.dtb == bytes(range(200))

    # the bundled tool's own bytes survive a parse/pack round trip unchanged
    assert pack(image) == raw
    assert parse(pack(image)).vendor_ramdisk == bytes(range(100))
    assert parse(pack(image)).dtb == bytes(range(200))


# --------------------------------------------------------------------------
# real device images
# --------------------------------------------------------------------------


def _real_available() -> list[pathlib.Path]:
    return [p for p in REAL_IMAGES if p.exists()]


@pytest.mark.skipif(not _real_available(), reason="no real vendor_boot.img present")
@pytest.mark.parametrize("path", _real_available(), ids=lambda p: p.name)
def test_real_images_parse(path: pathlib.Path) -> None:
    raw = path.read_bytes()
    assert vendor.detect(raw)

    image = parse(raw)
    assert image.header_version == 4
    assert image.page_size == 4096
    assert image.header_size == vendor.HEADER_SIZE_V4
    assert image.vendor_ramdisk_size > 1_000_000
    assert image.dtb is not None and image.dtb_size > 0
    # real v4 images carry exactly one whole-ramdisk fragment
    assert len(image.fragments) == 1
    assert image.fragments[0].data == image.vendor_ramdisk
    assert image.cmdline
    # trailing region holds an AVB vbmeta partition the format does not describe
    assert image.trailing
    assert parse(pack(image)).vendor_ramdisk == image.vendor_ramdisk


@pytest.mark.skipif(not _real_available(), reason="no real vendor_boot.img present")
def test_real_image_preserves_trailing_bytes_on_round_trip() -> None:
    image = parse(_real_available()[0].read_bytes())
    assert parse(pack(image)).trailing == image.trailing


# --------------------------------------------------------------------------
# error handling
# --------------------------------------------------------------------------


def test_rejects_bad_magic() -> None:
    raw = bytearray(pack(_sample(3)))
    raw[0:8] = b"ANDROID!"
    with pytest.raises(vendor.ImageFormatError, match="bad magic"):
        parse(bytes(raw))


def test_rejects_truncated_header() -> None:
    raw = pack(_sample(3))
    with pytest.raises(vendor.ImageTruncatedError):
        parse(raw[:16])
    with pytest.raises(vendor.ImageTruncatedError):
        parse(raw[:40])


def test_rejects_input_shorter_than_magic() -> None:
    with pytest.raises(vendor.ImageTruncatedError):
        parse(b"VNDR")


def test_rejects_empty_input() -> None:
    with pytest.raises(vendor.ImageTruncatedError):
        parse(b"")


def test_rejects_ramdisk_larger_than_file() -> None:
    raw = bytearray(pack(_sample(3)))
    struct.pack_into("<I", raw, 0x18, 10_000_000)
    with pytest.raises(vendor.ImageTruncatedError, match="vendor ramdisk"):
        parse(bytes(raw))


def test_rejects_dtb_larger_than_file() -> None:
    raw = bytearray(pack(_sample(3)))
    struct.pack_into("<I", raw, 0x834, 10_000_000)
    with pytest.raises(vendor.ImageTruncatedError, match="dtb"):
        parse(bytes(raw))


def test_rejects_truncated_declared_payload() -> None:
    """Trimming inter-page padding is fine, but cutting a declared payload is not."""
    image = _sample(3)
    raw = pack(image)
    page = image.page_size
    align = lambda n: -(-n // page) * page
    ramdisk_start = align(vendor.HEADER_SIZE_V3)
    dtb_start = align(ramdisk_start + len(image.vendor_ramdisk))

    with pytest.raises(vendor.ImageTruncatedError, match="dtb"):
        parse(raw[: dtb_start + len(image.dtb) - 1])
    # a full-length dtb still parses
    assert parse(raw[: dtb_start + len(image.dtb)]).dtb == image.dtb


def test_rejects_unsupported_header_version() -> None:
    raw = bytearray(pack(_sample(3)))
    struct.pack_into("<I", raw, 0x08, 9)
    with pytest.raises(vendor.ImageFormatError, match="header_version"):
        parse(bytes(raw))


def test_rejects_non_power_of_two_page_size() -> None:
    raw = bytearray(pack(_sample(3)))
    struct.pack_into("<I", raw, 0x0C, 3000)
    with pytest.raises(vendor.ImageFormatError, match="page_size"):
        parse(bytes(raw))


def test_rejects_absurd_header_size() -> None:
    raw = bytearray(pack(_sample(3)))
    struct.pack_into("<I", raw, 0x830, 16)
    with pytest.raises(vendor.ImageFormatError, match="header_size"):
        parse(bytes(raw))


def test_rejects_oversized_cmdline() -> None:
    with pytest.raises(vendor.ImageValueError, match="vendor_cmdline"):
        pack(_sample(3, vendor_cmdline=b"x" * (vendor.CMDLINE_SIZE + 1)))


def test_rejects_oversized_name() -> None:
    with pytest.raises(vendor.ImageValueError, match="board name"):
        pack(_sample(3, name=b"n" * (vendor.NAME_SIZE + 1)))


def test_fully_filled_fixed_fields_round_trip() -> None:
    """AOSP pads with pack(f'{N}s', value), which permits a completely full field."""
    image = _sample(
        3,
        vendor_cmdline=b"c" * vendor.CMDLINE_SIZE,
        name=b"n" * vendor.NAME_SIZE,
    )
    parsed = parse(pack(image))
    assert parsed.vendor_cmdline == image.vendor_cmdline
    assert parsed.name == image.name
    assert pack(parsed) == pack(image)


def test_rejects_oversized_fragment_name() -> None:
    image = _sample(4, fragments=(VendorRamdiskFragment(b"abcd", 1, b"n" * 33),))
    with pytest.raises(vendor.ImageValueError, match="fragment name"):
        pack(image)


def test_rejects_oversized_address() -> None:
    with pytest.raises(vendor.ImageValueError, match="kernel_addr"):
        pack(_sample(3, kernel_addr=0x1_0000_0000))


def test_rejects_bad_fragment_table_entry_size() -> None:
    raw = bytearray(
        pack(_sample(4, fragments=(VendorRamdiskFragment(b"abcd", 1, b"x"),)))
    )
    struct.pack_into("<I", raw, 0x848, 8)  # entry_size far below the 108-byte layout
    with pytest.raises(vendor.ImageFormatError, match="entry_size"):
        parse(bytes(raw))


def test_rejects_fragment_table_size_mismatch() -> None:
    raw = bytearray(
        pack(_sample(4, fragments=(VendorRamdiskFragment(b"abcd", 1, b"x"),)))
    )
    struct.pack_into("<I", raw, 0x840, 216)  # table_size not entry_num * entry_size
    with pytest.raises(vendor.ImageFormatError, match="table_size"):
        parse(bytes(raw))


def _table_offset(raw: bytes) -> int:
    """Byte offset of the v4 fragment table inside a packed image."""
    u32 = lambda off: struct.unpack_from("<I", raw, off)[0]
    page, header_size = u32(0x0C), u32(0x830)
    align = lambda n: -(-n // page) * page
    return align(align(header_size) + u32(0x18)) + _dtb_padding(raw)


def _dtb_padding(raw: bytes) -> int:
    u32 = lambda off: struct.unpack_from("<I", raw, off)[0]
    page = u32(0x0C)
    dtb_len = u32(0x834)
    return (-(-dtb_len // page) * page) if dtb_len else 0


def test_rejects_fragment_outside_the_ramdisk() -> None:
    raw = bytearray(
        pack(_sample(4, fragments=(VendorRamdiskFragment(b"abcd", 1, b"x"),)))
    )
    assert struct.unpack_from("<I", raw, 0x840)[0] == 108
    # point the entry's offset past the end of the ramdisk
    struct.pack_into("<I", raw, _table_offset(bytes(raw)) + 4, 4096)
    with pytest.raises(vendor.ImageFormatError, match="fragment"):
        parse(bytes(raw))


def test_parse_never_raises_struct_or_index_error() -> None:
    """Every prefix of a valid image must fail with a clear domain error."""
    raw = pack(_sample(4))
    for cut in range(len(raw)):
        try:
            parse(raw[:cut])
        except (vendor.ImageFormatError, vendor.ImageTruncatedError):
            continue
        except (struct.error, IndexError) as exc:  # pragma: no cover
            pytest.fail(f"prefix of {cut} bytes leaked {type(exc).__name__}: {exc}")


def test_detect() -> None:
    assert vendor.detect(pack(_sample(3)))
    assert not vendor.detect(b"ANDROID!" + bytes(100))
