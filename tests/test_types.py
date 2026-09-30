"""Lock in the empirically-verified header layout.

Every offset asserted here was produced by packing real images with the mkbootimg
bundled in ``bin/`` and parsing the resulting bytes, so this module is the guard
against someone "correcting" these constants to match AOSP's published layout.
The bundled build is not AOSP: it uses BOOT_NAME_SIZE=16 / BOOT_ARGS_SIZE=512.
"""

import struct

import pytest

from aik_py.errors import AikError, CorruptImageError
from aik_py.types import (
    BOOT_ARGS_SIZE,
    BOOT_HEADER_V1_SIZE,
    BOOT_HEADER_V2_SIZE,
    BOOT_ID_SIZE,
    BOOT_NAME_SIZE,
    BOOT_V1_DTB_ADDR,
    BOOT_V1_DTB_SIZE,
    BOOT_V1_HEADER_SIZE,
    BOOT_V1_RECOVERY_DTBO_ADDR,
    BOOT_V1_RECOVERY_DTBO_SIZE,
    BOOT_V3_HEADER_SIZE,
    BOOT_V3_HEADER_VERSION,
    BOOT_V3_KERNEL_SIZE,
    BOOT_V3_RAMDISK_SIZE,
    HEADER_SIZE_BY_VERSION,
    MAGIC_BOOT,
    MAGIC_VENDOR,
    VENDOR_CMDLINE_MAX,
    VENDOR_DTB_ADDR,
    VENDOR_DTB_SIZE,
    VENDOR_HEADER_SIZE,
    VENDOR_HEADER_SIZE_VALUE,
    VENDOR_NAME_MAX,
    VENDOR_VENDOR_RAMDISK_SIZE,
    BootHeader,
    Compression,
    HeaderVersion,
    ImageType,
    VendorBootHeader,
)


def test_scheme_constants_differ_from_aosp():
    assert (BOOT_NAME_SIZE, BOOT_ARGS_SIZE, BOOT_ID_SIZE) == (16, 512, 32)


def test_magic_constants():
    assert MAGIC_BOOT == b"ANDROID!"
    assert MAGIC_VENDOR == b"VNDRBOOT"


def test_header_sizes_reported_per_version():
    assert HEADER_SIZE_BY_VERSION[HeaderVersion.V1] == 1648
    assert HEADER_SIZE_BY_VERSION[HeaderVersion.V2] == 1660
    assert HEADER_SIZE_BY_VERSION[HeaderVersion.V3] == 1580
    assert HEADER_SIZE_BY_VERSION[HeaderVersion.V4] == 1580


def test_v0_has_no_header_size():
    assert HeaderVersion.V0 not in HEADER_SIZE_BY_VERSION


def test_v1_tail_offsets():
    assert (BOOT_V1_RECOVERY_DTBO_SIZE, BOOT_V1_RECOVERY_DTBO_ADDR) == (0x660, 0x664)
    assert (BOOT_V1_HEADER_SIZE, BOOT_V1_DTB_SIZE, BOOT_V1_DTB_ADDR) == (
        0x66C,
        0x670,
        0x674,
    )


def test_v3_offsets():
    assert (BOOT_V3_KERNEL_SIZE, BOOT_V3_RAMDISK_SIZE) == (0x08, 0x0C)
    assert (BOOT_V3_HEADER_SIZE, BOOT_V3_HEADER_VERSION) == (0x14, 0x28)


def test_vendor_offsets():
    assert VENDOR_HEADER_SIZE_VALUE == 2112
    assert (VENDOR_VENDOR_RAMDISK_SIZE, VENDOR_HEADER_SIZE) == (0x18, 0x830)
    assert (VENDOR_DTB_SIZE, VENDOR_DTB_ADDR) == (0x834, 0x838)
    assert VENDOR_CMDLINE_MAX == 2048
    assert VENDOR_NAME_MAX == 16


def test_classic_header_field_offsets_match_packed_image():
    """Reproduce the byte layout of an image packed by the bundled mkbootimg."""
    page_size = 2048
    kernel_size, ramdisk_size = 1024, 512
    image = bytearray(page_size)
    image[0x00:0x08] = MAGIC_BOOT

    def put(off: int, value: int) -> None:
        struct.pack_into("<I", image, off, value)

    put(0x08, kernel_size)
    put(0x10, ramdisk_size)
    put(0x24, page_size)
    put(0x28, HeaderVersion.V1)

    assert struct.unpack_from("<I", image, 0x08)[0] == kernel_size
    assert struct.unpack_from("<I", image, 0x10)[0] == ramdisk_size
    assert struct.unpack_from("<I", image, 0x24)[0] == page_size
    assert struct.unpack_from("<I", image, 0x28)[0] == HeaderVersion.V1
    assert image[0x30 : 0x30 + BOOT_NAME_SIZE] == bytes(BOOT_NAME_SIZE)
    assert image[0x40 : 0x40 + BOOT_ARGS_SIZE] == bytes(BOOT_ARGS_SIZE)


def test_name_cmdline_id_are_contiguous_and_end_before_v1_tail():
    assert 0x40 + BOOT_ARGS_SIZE == 0x240
    assert 0x240 + BOOT_ID_SIZE == 0x260
    # The v1/v2 tail fields live well past id, in the second page.
    assert BOOT_V1_RECOVERY_DTBO_SIZE > 0x260


def test_image_type_tokens_match_bash_imgtype_strings():
    assert ImageType.AOSP.token == "AOSP"
    assert ImageType.AOSP_VNDR.token == "AOSP_VNDR"
    assert ImageType.AOSP_PXA.token == "AOSP-PXA"
    assert ImageType.U_BOOT.token == "U-Boot"


def test_compression_extensions():
    expected = {
        Compression.GZIP: ".gz",
        Compression.BZIP2: ".bz2",
        Compression.XZ: ".xz",
        Compression.LZMA: ".lzma",
        Compression.LZ4: ".lz4",
        Compression.LZ4_LEGACY: ".lz4",
        Compression.CPIO: "",
        Compression.LZOP: ".lzo",
        Compression.EMPTY: ".empty",
    }
    assert {c: c.extension for c in Compression} == expected


def test_compression_magic_names():
    assert Compression.GZIP.magic_name == "gzip"
    assert Compression.BZIP2.magic_name == "bzip2"
    assert Compression.LZ4.magic_name == "lz4"
    assert Compression.LZ4_LEGACY.magic_name == "lz4-l"
    assert Compression.LZOP.magic_name == "lzop"


def test_boot_header_coerces_and_derives_header_size():
    h = BootHeader(kernel_size=1024, page_size=4096, header_version=1)
    assert h.header_version is HeaderVersion.V1
    assert h.header_size == BOOT_HEADER_V1_SIZE

    v2 = BootHeader(header_version=2)
    assert v2.header_size == BOOT_HEADER_V2_SIZE

    # v0 has no header_size field, so it stays absent rather than guessing.
    assert BootHeader().header_size == 0


def test_boot_header_respects_explicit_header_size():
    assert BootHeader(header_version=1, header_size=9999).header_size == 9999


def test_unknown_header_version_raises_aik_error_not_value_error():
    with pytest.raises(AikError) as excinfo:
        BootHeader(header_version=5)
    assert isinstance(excinfo.value, CorruptImageError)
    assert "5" in str(excinfo.value)


def test_vendor_header_coerces_version():
    assert VendorBootHeader(header_version=3).header_version is HeaderVersion.V3
    with pytest.raises(CorruptImageError):
        VendorBootHeader(header_version=9)


def test_vendor_header_defaults():
    h = VendorBootHeader()
    assert h.header_size == VENDOR_HEADER_SIZE_VALUE
    assert h.name_size == VENDOR_NAME_MAX
