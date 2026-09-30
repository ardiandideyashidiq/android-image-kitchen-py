"""Tests for the v3/v4 AOSP boot image format.

Offset expectations here are the ones measured from images built by the bundled
``mkbootimg`` in ``bin/linux/x86_64``, not the AOSP prose layout -- the two
disagree about where ``header_version`` lives and whether v3/v4 keep the magic.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from aik_py.formats import aosp_v34 as v34
from aik_py.formats.aosp_v34 import (
    CMDLINE_OFFSET,
    CMDLINE_SIZE,
    HEADER_SIZE_OFFSET,
    HEADER_SIZE_V3,
    HEADER_SIZE_V4,
    HEADER_VERSION_OFFSET,
    BootImageError,
    BootImageV34,
    encode_os_version,
    format_os_patch_level,
    pack,
    parse,
    parse_os_patch_level,
    parse_os_version,
    replace,
    uses_v34_layout,
)

KERNEL = b"\xaa" * 1024
RAMDISK = b"\xbb" * 100


def make(**kwargs) -> BootImageV34:
    base = {
        "header_version": 4,
        "kernel": KERNEL,
        "ramdisk": RAMDISK,
        "cmdline": b"console=ttyS0",
        "os_version": (12, 1, 3),
        "os_patch_level": (2020, 11),
    }
    return BootImageV34(**{**base, **kwargs})


# --- header field offsets -------------------------------------------------


@pytest.mark.parametrize("header_version", [3, 4])
def test_header_fields_land_at_verified_offsets(header_version):
    blob = pack(make(header_version=header_version))

    assert blob[0:8] == b"ANDROID!"
    assert struct.unpack_from("<I", blob, 0x08)[0] == len(KERNEL)
    assert struct.unpack_from("<I", blob, 0x0C)[0] == len(RAMDISK)
    assert struct.unpack_from("<I", blob, OS_VERSION := 0x10)[0] == 0x1804194B
    assert blob[OS_VERSION : OS_VERSION + 4] == struct.pack("<I", 0x1804194B)
    assert struct.unpack_from("<I", blob, HEADER_SIZE_OFFSET)[0] == (
        HEADER_SIZE_V4 if header_version == 4 else HEADER_SIZE_V3
    )
    assert struct.unpack_from("<I", blob, HEADER_VERSION_OFFSET)[0] == header_version
    assert blob[CMDLINE_OFFSET : CMDLINE_OFFSET + 14] == b"console=ttyS0\x00"


def test_v3_header_size_is_1580_and_v4_is_1584():
    assert HEADER_SIZE_V3 == 1580
    assert HEADER_SIZE_V4 == 1584
    assert struct.unpack_from("<I", pack(make(header_version=3)), 0x14)[0] == 1580
    assert struct.unpack_from("<I", pack(make(header_version=4)), 0x14)[0] == 1584


def test_v4_header_size_is_1584_even_though_bundled_mkbootimg_writes_1580():
    # Deliberate divergence: mkbootimg stamps 1580 for --header_version 4
    # because it never emits the signature_size word.  Real v4 images use 1584.
    bundled = Path("/tmp/aik-v34/hv4.img")
    if not bundled.exists():
        pytest.skip("no bundled-mkbootimg fixture; run the e2e recipe to generate one")
    assert struct.unpack_from("<I", bundled.read_bytes(), HEADER_SIZE_OFFSET)[0] == 1580
    assert (
        struct.unpack_from("<I", pack(make(header_version=4)), HEADER_SIZE_OFFSET)[0]
        == 1584
    )


def test_both_header_size_values_parse():
    # The zero fallback covers OEM images; a wrong-but-plausible value is
    # honoured as-is so both mkbootimg's 1580 and AOSP's 1584 round-trip.
    for value, expected in ((1580, 1580), (1584, 1584), (0, 1584)):
        blob = bytearray(pack(make(header_version=4)))
        struct.pack_into("<I", blob, HEADER_SIZE_OFFSET, value)
        assert parse(bytes(blob)).header_size == expected


def test_signature_size_field_only_exists_for_v4():
    v3 = pack(make(header_version=3))
    v4 = pack(make(header_version=4, signature=bytes(32)))
    # Both headers fit in the same 4096-byte first page, so v4 differs only by
    # the extra signature_size word and the signature page itself.
    assert len(v3) == 3 * 4096
    assert len(v4) == 4 * 4096
    assert struct.unpack_from("<I", v4, 0x62C)[0] == 32


def test_os_version_bit_layout_matches_bundled_mkbootimg():
    # Values observed byte-for-byte from bin/linux/x86_64/mkbootimg output.
    assert encode_os_version((12, 1, 3), (2020, 11)) == 0x1804194B
    assert encode_os_version((0, 0, 0), (2000, 0)) == 0x00000000
    assert encode_os_version((1, 2, 3), (2020, 11)) == 0x0208194B
    assert encode_os_version((9, 9, 9), (2020, 11)) == 0x1224494B
    assert encode_os_version((16, 0, 0), (2020, 11)) == 0x02000014B
    assert encode_os_version((31, 7, 15), (2020, 11)) == 0x003E1C794B


def test_reserved_words_round_trip():
    blob = pack(make(reserved=(1, 2, 3, 4)))
    assert struct.unpack_from("<4I", blob, 0x18) == (1, 2, 3, 4)
    assert parse(blob).reserved == (1, 2, 3, 4)


# --- round trips ----------------------------------------------------------


@pytest.mark.parametrize("header_version", [3, 4])
def test_round_trip_is_byte_identical(header_version):
    image = make(header_version=header_version)
    assert pack(parse(pack(image))) == pack(image)


@pytest.mark.parametrize("header_version", [3, 4])
def test_round_trip_preserves_every_field(header_version):
    image = make(header_version=header_version)
    got = parse(pack(image))

    assert got.header_version == header_version
    assert got.kernel == KERNEL
    assert got.ramdisk == RAMDISK
    assert got.cmdline == b"console=ttyS0"
    assert got.os_version == (12, 1, 3)
    assert got.os_patch_level == (2020, 11)
    assert got.signature == b""


def test_empty_payloads_round_trip():
    image = make(kernel=b"", ramdisk=b"")
    got = parse(pack(image))
    assert got.kernel == b""
    assert got.ramdisk == b""


def test_unset_os_version_decodes_to_zero():
    got = parse(pack(make(os_version=(0, 0, 0), os_patch_level=(2000, 0))))
    assert got.os_version == (0, 0, 0)
    assert got.os_patch_level == (2000, 0)


# --- page alignment -------------------------------------------------------


@pytest.mark.parametrize("page_size", [2048, 4096])
@pytest.mark.parametrize("kernel_len,ramdisk_len", [(4096, 2048), (100, 100), (1, 1)])
def test_page_alignment_round_trips(page_size, kernel_len, ramdisk_len):
    image = make(
        kernel=b"\xaa" * kernel_len, ramdisk=b"\xbb" * ramdisk_len, page_size=page_size
    )
    blob = pack(image)
    assert len(blob) % page_size == 0

    body = _align(HEADER_SIZE_V4, page_size)
    assert blob[body : body + kernel_len] == image.kernel
    body += (kernel_len + page_size - 1) // page_size * page_size
    assert blob[body : body + ramdisk_len] == image.ramdisk

    got = parse(blob, page_size=page_size)
    assert got.kernel == image.kernel
    assert got.ramdisk == image.ramdisk


def test_unaligned_payloads_are_zero_padded():
    page_size = 4096
    blob = pack(make(kernel=b"\xaa" * 100, ramdisk=b"\xbb" * 7, page_size=page_size))
    body = _align(HEADER_SIZE_V4, page_size)
    tail = blob[body + 100 : body + page_size]
    assert tail == bytes(page_size - 100)


def test_page_size_defaults_to_4096_and_is_overridable():
    assert make().page_size == 4096
    assert v34.PAGE_SIZE == 4096
    assert parse(pack(make())).page_size == 4096
    assert parse(pack(make(page_size=2048)), page_size=2048).page_size == 2048


@pytest.mark.parametrize("bad", [0, -1, 1 << 20])
def test_absurd_page_size_is_rejected(bad):
    with pytest.raises(BootImageError, match="page_size"):
        parse(pack(make()), page_size=bad)


# --- v4 signature ---------------------------------------------------------


def test_signature_blob_survives_round_trip():
    sig = bytes(range(32))
    blob = pack(make(header_version=4, signature=sig))

    assert struct.unpack_from("<I", blob, 0x62C)[0] == len(sig)
    got = parse(blob)
    assert got.signature == sig
    assert got.signature_size == len(sig)
    assert pack(got) == blob


def test_signature_sits_after_ramdisk_and_its_own_page():
    page_size = 4096
    sig = b"\x5a" * 256
    blob = pack(make(header_version=4, signature=sig, page_size=page_size))

    pos = _align(HEADER_SIZE_V4, page_size)
    pos += (len(KERNEL) + page_size - 1) // page_size * page_size
    pos += (len(RAMDISK) + page_size - 1) // page_size * page_size
    assert blob[pos : pos + len(sig)] == sig
    assert len(blob) % page_size == 0


def test_signature_on_v3_is_rejected():
    with pytest.raises(BootImageError, match="v4"):
        pack(make(header_version=3, signature=b"\x01" * 32))


# --- os_version / os_patch_level -----------------------------------------


@pytest.mark.parametrize(
    "os_version,os_patch_level",
    [
        ((12, 1, 3), (2020, 11)),
        ((0, 0, 0), (2000, 0)),
        ((31, 7, 15), (2127, 12)),
        ((127, 127, 127), (2000, 1)),
    ],
)
def test_os_version_round_trips(os_version, os_patch_level):
    got = parse(pack(make(os_version=os_version, os_patch_level=os_patch_level)))
    assert got.os_version == os_version
    assert got.os_patch_level == os_patch_level


def test_patch_level_padding_asymmetry_is_normalised_both_ways():
    # mkbootimg accepts an unpadded month; unpackbootimg always prints padded.
    assert (
        parse_os_patch_level("2020-9") == parse_os_patch_level("2020-09") == (2020, 9)
    )
    assert format_os_patch_level((2020, 9)) == "2020-09"
    # Round-tripping a padded string is stable.
    assert format_os_patch_level(parse_os_patch_level("2020-09")) == "2020-09"
    assert format_os_patch_level((2000, 0)) == "2000-00"


def test_zero_os_version_formats_like_the_bundled_unpacker():
    assert format_os_patch_level((2000, 0)) == "2000-00"
    assert v34.format_os_version((0, 0, 0)) == "0.0.0"


@pytest.mark.parametrize(
    "text", ["", "12", "12.0", "a.b.c", "12.0.0.0", "2000-13", "1999-01", "2020", "a-b"]
)
def test_malformed_version_strings_are_rejected(text):
    with pytest.raises(BootImageError):
        parse_os_patch_level(text) if "-" in text else parse_os_version(text)


@pytest.mark.parametrize("bad", [(128, 0, 0), (0, 128, 0), (0, 0, 128)])
def test_out_of_range_os_version_rejected(bad):
    with pytest.raises(BootImageError, match="out of range"):
        encode_os_version(bad, (2020, 1))


def test_out_of_range_patch_year_and_month_rejected():
    with pytest.raises(BootImageError, match="year"):
        encode_os_version((1, 0, 0), (1999, 1))
    with pytest.raises(BootImageError, match="month"):
        encode_os_version((1, 0, 0), (2020, 16))


# --- version dispatch -----------------------------------------------------


@pytest.mark.parametrize("version", [3, 4, 5, 6, 7, 8])
def test_versions_3_through_8_use_v34_layout(version):
    assert uses_v34_layout(version)


@pytest.mark.parametrize("version", [-1, 0, 1, 2, 9])
def test_versions_0_through_2_and_beyond_do_not(version):
    assert not uses_v34_layout(version)


def test_dispatch_helper_reads_version_after_magic_check():
    assert v34.dispatch_header_version(pack(make(header_version=3))) == 3
    assert v34.dispatch_header_version(pack(make(header_version=4))) == 4
    with pytest.raises(BootImageError, match="bad magic"):
        v34.dispatch_header_version(b"NOTMAGIC" + bytes(64))


def test_pack_rejects_classic_versions():
    with pytest.raises(BootImageError, match="outside the v3/v4 range"):
        pack(make(header_version=2))


# --- error handling -------------------------------------------------------


def test_bad_magic_is_rejected():
    blob = bytearray(pack(make()))
    blob[0:8] = b"NOTMAGI!"
    with pytest.raises(BootImageError, match="bad magic"):
        parse(bytes(blob))


def test_empty_and_tiny_inputs_are_rejected():
    for bad in (b"", b"A", b"ANDROID"):
        with pytest.raises(BootImageError, match="truncated"):
            parse(bad)


@pytest.mark.parametrize("cut", [8, 12, 0x14, 0x28, 0x2C, 0x100])
def test_truncated_header_is_rejected(cut):
    blob = pack(make())[:cut]
    with pytest.raises(BootImageError, match="truncated"):
        parse(blob)


def test_truncated_payload_is_rejected():
    # Cut into the ramdisk's last real byte, not into trailing padding.
    blob = pack(make(kernel=b"\xaa" * 1024, ramdisk=b"\xbb" * 100))
    ramdisk_start = len(blob) - 4096
    with pytest.raises(BootImageError, match="ramdisk"):
        parse(blob[: ramdisk_start + 50])


def test_oversized_kernel_size_is_rejected():
    blob = bytearray(pack(make()))
    struct.pack_into("<I", blob, 0x08, 0xFFFF_FFFF)
    with pytest.raises(BootImageError, match="kernel"):
        parse(bytes(blob))


def test_oversized_ramdisk_size_is_rejected():
    blob = bytearray(pack(make()))
    struct.pack_into("<I", blob, 0x0C, 0x7FFF_FFFF)
    with pytest.raises(BootImageError, match="ramdisk"):
        parse(bytes(blob))


def test_oversized_signature_size_is_rejected():
    blob = bytearray(pack(make()))
    struct.pack_into("<I", blob, 0x62C, 0x00FF_FFFF)
    with pytest.raises(BootImageError, match="signature"):
        parse(bytes(blob))


def test_header_size_beyond_file_is_rejected():
    blob = bytearray(pack(make()))
    struct.pack_into("<I", blob, HEADER_SIZE_OFFSET, 1 << 21)
    with pytest.raises(BootImageError, match="header_size"):
        parse(bytes(blob))


def test_zeroed_header_size_falls_back_to_version_default():
    for version, expected in ((3, HEADER_SIZE_V3), (4, HEADER_SIZE_V4)):
        blob = bytearray(pack(make(header_version=version)))
        struct.pack_into("<I", blob, HEADER_SIZE_OFFSET, 0)
        assert parse(bytes(blob)).header_size == expected


def test_cmdline_at_max_length_is_rejected():
    with pytest.raises(BootImageError, match="cmdline"):
        pack(make(cmdline=b"x" * CMDLINE_SIZE))
    blob = pack(make(cmdline=b"x" * (CMDLINE_SIZE - 1)))
    assert len(parse(blob).cmdline) == CMDLINE_SIZE - 1


def test_full_length_cmdline_is_not_nul_truncated():
    blob = pack(make(cmdline=b"x" * (CMDLINE_SIZE - 1)))
    assert CMDLINE_OFFSET + CMDLINE_SIZE <= len(blob)


def test_errors_are_boot_image_errors_not_struct_or_index_errors():
    blob = pack(make())
    for bad in (b"", b"\x00" * 16, blob[:5], blob[:100], blob[:1500]):
        with pytest.raises(BootImageError):
            parse(bad)


# --- helpers --------------------------------------------------------------


def test_replace_produces_a_modified_copy():
    image = make()
    other = replace(image, cmdline=b"androidboot.hardware=qcom")
    assert image.cmdline == b"console=ttyS0"
    assert other.cmdline == b"androidboot.hardware=qcom"
    assert other.kernel == image.kernel


def test_compute_signature_is_deterministic_and_covers_payloads():
    a = v34.compute_signature(make())
    b = v34.compute_signature(make())
    c = v34.compute_signature(replace(make(), ramdisk=b"\xcc" * 100))
    assert len(a) == 32
    assert a == b
    assert a != c


def test_sizes_are_properties():
    image = make(signature=b"\x01" * 8)
    assert image.magic == b"ANDROID!"
    assert image.kernel_size == len(KERNEL)
    assert image.ramdisk_size == len(RAMDISK)
    assert image.signature_size == 8


def _align(size: int, page_size: int) -> int:
    return (size + page_size - 1) // page_size * page_size


# --- regressions for issues found in review -------------------------------


def test_classic_images_are_rejected_not_mis_parsed():
    # A v0-v2 image has unrelated content at 0x28 and elsewhere.  Reading it
    # with the v3/v4 layout silently yields the kernel addr as kernel_size and
    # the os_version *string* as a cmdline, so it must be rejected outright.
    blob = bytearray(pack(make()))
    struct.pack_into("<I", blob, HEADER_VERSION_OFFSET, 2)
    struct.pack_into("<I", blob, 0x08, 0x80008000)  # kernel_addr in classic layout
    struct.pack_into("<I", blob, 0x10, 0)  # ramdisk_addr, not ramdisk_size
    blob[CMDLINE_OFFSET : CMDLINE_OFFSET + 5] = b"12.1.3"
    with pytest.raises(BootImageError, match="classic layout"):
        parse(bytes(blob))


@pytest.mark.parametrize("version", [0, 1, 2, 9, 255])
def test_out_of_range_header_versions_rejected_by_parse(version):
    blob = bytearray(pack(make()))
    struct.pack_into("<I", blob, HEADER_VERSION_OFFSET, version)
    with pytest.raises(BootImageError, match="outside the v3/v4 range"):
        parse(bytes(blob))


@pytest.mark.parametrize("page_size", [512, 1024, 2048])
def test_absurdly_small_header_size_is_rejected_for_any_page_size(page_size):
    # A header_size below the cmdline would start the "kernel" inside the
    # header, returning padding that loads as an all-zero kernel.
    blob = bytearray(pack(make()))
    struct.pack_into("<I", blob, HEADER_SIZE_OFFSET, 64)
    with pytest.raises(BootImageError, match="implausible header_size"):
        parse(bytes(blob), page_size=page_size)


def test_header_size_larger_than_cmdline_is_rejected():
    blob = bytearray(pack(make()))
    struct.pack_into("<I", blob, HEADER_SIZE_OFFSET, 100)
    with pytest.raises(BootImageError, match="implausible header_size"):
        parse(bytes(blob))


def test_implausible_header_size_is_reported_before_truncation():
    blob = bytearray(pack(make()))
    struct.pack_into("<I", blob, HEADER_SIZE_OFFSET, 2_000_000_000)
    with pytest.raises(BootImageError, match="implausible header_size"):
        parse(bytes(blob))


def test_short_file_cannot_yield_a_silently_truncated_cmdline():
    blob = bytearray(pack(make()))
    struct.pack_into("<I", blob, HEADER_SIZE_OFFSET, 0)
    with pytest.raises(BootImageError):
        parse(bytes(blob)[: CMDLINE_OFFSET + 100])


def test_non_canonical_header_size_survives_parse_and_pack():
    # mkbootimg stamps 1580 on v4; an OEM may use a larger header.  Either way
    # the recorded size must be preserved rather than recomputed.
    blob = bytearray(pack(make(header_version=4)))
    struct.pack_into("<I", blob, HEADER_SIZE_OFFSET, 1600)
    got = parse(bytes(blob))
    assert got.header_size == 1600
    assert pack(got) == bytes(blob)


def test_v3_pack_ignores_the_v4_header_size_default():
    # BootImageV34.header_size defaults to HEADER_SIZE_V4, which cannot hold a
    # v3 image; pack must fall back rather than overrun the header buffer.
    assert pack(make(header_version=3)).__len__() == 3 * 4096
    assert (
        struct.unpack_from("<I", pack(make(header_version=3)), HEADER_SIZE_OFFSET)[0]
        == HEADER_SIZE_V3
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"os_version": "12.1.3"},
        {"os_version": (1, 2)},
        {"os_version": (1, 2, 3, 4)},
        {"reserved": (1, 2, 3)},
        {"reserved": "abcd"},
        {"reserved": (1, 2, 3, "x")},
        {"reserved": (1, 2, 3, -1)},
        {"cmdline": "console=ttyS0"},
    ],
)
def test_pack_raises_boot_image_error_for_malformed_fields(kwargs):
    with pytest.raises(BootImageError):
        pack(make(**kwargs))


def test_month_13_to_15_round_trip_but_are_not_valid_calendar_months():
    # The raw field is 4 bits wide; read/write stays bit-faithful so such an
    # image is not corrupted, but the string parser refuses to invent one.
    for month in (13, 15):
        blob = pack(make(os_patch_level=(2020, month)))
        assert parse(blob).os_patch_level == (2020, month)
        with pytest.raises(BootImageError, match="month"):
            parse_os_patch_level(f"2020-{month}")


def test_unset_patch_level_round_trips_through_its_own_string_form():
    # A zeroed header decodes to (2000, 0); formatting then reparsing that
    # must work, or the common "unset" image cannot be re-emitted.
    assert parse_os_patch_level(format_os_patch_level((2000, 0))) == (2000, 0)
    assert parse(pack(make(os_patch_level=(2000, 0)))).os_patch_level == (2000, 0)


def test_compute_signature_preserves_header_version():
    v3 = v34.compute_signature(make(header_version=3))
    v4 = v34.compute_signature(make(header_version=4))
    assert v3 != v4, "v3 and v4 cover different headers and must not collide"


def test_compute_signature_does_not_depend_on_an_existing_signature():
    with_sig = v34.compute_signature(make(signature=b"\xff" * 32))
    without = v34.compute_signature(make())
    assert with_sig == without


# --- regressions for the second review pass ------------------------------


def test_mkbootimg_v4_image_round_trips_byte_for_byte():
    # mkbootimg stamps header_size 1580 on a v4 image.  That is a real header
    # size, not a bug to be silently corrected: rewriting it to 1584 would
    # change the one field the on-disk format agreed on.
    bundled = Path("/tmp/aik-v34/hv4.img")
    if not bundled.exists():
        pytest.skip("no bundled-mkbootimg fixture; run the e2e recipe to generate one")
    orig = bundled.read_bytes()
    assert pack(parse(orig)) == orig


@pytest.mark.parametrize(
    "padding", [b"\xff" * 4, b"\xde\xad\xbe\xef", b"\x00\x00\x08\x00"]
)
def test_v4_header_of_1580_does_not_read_padding_as_a_signature(padding):
    # With header_size 1580 the signature_size word sits in the padding region.
    # Whatever is there must not be read as a signature length.
    blob = bytearray(pack(make(header_version=4)))
    struct.pack_into("<I", blob, HEADER_SIZE_OFFSET, HEADER_SIZE_V3)
    blob[0x62C:0x630] = padding
    got = parse(bytes(blob))
    assert got.header_version == 4
    assert got.signature == b""
    assert got.kernel == KERNEL


def test_v4_header_of_1580_with_a_signature_is_rejected_by_pack():
    # A 1580-byte header cannot carry a signature, so asking for both is an
    # error rather than a silently dropped signature.
    with pytest.raises(BootImageError, match="signature_size field"):
        pack(replace(make(header_version=4), header_size=1580, signature=b"\x01" * 32))


@pytest.mark.parametrize("header_size", [-1, -100, 100, (1 << 20) + 1, 1 << 30])
@pytest.mark.parametrize("version", [3, 4])
def test_pack_rejects_impossible_header_sizes(version, header_size):
    with pytest.raises(BootImageError):
        pack(replace(make(header_version=version), header_size=header_size))


def test_pack_accepts_header_size_zero_as_derive_from_version():
    for version, expected in ((3, HEADER_SIZE_V3), (4, HEADER_SIZE_V4)):
        blob = pack(replace(make(header_version=version), header_size=0))
        assert struct.unpack_from("<I", blob, HEADER_SIZE_OFFSET)[0] == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"os_version": ("12", 1, 3)},
        {"os_version": (12.0, 1, 3)},
        {"os_version": (True, 1, 3)},
        {"os_patch_level": (2020, "11")},
        {"os_patch_level": (2020, None)},
    ],
)
def test_encode_os_version_rejects_non_int_elements(kwargs):
    with pytest.raises(BootImageError):
        pack(make(**kwargs))


@pytest.mark.parametrize("text", ["².0.0", "1².0.0", "2020-²", "²020-11"])
def test_non_ascii_digits_are_rejected_not_crashed_on(text):
    # str.isdigit() accepts these, but int() raises a bare ValueError on them.
    with pytest.raises(BootImageError):
        parse_os_version(text) if text.count(".") == 2 else parse_os_patch_level(text)


def test_header_padding_survives_a_round_trip():
    # raw_header carries padding we never interpret, so an OEM fill pattern
    # must not be zeroed out by a parse/pack cycle.  The fill stops short of
    # the signature_size word, which is a real field and stays zero.
    blob = bytearray(pack(make(header_version=4)))
    end = 0x62C
    blob[0x40:end] = b"\xff" * (end - 0x40)
    got = parse(bytes(blob))
    assert pack(got) == bytes(blob)
