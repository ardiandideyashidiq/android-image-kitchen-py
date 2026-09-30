"""Format detection tests.

Synthetic buffers cover every rule in ``bin/androidbootimg.magic``. The PXA and
LOKI offsets were pinned against file-5.48 and real images from the bundled
``mkbootimg``/``pxa-mkbootimg``; see the comments in ``aik_py.detect``.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import pytest

from aik_py.detect import (
    LOKI_OFFSET,
    LOKI_PARTITION_OFFSET,
    AvbPartition,
    Detection,
    ImageType,
    LokiPartition,
    MtkType,
    PatchType,
    SignatureType,
    TailType,
    avb1_partition,
    check_imgtype,
    describe,
    describe_mtk,
    detect_image_type,
    detect_mtk,
    detect_patch,
    detect_qcdt,
    detect_signature,
    detect_tail,
    has_mtk,
    header_version,
    loki_partition,
)

try:
    from aik_py.errors import CorruptImageError, UnsupportedFormatError
except ImportError:
    from aik_py.detect import CorruptImageError, UnsupportedFormatError

BIN = Path(__file__).resolve().parents[1] / "bin" / "linux" / "x86_64"
MTK = b"\x88\x16\x88\x58"
BUMP = b"\x41\xa9\xe4\x67\x74\x4d\x1d\x1b\xa4\x29\xf2\xec\xea\x65\x52\x79"


def pad(size: int, filler: bytes = b"\x00") -> bytes:
    return (filler * size)[:size]


def at(offset: int, body: bytes, size: int = 8192, filler: bytes = b"\x00") -> bytes:
    """Buffer of ``size`` bytes with ``body`` written at ``offset``."""
    buf = bytearray(pad(size, filler))
    buf[offset : offset + len(body)] = body
    return bytes(buf)


def aosp(header_version: int = 0, size: int = 8192) -> bytes:
    """Minimal ``ANDROID!`` header with a real header_version at 0x28."""
    buf = bytearray(pad(size))
    buf[0:8] = b"ANDROID!"
    struct.pack_into("<I", buf, 0x28, header_version)
    return bytes(buf)


def pxa(variant: bytes = b"\x00\x00\x00\x03", size: int = 8192) -> bytes:
    """AOSP header carrying a PXA marker in the u32 at 0x24."""
    buf = bytearray(aosp(0, size))
    buf[0x24:0x28] = variant
    return bytes(buf)


def write(buf: bytearray, offset: int, body: bytes) -> None:
    """Overwrite in place.

    Slice assignment would splice when ``len(body)`` exceeds the slice, silently
    shifting every following byte and invalidating the very offsets under test.
    """
    buf[offset : offset + len(body)] = body


def loki(partition: int = 0, microloader: bool = False, android_at: int = 0) -> bytes:
    buf = bytearray(aosp(0))
    write(buf, LOKI_OFFSET, b"LOKI")
    buf[LOKI_PARTITION_OFFSET] = partition
    if android_at:
        write(buf, android_at, b"ANDROID!")
    if microloader:
        write(buf, 96, b"microloader")
    return bytes(buf)


def mtk(kind: bytes = b"") -> bytes:
    """MTK header prepended to a split component.

    ``kind`` is a keyword to search for after the marker; empty means a bare
    header with no type keyword, which the Bash records as no type at all.
    """
    return at(0, MTK + pad(28) + kind)


# --------------------------------------------------------------------------
# signatures
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (at(0, b"-SIGNED-BY-SIGNBLOB-"), SignatureType.BLOB),
        (at(0, b"CHROMEOS"), SignatureType.CHROMEOS),
        (at(0, b"DHTB\x01\x00\x00"), SignatureType.DHTB),
        (at(0, b"\x01\x00\x00\x00"), SignatureType.SINV1),
        (at(0, b"\x02\x00\x00\x00"), SignatureType.SINV2),
        (at(0, b"\x03SIN"), SignatureType.SINV3),
        (at(64, b"Red Loader"), SignatureType.NOOK),
        (at(64, b"Green Loader"), SignatureType.NOOK),
        (at(64, b"Green Recovery"), SignatureType.NOOK),
        (at(64, b"eMMC boot.img+secondloader"), SignatureType.NOOK),
        (at(64, b"eMMC recovery.img+secondloader"), SignatureType.NOOK),
        (at(48, b"BauwksBoot"), SignatureType.NOOKTAB),
    ],
)
def test_detect_signature_at_its_offset(data, expected):
    assert detect_signature(data) is expected


def test_nook_signature_at_wrong_offset_does_not_match():
    # The magic file pins the NOOK strings at 64 and NOOKTAB at 48.
    assert detect_signature(at(63, b"Red Loader")) is None
    assert detect_signature(at(65, b"Red Loader")) is None
    assert detect_signature(at(47, b"BauwksBoot")) is None
    assert detect_signature(at(49, b"BauwksBoot")) is None


def test_signature_strings_are_exact():
    # DHTB needs its three trailing bytes; a bare "DHTB" is a different string.
    assert detect_signature(at(0, b"DHTB")) is None
    assert detect_signature(at(0, b"\x03SIX")) is None
    assert detect_signature(at(0, b"\x03sin")) is None


def test_no_signature_in_plain_image():
    assert detect_signature(aosp()) is None


def test_dhtb_detected_under_aosp_payload():
    data = at(0, b"DHTB\x01\x00\x00" + b"ANDROID!")
    assert detect_signature(data) is SignatureType.DHTB


# --------------------------------------------------------------------------
# image types
# --------------------------------------------------------------------------


def test_aosp():
    assert detect_image_type(aosp()) is ImageType.AOSP
    assert detect_image_type(aosp(4)) is ImageType.AOSP


def test_aosp_vndr():
    assert detect_image_type(at(0, b"VNDRBOOT")) is ImageType.AOSP_VNDR


def test_u_boot():
    assert detect_image_type(at(0, b"\x27\x05\x19\x56")) is ImageType.U_BOOT


def test_krnl():
    assert detect_image_type(at(0, b"KRNL")) is ImageType.KRNL


def test_elf():
    assert detect_image_type(at(0, b"\x7fELF")) is ImageType.ELF


@pytest.mark.parametrize("partition", [0, 1])
def test_osip_headered(partition):
    body = b"$OS$\x00\x00\x01" + pad(44) + bytes([partition]) + pad(3)
    assert detect_image_type(at(0, body)) is ImageType.OSIP


def test_osip_headerless():
    assert detect_image_type(at(1000, b"\xfc\xfa\xbc\x00")) is ImageType.OSIP


def test_osip_headerless_needs_offset_1000():
    assert detect_image_type(at(999, b"\xfc\xfa\xbc\x00")) is None
    assert detect_image_type(at(1001, b"\xfc\xfa\xbc\x00")) is None


@pytest.mark.parametrize(
    "variant", [b"\x00\x00\x00\x02", b"\x00\x00\x80\x02", b"\x00\x00\x00\x03"]
)
def test_all_three_pxa_variants(variant):
    assert detect_image_type(pxa(variant)) is ImageType.AOSP_PXA


@pytest.mark.parametrize("version", [0, 1, 4])
def test_non_pxa_header_versions_are_plain_aosp(version):
    assert detect_image_type(aosp(version)) is ImageType.AOSP


def test_pxa_marker_lives_at_0x24_not_at_header_version():
    """header_version 2 or 3 must not be mistaken for PXA.

    The magic file reads the marker from 0x24; header_version at 0x28 is a
    different field that stock images really do populate with 2 and 3.
    """
    assert detect_image_type(aosp(2)) is ImageType.AOSP
    assert detect_image_type(aosp(3)) is ImageType.AOSP
    assert header_version(aosp(2)) == 2
    assert header_version(aosp(3)) == 3


def test_header_version_is_not_reported_for_pxa():
    # A real pxa-mkbootimg image carries 00 01 00 80 at 0x28, which would read
    # back as a nonsense version, so PXA reports None instead.
    assert detect_image_type(pxa()) is ImageType.AOSP_PXA
    assert header_version(pxa()) is None
    buf = bytearray(pxa())
    buf[0x28:0x2C] = b"\x00\x01\x00\x80"  # exactly what pxa-mkbootimg writes
    assert header_version(bytes(buf)) is None


def test_unknown_data_is_unrecognized():
    assert detect_image_type(at(0, b"\xde\xad\xbe\xef")) is None
    assert detect_image_type(b"") is None


# --------------------------------------------------------------------------
# LOKI / AMONET
# --------------------------------------------------------------------------


def test_loki_boot():
    data = loki(partition=0)
    assert detect_patch(data) is PatchType.LOKI
    assert loki_partition(data) is LokiPartition.BOOT


def test_loki_recovery():
    data = loki(partition=1)
    assert detect_patch(data) is PatchType.LOKI
    assert loki_partition(data) is LokiPartition.RECOVERY


def test_loki_without_microloader_still_detected():
    # The magic file gates LOKI on ANDROID!, not on microloader.
    assert detect_patch(loki(0, microloader=False)) is PatchType.LOKI
    assert detect_patch(loki(0, microloader=True)) is PatchType.LOKI


def test_loki_with_android_nested_at_2048():
    assert detect_patch(loki(0, microloader=True, android_at=2048)) is PatchType.LOKI


def test_loki_without_android_marker_is_not_detected():
    data = at(1024, b"LOKI")
    assert detect_patch(data) is None


def test_loki_partition_byte_outside_known_values():
    # The Bash writes an empty $file-lokitype for anything but 00/01.
    assert loki_partition(loki(partition=2)) is None


def test_loki_partition_requires_a_loki_header():
    # Byte 1028 only means a partition under a LOKI header at 1024. Reading it
    # unconditionally would name a partition for every AOSP image.
    buf = bytearray(aosp(0))
    buf[LOKI_PARTITION_OFFSET] = 0
    assert loki_partition(bytes(buf)) is None
    assert loki_partition(aosp()) is None


def test_amonet():
    buf = bytearray(aosp(0))
    write(buf, 96, b"microloader")
    write(buf, 1024, b"ANDROID!")
    assert detect_patch(bytes(buf)) is PatchType.AMONET


def test_amonet_requires_microloader():
    buf = bytearray(aosp(0))
    write(buf, 1024, b"ANDROID!")
    assert detect_patch(bytes(buf)) is None


def test_amonet_requires_microloader_at_or_after_offset_96():
    buf = bytearray(aosp(0))
    write(buf, 80, b"microloader")
    write(buf, 1024, b"ANDROID!")
    assert detect_patch(bytes(buf)) is None


def test_amonet_requires_android_at_1024():
    buf = bytearray(aosp(0))
    write(buf, 96, b"microloader")
    assert detect_patch(bytes(buf)) is None


def test_amonet_detected_without_android_marker_at_0():
    # AMONET puts its own ANDROID! at 1024, so the nested rule fires on that.
    buf = bytearray(pad(8192))
    write(buf, 96, b"microloader")
    write(buf, 1024, b"ANDROID!")
    assert detect_patch(bytes(buf)) is PatchType.AMONET


def test_loki_takes_precedence_over_amonet():
    buf = bytearray(aosp(0))
    write(buf, 96, b"microloader")
    write(buf, LOKI_OFFSET, b"LOKI")
    buf[LOKI_PARTITION_OFFSET] = 0
    assert detect_patch(bytes(buf)) is PatchType.LOKI


def test_plain_aosp_has_no_patch():
    assert detect_patch(aosp()) is None


# --------------------------------------------------------------------------
# tail footers
# --------------------------------------------------------------------------


def test_avbv1_boot():
    data = at(0, b"\x02\x01\x01\x30\x82" + pad(20) + b"/boot" + pad(20), 4096)
    assert detect_tail(data) is TailType.AVBV1
    assert avb1_partition(data) is AvbPartition.BOOT


def test_avbv1_recovery():
    data = at(0, b"\x02\x01\x01\x30\x82" + pad(20) + b"/recovery" + pad(20), 4096)
    assert detect_tail(data) is TailType.AVBV1
    assert avb1_partition(data) is AvbPartition.RECOVERY


def test_avbv1_without_a_partition_name():
    data = at(0, b"\x02\x01\x01\x30\x82" + pad(64), 4096)
    assert detect_tail(data) is TailType.AVBV1
    assert avb1_partition(data) is None


def test_avbv1_boot_wins_when_footer_names_both():
    # The magic file lists /boot first and the Bash reads the first continuation.
    data = at(
        0,
        b"\x02\x01\x01\x30\x82" + pad(20) + b"/boot" + pad(20) + b"/recovery" + pad(20),
        4096,
    )
    assert avb1_partition(data) is AvbPartition.BOOT
    both = at(
        0,
        b"\x02\x01\x01\x30\x82" + pad(20) + b"/recovery" + pad(20) + b"/boot" + pad(20),
        4096,
    )
    assert avb1_partition(both) is AvbPartition.BOOT


def test_avbv2():
    assert detect_tail(at(0, pad(20) + b"AVBf" + pad(20), 4096)) is TailType.AVBV2


def test_bump():
    assert detect_tail(at(0, BUMP + pad(20), 4096)) is TailType.BUMP


def test_seandroid():
    assert detect_tail(at(0, b"SEANDROIDENFORCE" + pad(20), 4096)) is TailType.SEANDROID


def test_no_footer_returns_none():
    assert detect_tail(aosp()) is None
    assert detect_tail(b"") is None


def test_bump_footer_is_full_16_bytes():
    assert detect_tail(at(0, BUMP[:15] + b"\x00", 4096)) is None


def test_seandroid_prefix_is_not_enough():
    assert detect_tail(at(0, b"SEANDROIDENFORC", 4096)) is None


def test_tail_beyond_8192_byte_window_still_found():
    """The Bash widens the tail window when the last 8 KiB yields nothing."""
    body = bytearray(pad(60000, b"\x11"))
    body[30000:30003] = b"AVBf"
    assert detect_tail(bytes(body)) is TailType.AVBV2


def test_tail_scan_stays_bounded():
    """A stray 4-byte "AVBf" in the payload must not read as a footer.

    The Bash's fallback is `tail -n50`, another tail -- not a rescan of the
    whole image, which would let incidental payload text trigger a footer.
    """
    body = bytearray(pad(4_000_000, b"\x11"))
    body[10:13] = b"AVBf"
    assert detect_tail(bytes(body)) is None


def test_footer_found_inside_window():
    body = bytearray(pad(30000, b"\x11"))
    body[-100:-97] = b"AVBf"
    assert detect_tail(bytes(body)) is TailType.AVBV2


def test_avb1_partition_is_read_from_the_footer_only():
    """A "/boot" in the payload must not outrank the footer's real partition."""
    payload = aosp(0) + pad(30000, b"\x11") + b"Loading /boot/vmlinuz now\n"
    footer = b"\x02\x01\x01\x30\x82" + pad(20) + b"/recovery" + pad(40)
    result = describe(payload + footer)
    assert result.tail is TailType.AVBV1
    assert result.avb_partition is AvbPartition.RECOVERY


# --------------------------------------------------------------------------
# MTK and QCDT
# --------------------------------------------------------------------------


def test_mtk_kernel():
    assert detect_mtk(mtk(b"KERNEL")) is MtkType.KERNEL
    assert has_mtk(mtk(b"KERNEL"))


def test_mtk_rootfs():
    assert detect_mtk(mtk(b"ROOTFS")) is MtkType.ROOTFS


def test_mtk_recovery():
    assert detect_mtk(mtk(b"RECOVERY")) is MtkType.RECOVERY


def test_mtk_without_type_keyword():
    assert has_mtk(mtk())
    assert detect_mtk(mtk()) is None


def test_no_mtk():
    assert detect_mtk(aosp()) is None
    assert not has_mtk(aosp())
    assert detect_mtk(b"") is None


def test_mtk_marker_searched_anywhere():
    assert has_mtk(at(600, MTK))


def test_describe_mtk_prefers_ramdisk():
    assert describe_mtk(mtk(b"KERNEL"), mtk(b"ROOTFS")) is MtkType.ROOTFS


def test_describe_mtk_falls_back_to_kernel():
    assert describe_mtk(mtk(b"KERNEL"), b"plain") is MtkType.KERNEL


def test_describe_mtk_defaults_to_rootfs_when_kernel_only():
    # Bash: "Warning: No MTK header found in ramdisk, assuming rootfs type!"
    assert describe_mtk(mtk(), b"plain") is MtkType.ROOTFS


def test_describe_mtk_none_when_absent_everywhere():
    assert describe_mtk(b"a", b"b") is None
    assert describe_mtk(None, None) is None


def test_qcdt():
    assert detect_qcdt(at(0, b"QCDT" + pad(28)))
    assert not detect_qcdt(aosp())
    assert not detect_qcdt(at(4, b"QCDT"))


# --------------------------------------------------------------------------
# describe()
# --------------------------------------------------------------------------


def test_describe_plain_aosp():
    result = describe(aosp(1))
    assert result == Detection(image_type=ImageType.AOSP, header_version=1)


def test_describe_reports_signature_before_reading_layout():
    # The Bash detects and strips the signature before reading the imgtype, so a
    # still-wrapped buffer yields the signature and nothing else.
    body = bytearray(aosp(2))
    write(body, 0, b"CHROMEOS")
    data = bytes(body) + pad(4000, b"\x11") + b"SEANDROIDENFORCE" + pad(40)
    result = describe(data)
    assert result.signature is SignatureType.CHROMEOS
    assert result.image_type is None


def test_describe_reports_mtk_inside_an_aosp_image():
    # The magic file appends ", MTK headers" under the AOSP rule as well as the
    # ELF one, and unpackimg.sh strips the header off the split kernel whatever
    # the imgtype is.
    body = bytearray(aosp(0))
    write(body, 2048, MTK)
    write(body, 2100, b"KERNEL")
    result = describe(bytes(body))
    assert result.image_type is ImageType.AOSP
    assert result.mtk is MtkType.KERNEL


def test_describe_signature_and_tail_after_unwrapping():
    # Once the signature is stripped the payload describes normally: AOSP
    # header_version 2 plus a Samsung footer.
    data = aosp(2) + pad(4000, b"\x11") + b"SEANDROIDENFORCE" + pad(40)
    result = describe(data)
    assert result.signature is None
    assert result.image_type is ImageType.AOSP
    assert result.header_version == 2
    assert result.tail is TailType.SEANDROID


def test_describe_reports_loki_partition():
    result = describe(loki(partition=1))
    assert result.patch is PatchType.LOKI
    assert result.loki_partition is LokiPartition.RECOVERY


def test_describe_raises_on_unrecognized():
    with pytest.raises(CorruptImageError) as excinfo:
        describe(at(0, b"\xde\xad\xbe\xef"))
    assert "Unrecognized" in str(excinfo.value)


def test_describe_raises_on_empty():
    with pytest.raises(CorruptImageError):
        describe(b"")


def test_check_imgtype_accepts_all_supported():
    for name in ("AOSP", "AOSP_VNDR", "AOSP-PXA", "ELF", "KRNL", "OSIP", "U-Boot"):
        assert check_imgtype(name) is ImageType(name)


def test_check_imgtype_rejects_unknown():
    with pytest.raises(UnsupportedFormatError):
        check_imgtype("SHRIMPTON")


# --------------------------------------------------------------------------
# short / truncated buffers must not raise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 8, 16, 32, 63, 64, 100, 512, 1023])
def test_short_buffers_return_none_or_value(size):
    data = pad(size)
    for fn in (
        detect_signature,
        detect_image_type,
        detect_patch,
        detect_tail,
        detect_mtk,
    ):
        fn(data)  # must not raise
    loki_partition(data)
    header_version(data)
    has_mtk(data)
    detect_qcdt(data)


def test_truncated_candidates_return_none():
    assert detect_signature(b"CHROMEO") is None
    assert detect_image_type(b"ANDROID") is None
    assert detect_patch(b"ANDROID!") is None
    assert detect_tail(b"AVB") is None
    assert detect_mtk(b"\x88\x16\x88") is None


def test_bare_aosp_magic_is_still_aosp():
    # Only 8 bytes: the PXA marker is unreadable, which must not raise.
    assert detect_image_type(b"ANDROID!") is ImageType.AOSP
    assert header_version(b"ANDROID!") is None


# --------------------------------------------------------------------------
# real images from the bundled mkbootimg
# --------------------------------------------------------------------------

MKBOOTIMG = BIN / "mkbootimg"
PXA_MKBOOTIMG = BIN / "pxa-mkbootimg"

needs_tools = pytest.mark.skipif(
    not MKBOOTIMG.exists() or not PXA_MKBOOTIMG.exists(),
    reason="bundled mkbootimg binaries not available for this platform",
)


def build_image(tmp_path: Path, tool: Path, extra: list[str]) -> bytes:
    (tmp_path / "k.bin").write_bytes(b"K" * 1024)
    (tmp_path / "rd.bin").write_bytes(b"R" * 100)
    out = tmp_path / "out.img"
    subprocess.run(
        [
            str(tool),
            "--kernel",
            str(tmp_path / "k.bin"),
            "--ramdisk",
            str(tmp_path / "rd.bin"),
            "--cmdline",
            "console=ttyS0",
            "--board",
            "test",
            "--base",
            "0x80000000",
            "--pagesize",
            "4096",
            *extra,
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out.read_bytes()


@needs_tools
@pytest.mark.parametrize("version", [0, 1, 2, 3, 4])
def test_real_mkbootimg_images(tmp_path, version):
    data = build_image(tmp_path, MKBOOTIMG, ["--header_version", str(version)])
    assert detect_image_type(data) is ImageType.AOSP
    assert header_version(data) == version


@needs_tools
def test_real_pxa_image(tmp_path):
    data = build_image(tmp_path, PXA_MKBOOTIMG, ["--pagesize", "2048"])
    assert data[:8] == b"ANDROID!"
    # The real tool puts 00 00 00 03 at the PXA offset 0x24.
    assert data[0x24:0x28] == b"\x00\x00\x00\x03"
    assert detect_image_type(data) is ImageType.AOSP_PXA
    assert detect_patch(data) is None
    # 0x28 holds 00 01 00 80 in a real PXA image, so no version is reported.
    assert header_version(data) is None
    assert describe(data).image_type is ImageType.AOSP_PXA


@needs_tools
def test_real_images_have_no_footers_or_signatures(tmp_path):
    data = build_image(tmp_path, MKBOOTIMG, ["--header_version", "0"])
    assert detect_signature(data) is None
    assert detect_tail(data) is None
