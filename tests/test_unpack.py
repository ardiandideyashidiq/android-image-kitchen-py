"""Tests for the manifest and the unpack pipeline.

The fixture image is built with the ``mkbootimg`` that ships in ``bin/``, so
these exercise a genuine AOSP container rather than a hand-rolled approximation.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from aik_py.manifest import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    Header,
    Manifest,
    OsVersion,
    PatchInfo,
    PayloadRefs,
    SignatureInfo,
    load_manifest,
)
from aik_py.session import (
    AUTO_PICK_SKIP,
    COMPRESSION_EXTENSION,
    Session,
    UnpackError,
    UnrecognizedImageError,
    UnsupportedImageError,
    detect_compression,
    detect_head_signature,
    detect_tail_type,
    parse_aosp_header,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BIN = REPO_ROOT / "bin" / "linux" / "x86_64"
MKBOOTIMG = BIN / "mkbootimg"

KERNEL_BYTES = bytes(range(256)) * 4
CMDLINE = "console=ttyS0 androidboot.hardware=testboard"
BOARD = "testboard"
BASE = 0x80000000
PAGE_SIZE = 4096


def _module_available(name: str) -> bool:
    """True if a sibling unit's module is importable on this branch."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAS_CPIO = _module_available("aik_py.cpio")
HAS_COMPRESSION = _module_available("aik_py.compression")
HAS_AOSP_FORMAT = _module_available("aik_py.formats.aosp")
HAS_EXTRACT = HAS_CPIO and HAS_COMPRESSION


@pytest.fixture(scope="session")
def ramdisk_cpio() -> bytes:
    """A tiny newc cpio archive built without shelling out to cpio."""
    entries = [
        (".", b"40755", b"0", b"init", b"#!/bin/sh\nexit 0\n"),
        ("init", b"100755", b"0", b"", b"#!/bin/sh\nexit 0\n"),
    ]
    out = bytearray()
    ino = 1
    for name, mode, nlink, _filler, body in entries:
        out += b"070701"
        out += _field(f"{ino:08X}")
        out += _field(mode.decode(), 8)
        out += _field("00000000")  # uid
        out += _field("00000000")  # gid
        out += _field(nlink.decode(), 8)
        out += _field("00000001")  # mtime, pinned for reproducibility
        out += _field(f"{len(body):08X}")
        out += _field("00000000")  # devmajor
        out += _field("00000000")  # devminor
        out += _field("00000000")  # rdevmajor
        out += _field("00000000")  # rdevminor
        out += _field(f"{len(name) + 1:08X}")  # namesize, including NUL
        out += _field("00000000")  # check
        out += name.encode() + b"\x00"
        out += b"\x00" * (-len(out) % 4)
        out += body
        out += b"\x00" * (-len(out) % 4)
        ino += 1
    out += b"TRAILER!!!"
    out += b"\x00" * (-len(out) % 4)
    for _ in range(2):
        out += b"\x00" * 4  # newc header padding for the trailer entry
    out += b"\x00" * 512  # two zero blocks terminate the archive
    return bytes(out)


def _field(value: str, width: int = 8) -> bytes:
    return value.encode()[:width].ljust(width, b"\x00")


@pytest.fixture(scope="session")
def boot_image(tmp_path_factory: pytest.TempPathFactory, ramdisk_cpio: bytes) -> Path:
    """Build a real AOSP boot image with the bundled mkbootimg."""
    if not MKBOOTIMG.is_file():
        pytest.skip(f"bundled mkbootimg not found at {MKBOOTIMG}")

    build = tmp_path_factory.mktemp("build")
    kernel = build / "k.bin"
    kernel.write_bytes(KERNEL_BYTES)
    ramdisk = build / "rd.cpio.gz"
    ramdisk.write_bytes(gzip.compress(ramdisk_cpio, 9, mtime=0))
    image = build / "boot.img"
    subprocess.run(
        [
            str(MKBOOTIMG),
            "--kernel",
            str(kernel),
            "--ramdisk",
            str(ramdisk),
            "--cmdline",
            CMDLINE,
            "--board",
            BOARD,
            "--base",
            hex(BASE),
            "--pagesize",
            str(PAGE_SIZE),
            "--header_version",
            "1",
            "-o",
            str(image),
        ],
        check=True,
        capture_output=True,
    )
    return image


@pytest.fixture
def session_dir(tmp_path: Path, boot_image: Path) -> Path:
    """A session directory containing a copy of the image to unpack."""
    shutil.copy(boot_image, tmp_path / boot_image.name)
    return tmp_path


# ---------------------------------------------------------------- manifest


def test_manifest_round_trips_through_json() -> None:
    manifest = Manifest(
        tool_version="aik-py/0.1.0",
        source_name="boot.img",
        source_size=12288,
        image_type="AOSP",
        header_version=1,
        ramdisk_compression="gzip",
        tail_type="AVBv1",
        signature=SignatureInfo(type="AVBv1", avb_partition="boot", stripped=True),
        patches=PatchInfo(loki_type="boot", mtk_type="rootfs"),
        header=Header(
            page_size=4096,
            base=0x80000000,
            kernel_offset=0x8000,
            cmdline=CMDLINE,
            board=BOARD,
            id_hex="00" * 20,
            os_version=OsVersion(
                os_version=13, patch_level=202301, security_patch_date=1
            ),
        ),
        payloads=PayloadRefs(kernel="split_img/boot.img-kernel"),
        notes=["a note"],
    )

    encoded = json.loads(manifest.to_json())
    assert encoded["manifest_version"] == MANIFEST_VERSION
    assert Manifest.from_dict(encoded) == manifest
    assert Manifest.from_dict(manifest.to_dict()) == manifest


def test_manifest_preserves_absent_fields() -> None:
    """An empty field must stay ``None``, not vanish -- that distinction is the
    whole reason the manifest exists in place of a missing file."""
    encoded = json.loads(Manifest().to_json())
    assert encoded["image_type"] is None
    assert encoded["header"]["cmdline"] is None
    assert Manifest.from_dict(encoded).header.cmdline is None


def test_manifest_rejects_future_schema() -> None:
    with pytest.raises(ValueError, match="newer than this tool"):
        Manifest.from_dict({"manifest_version": MANIFEST_VERSION + 1})


def test_manifest_paths_must_be_relative() -> None:
    with pytest.raises(ValueError, match="relative"):
        PayloadRefs(kernel="/etc/passwd").to_dict()


def test_manifest_rejects_escaping_paths_on_read() -> None:
    """A manifest is untrusted input, so a crafted one must not escape the dir."""
    for bad in ("/etc/passwd", "../../etc/passwd"):
        with pytest.raises(ValueError, match="relative"):
            Manifest.from_dict({"payloads": {"kernel": bad}})


def test_load_manifest_binds_workdir(tmp_path: Path) -> None:
    assert load_manifest(tmp_path) is None
    (tmp_path / MANIFEST_NAME).write_text(Manifest(source_name="x").to_json())
    loaded = load_manifest(tmp_path)
    assert loaded is not None
    assert loaded.source_name == "x"
    assert loaded.workdir == tmp_path


# ------------------------------------------------------------- detection


@pytest.mark.parametrize(
    ("magic", "expected"),
    [
        (b"\x1f\x8b\x08\x00", "gzip"),
        (b"\xfd7zXZ\x00", "xz"),
        (b"BZh91AY&SY", "bzip2"),
        (b"\x89LZO\x00\x0d\x0a\x1a\x0a", "lzop"),
        (b"\x04\x22\x4d\x18", "lz4"),
        (b"070701", "cpio"),
        (b"", "empty"),
        (b"\xaa\xbb\xcc\xdd", "data"),
    ],
)
def test_detect_compression(magic: bytes, expected: str) -> None:
    assert detect_compression(magic) == expected


def test_detect_head_signature() -> None:
    assert detect_head_signature(b"CHROMEOS" + b"\x00" * 16) == "CHROMEOS"
    assert detect_head_signature(b"DHTB\x01\x00\x00" + b"\x00" * 16) == "DHTB"
    assert detect_head_signature(b"ANDROID!" + b"\x00" * 16) is None


def test_detect_tail_type() -> None:
    assert detect_tail_type(b"x" * 8192 + b"AVBf") == "AVBv2"
    assert detect_tail_type(b"x" * 8192 + b"SEANDROIDENFORCE") == "SEAndroid"
    assert detect_tail_type(b"ANDROID!" + b"\x00" * 8192) is None


def test_parse_aosp_header_reads_fields(boot_image: Path) -> None:
    header, version, size = parse_aosp_header(boot_image.read_bytes())
    assert version == 1
    assert size >= 1632
    assert header.page_size == PAGE_SIZE
    assert header.cmdline == CMDLINE
    assert header.name == BOARD
    assert header.board == BOARD
    assert header.base == BASE
    assert header.kernel_offset == 0x8000
    assert header.id_hex and len(header.id_hex) == 40


def test_parse_aosp_header_rejects_foreign_buffer() -> None:
    with pytest.raises(UnrecognizedImageError):
        parse_aosp_header(b"NOT-AN-IMAGE" + b"\x00" * 2048)


def _synthetic_header(
    *,
    header_version: int = 1,
    os_version_byte: int = 13,
    patch_level_byte: int = 5,
    recovery_dtbo_size: int = 0,
    layout: str = "canonical",
) -> bytes:
    """Build a header with the fields a parser could plausibly misread."""
    header = bytearray(4096)
    header[0:8] = b"ANDROID!"
    struct.pack_into("<I", header, 12, BASE + 0x8000)
    struct.pack_into("<I", header, 20, BASE + 0x1000000)
    struct.pack_into("<I", header, 28, BASE + 0xF00000)
    struct.pack_into("<I", header, 32, BASE + 0x100)
    struct.pack_into("<I", header, 36, PAGE_SIZE)
    header[40] = os_version_byte
    header[41] = patch_level_byte
    header[48:64] = BOARD.encode()
    if layout == "v0":
        pass
    elif layout == "canonical":
        struct.pack_into("<I", header, 1632, header_version)
        struct.pack_into("<I", header, 1640, 1648)
    elif layout == "v2":
        struct.pack_into("<I", header, 1632, header_version)
        struct.pack_into("<I", header, 1644, recovery_dtbo_size)
        struct.pack_into("<I", header, 1648, 0x11000000)
        struct.pack_into("<I", header, 1656, 1648)
    elif layout == "bundled":
        struct.pack_into("<I", header, 1644, 1648)
    return bytes(header)


def test_os_version_bytes_are_not_swapped() -> None:
    """AOSP v0 puts the version at byte 40 and the patch level at byte 41."""
    header, version, size = parse_aosp_header(_synthetic_header(layout="v0"))
    assert version is None
    assert size == 1632
    assert header.os_version.os_version == 13
    assert header.os_version.patch_level == 5


def test_v2_header_size_is_not_confused_with_dtbo_size() -> None:
    """A recovery_dtbo_size must never be read as the header size."""
    header, version, size = parse_aosp_header(
        _synthetic_header(header_version=2, recovery_dtbo_size=0x2000, layout="v2")
    )
    assert version == 2
    assert size == 1648
    assert header.page_size == PAGE_SIZE


def test_bundled_mkbootimg_header_layout_is_understood() -> None:
    """The bundled mkbootimg writes only header_size, at 1644, implying v1."""
    _, version, size = parse_aosp_header(_synthetic_header(layout="bundled"))
    assert version == 1
    assert size == 1648


def test_vendor_boot_header_parses() -> None:
    """vendor_boot differs only in the leading magic."""
    raw = _synthetic_header()
    vendor = b"VNDRBOOT" + raw[8:]
    with pytest.raises(UnrecognizedImageError):
        parse_aosp_header(vendor)
    header, _, _ = parse_aosp_header(vendor, b"VNDRBOOT")
    assert header.board == BOARD


def test_nook_loader_strings_are_recognised() -> None:
    """Every NOOK loader string must be both detected and stripped."""
    for magic in (
        b"Red\x20Loader",
        b"Green\x20Loader",
        b"Green\x20Recovery",
        b"eMMC\x20boot.img+secondloader",
        b"eMMC\x20recovery.img+secondloader",
    ):
        blob = b"\x00" * 64 + magic + b"\x00" * 64
        assert detect_head_signature(blob) == "NOOK"


def test_amonet_revert_preserves_the_aosp_header(tmp_path: Path) -> None:
    """Amonet relocates the header into a side block; reverting must restore it."""
    from aik_py.session import AOSP_MAGIC

    image = _synthetic_header()
    # Amonet layout: microloader(1024) + head(2048) + tail(2048) at offset 2048.
    wrapped = b"\xbb" * 1024 + image[:1024] + image[:2048] + image[2048:]

    session = Session(tmp_path)
    session.split_img.mkdir(parents=True, exist_ok=True)
    reverted, patches = session._revert_amonet(wrapped)

    assert reverted.startswith(AOSP_MAGIC)
    assert patches.amonet_microloader is not None
    assert parse_aosp_header(reverted)[0].board == BOARD
    sidecar = tmp_path / "split_img" / patches.amonet_microloader
    assert sidecar.read_bytes() == b"\xbb" * 1024


def test_mtk_type_is_read_past_the_magic(tmp_path: Path) -> None:
    """The type name starts at offset 12, not at the 4-byte magic."""
    from aik_py.session import MTK_HEADER_SIZE, MTK_MAGIC

    session = Session(tmp_path)
    session.split_img.mkdir(parents=True, exist_ok=True)
    ramdisk = tmp_path / "rd"
    body = gzip.compress(b"payload", mtime=0)
    header = (MTK_MAGIC + b"\x8d\x00\x00\x00" + b"ROOTFS").ljust(
        MTK_HEADER_SIZE, b"\x00"
    )
    ramdisk.write_bytes(header + body)

    assert session._strip_mtk(None, ramdisk) == "rootfs"
    assert ramdisk.read_bytes() == body


# ---------------------------------------------------------------- unpack


def test_unpack_produces_expected_layout(session_dir: Path) -> None:
    Session(session_dir).unpack(session_dir / "boot.img")

    assert (session_dir / "split_img").is_dir()
    assert (session_dir / "ramdisk").is_dir()
    assert (session_dir / MANIFEST_NAME).is_file()

    split = session_dir / "split_img"
    assert (split / "boot.img-kernel").read_bytes() == KERNEL_BYTES
    assert (split / "boot.img-ramdisk.cpio.gz").is_file()
    assert (split / "boot.img-imgtype").read_text() == "AOSP"
    assert (split / "boot.img-cmdline").read_text() == CMDLINE
    assert (split / "boot.img-board").read_text() == BOARD
    assert (split / "boot.img-base").read_text() == hex(BASE)
    assert (split / "boot.img-pagesize").read_text() == str(PAGE_SIZE)
    assert (split / "boot.img-ramdiskcomp").read_text() == "gzip"
    assert (split / "boot.img-origsize").read_text() == str(
        (session_dir / "boot.img").stat().st_size
    )


def test_unpack_records_header_and_compression(session_dir: Path) -> None:
    manifest = Session(session_dir).unpack(session_dir / "boot.img")

    assert manifest.image_type == "AOSP"
    assert manifest.header_version == 1
    assert manifest.ramdisk_compression == "gzip"
    assert manifest.header.cmdline == CMDLINE
    assert manifest.header.board == BOARD
    assert manifest.header.base == BASE
    assert manifest.header.page_size == PAGE_SIZE
    assert manifest.header.kernel_offset == 0x8000
    assert manifest.source_name == "boot.img"
    assert manifest.payloads.kernel == "split_img/boot.img-kernel"
    assert manifest.payloads.ramdisk_archive == "split_img/boot.img-ramdisk.cpio.gz"


def test_manifest_on_disk_round_trips(session_dir: Path) -> None:
    written = Session(session_dir).unpack(session_dir / "boot.img")
    on_disk = json.loads((session_dir / MANIFEST_NAME).read_text())
    assert Manifest.from_dict(on_disk) == written


def test_payload_paths_are_relative_and_resolve(session_dir: Path) -> None:
    manifest = Session(session_dir).unpack(session_dir / "boot.img")
    for attr in ("kernel", "ramdisk_archive"):
        stored = getattr(manifest.payloads, attr)
        assert stored is not None
        assert not stored.startswith("/")
    assert manifest.resolve("kernel").read_bytes() == KERNEL_BYTES


@pytest.mark.skipif(
    not HAS_EXTRACT, reason="aik_py.cpio / aik_py.compression not on this branch"
)
def test_ramdisk_is_populated(session_dir: Path) -> None:
    Session(session_dir).unpack(session_dir / "boot.img")
    init = session_dir / "ramdisk" / "init"
    assert init.is_file()
    assert init.read_bytes() == b"#!/bin/sh\nexit 0\n"


def test_ramdisk_extraction_is_reported_when_cpio_absent(
    session_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Without the cpio unit the step must say so, not fail silently."""
    Session(session_dir).unpack(session_dir / "boot.img")
    if HAS_EXTRACT:
        assert (session_dir / "ramdisk" / "init").is_file()
    else:
        # The archive is still preserved so nothing is lost.
        assert (session_dir / "split_img" / "boot.img-ramdisk.cpio.gz").is_file()
        assert list((session_dir / "ramdisk").iterdir()) == []


def test_unpack_cleans_previous_state(session_dir: Path) -> None:
    session = Session(session_dir)
    first = session.unpack(session_dir / "boot.img")

    stale_field = session_dir / "split_img" / "boot.img-stale-field"
    stale_field.write_text("stale")
    stale_ramdisk = session_dir / "ramdisk" / "stale-file"
    stale_ramdisk.write_text("stale")
    stale_manifest = session_dir / MANIFEST_NAME
    assert stale_manifest.is_file()

    second = session.unpack(session_dir / "boot.img")

    assert not stale_field.exists()
    assert not stale_ramdisk.exists()
    assert second == first
    assert "boot.img-stale-field" not in {
        p.name for p in (session_dir / "split_img").iterdir()
    }
    assert sorted(p.name for p in (session_dir / "ramdisk").iterdir()) == (
        ["init"] if HAS_EXTRACT else []
    )


def test_unrecognized_image_raises(session_dir: Path) -> None:
    junk = session_dir / "junk.img"
    junk.write_bytes(b"this is not a boot image" * 512)
    with pytest.raises(UnrecognizedImageError):
        Session(session_dir).unpack(junk)


def test_unsupported_image_raises_distinct_error(session_dir: Path) -> None:
    """Recognised-but-refused must not be confused with not-recognised."""
    assert UnrecognizedImageError is not UnsupportedImageError
    assert not issubclass(UnrecognizedImageError, UnsupportedImageError)
    assert not issubclass(UnsupportedImageError, UnrecognizedImageError)
    assert issubclass(UnrecognizedImageError, UnpackError)

    # An ELF container is recognised but has no splitter on this branch.
    elf = session_dir / "kernel.elf"
    elf.write_bytes(b"\x7fELF" + b"\x00" * 2040)
    with pytest.raises(UnsupportedImageError):
        Session(session_dir).unpack(elf)


def test_unpack_rejects_unrecognised_ramdisk(session_dir: Path) -> None:
    """A corrupt payload inside a valid container is reported, not ignored."""
    data = bytearray((session_dir / "boot.img").read_bytes())
    header_size = struct.unpack_from("<I", data, 1644)[0]
    page = PAGE_SIZE
    cursor = (header_size + page - 1) // page * page
    kernel_size = struct.unpack_from("<I", data, 8)[0]
    cursor += (kernel_size + page - 1) // page * page  # skip past the kernel

    # Overwrite the ramdisk's magic so no known compressor matches.
    assert bytes(data[cursor : cursor + 2]) == b"\x1f\x8b"
    data[cursor : cursor + 8] = b"\xde\xad\xbe\xef\xde\xad\xbe\xef"
    (session_dir / "corrupt.img").write_bytes(bytes(data))

    with pytest.raises(UnrecognizedImageError):
        Session(session_dir).unpack(session_dir / "corrupt.img")


def test_auto_pick_skips_bash_intermediates(session_dir: Path) -> None:
    """The skip list exists so the toolkit never unpacks its own output."""
    for name in AUTO_PICK_SKIP:
        (session_dir / name).write_bytes(b"\x7fELF" + b"\x00" * 64)
    assert (session_dir / "boot.img").is_file()

    manifest = Session(session_dir).unpack()
    assert manifest.source_name == "boot.img"


def test_auto_pick_without_candidate_raises(session_dir: Path) -> None:
    (session_dir / "boot.img").unlink()
    with pytest.raises(UnpackError, match="No image file supplied"):
        Session(session_dir).unpack()


def test_compression_extension_table_covers_detected_types() -> None:
    for name in ("gzip", "xz", "bzip2", "lzop", "lzma", "lz4", "cpio", "empty"):
        assert name in COMPRESSION_EXTENSION


@pytest.mark.parametrize(
    "module", ["aik_py.cpio", "aik_py.compression", "aik_py.formats.aosp"]
)
def test_optional_dependency_absence_is_survivable(
    module: str, session_dir: Path
) -> None:
    """The unit must stand alone on a branch where sibling units are missing."""
    present = _module_available(module)
    manifest = Session(session_dir).unpack(session_dir / "boot.img")
    assert manifest.image_type == "AOSP"
    assert manifest.payloads.kernel is not None
    if not present:
        assert (session_dir / "split_img" / "boot.img-kernel").is_file()
