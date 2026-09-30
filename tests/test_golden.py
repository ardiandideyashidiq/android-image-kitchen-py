"""Layer 2: round-trip tests against real OEM images.

These exercise the library against genuine 64 MiB images rather than
synthetic ones, which is the only way to catch the failure modes that matter
in production: a header shape that no real image uses, a page-size assumption
that only holds for 4096, a parser that trips over trailing padding.

Fixture provenance
------------------
``boot.img`` is a real OEM image, 67108864 bytes, ``ANDROID!`` magic,
``header_version`` 4, ``kernel_size`` 18885902, empty ``name`` and ``cmdline``.
Its header is sparse: most fields are zero and the interesting ones sit at the
compact v3/v4 offsets, i.e. ``kernel_size@0x08``, ``ramdisk_size@0x0C``,
``os_version@0x10``, ``header_size@0x14`` (1584), ``header_version@0x28`` (4).
The kernel payload begins at 0x1000.

The two ``vendor_boot`` files are a **non-standard variant**. Their first page
is only 52 bytes of header, and the cmdline string starts immediately at
offset 0x1C and runs to the end of the non-zero region. The bytes at the
nominal ``cmdline_size`` (0x1C), ``name_size`` (0x24) and ``header_size``
(0x28) offsets are actually cmdline text, not little-endian integers: reading
them as u32 yields 1953460066, 861090870 and 1311912748 respectively. So these
files are used only as opaque round-trip targets and as evidence the parser
does not crash on real data -- **no assertion here expects them to match the
AOSP field layout, because they provably do not.**

Every test in this module auto-skips when the fixture is absent. These files
are 64 MiB each and are not checked into the repository, so on CI or on
another machine every test here skips.
"""

from __future__ import annotations

import importlib
import os
import struct
from pathlib import Path

import pytest
from test_roundtrip import assert_clean_error, field, pack_image, parse_image

# --------------------------------------------------------------------------
# Fixture locations. Override with AIK_PY_TEST_IMAGE_DIR to relocate them.
# --------------------------------------------------------------------------

_IMAGE_DIR = Path(os.environ.get("AIK_PY_TEST_IMAGE_DIR", Path.home() / "Downloads"))


def _image(name: str) -> Path:
    return _IMAGE_DIR / name


BOOT_IMG = _image("boot.img")
VENDOR_BOOT_IMG = _image("vendor_boot.img")
VENDOR_BOOT_ALT_IMG = _image("vendor_boot(1).img")

# Module-level skip: none of these 64 MiB images are checked into the
# repository, so on CI or on another machine every test here skips cleanly.
# Individual tests still call _require_file() so that a partially-populated
# fixture directory skips the right tests rather than erroring on the rest.
pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(
        not BOOT_IMG.is_file() and not VENDOR_BOOT_IMG.is_file(),
        reason=(
            "no real OEM image fixtures found in "
            f"{_IMAGE_DIR} (set AIK_PY_TEST_IMAGE_DIR to relocate them); "
            "these 64 MiB images are not checked into the repository"
        ),
    ),
]

# The real images were measured on the machine that produced this suite; the
# values below are the ground truth the assertions are written against.
REAL_BOOT_SIZE = 67108864
REAL_KERNEL_SIZE = 18885902
REAL_HEADER_SIZE = 1584
REAL_PAGE_SIZE = 4096
REAL_KERNEL_OFFSET = 0x1000

VENDOR_MAGIC = b"VNDRBOOT"
VENDOR_NONSTANDARD_CMDLINE_OFFSET = 0x1C


def _require_file(path: Path) -> None:
    """Skip a test when its 64 MiB fixture is absent.

    These images are not in the repository, so on CI or on any other machine
    every test that calls this skips cleanly instead of erroring.
    """
    if not path.is_file():
        pytest.skip(f"real image fixture not found: {path}")


def _first_difference(left: bytes, right: bytes) -> str:
    """Describe where two byte strings diverge, for a useful assertion message."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return f"{index} ({a:#04x} vs {b:#04x})"
    return f"length {len(left)} vs {len(right)}"


def _read(path: Path) -> bytes:
    return path.read_bytes()


# --------------------------------------------------------------------------
# Vendor-boot adapter.
#
# The vendor_boot images carry the VNDRBOOT magic, so a boot-image parser is
# right to reject them. They need the vendor module, which is a separate
# concern from the ANDROID! format.
#
# This is deliberately NOT a module-level importorskip: a skip raised during
# collection discards every test in the file, so gating on the vendor module
# would silently take the valuable boot.img round-trip tests down with it.
# The module is resolved lazily and only the vendor tests need it.
# --------------------------------------------------------------------------


def _vendor_module():
    """Import the vendor module, skipping only the caller if it is absent."""
    try:
        return importlib.import_module("aik_py.formats.vendor")
    except ImportError as exc:
        pytest.skip(f"aik_py.formats.vendor not implemented yet: {exc}")


def parse_vendor(data: bytes):
    """Parse a VNDRBOOT image with the vendor module."""
    vendor = _vendor_module()
    for name in ("parse", "unpack", "load", "loads", "read", "from_bytes"):
        fn = getattr(vendor, name, None)
        if callable(fn):
            try:
                return fn(data)
            except TypeError:
                continue
    raise AssertionError(
        f"{vendor.__name__} exposes none of the expected parse entry points; "
        f"found {sorted(n for n in dir(vendor) if not n.startswith('_'))!r}"
    )


def pack_vendor(img) -> bytes:
    """Serialise a parsed vendor image back to bytes."""
    vendor = _vendor_module()
    for name in ("pack", "dump", "to_bytes", "build", "serialize", "write"):
        fn = getattr(img, name, None)
        if callable(fn):
            try:
                out = fn()
            except TypeError:
                continue
            if isinstance(out, (bytes, bytearray)):
                return bytes(out)
    for name in ("pack", "dump", "to_bytes", "build", "serialize", "write"):
        fn = getattr(vendor, name, None)
        if callable(fn):
            try:
                out = fn(img)
            except TypeError:
                continue
            if isinstance(out, (bytes, bytearray)):
                return bytes(out)
    raise AssertionError(
        f"could not find a packer for {type(img).__name__!r} in {vendor.__name__}"
    )


# --------------------------------------------------------------------------
# Session-scoped parses: a 64 MiB image is parsed once, not once per test.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def boot_bytes() -> bytes:
    _require_file(BOOT_IMG)
    return _read(BOOT_IMG)


@pytest.fixture(scope="module")
def boot_image(boot_bytes: bytes):
    return parse_image(boot_bytes, 4)


# --------------------------------------------------------------------------
# boot.img
# --------------------------------------------------------------------------


def test_real_boot_img_is_v4_android(boot_bytes: bytes):
    """The fixture is what it claims to be: a 64 MiB v4 ANDROID! image."""
    _require_file(BOOT_IMG)
    assert len(boot_bytes) == REAL_BOOT_SIZE, (
        f"boot.img is {len(boot_bytes)} bytes, expected {REAL_BOOT_SIZE}"
    )
    assert boot_bytes[0:8] == b"ANDROID!", (
        f"boot.img magic is {boot_bytes[0:8]!r}, expected b'ANDROID!'"
    )
    assert struct.unpack_from("<I", boot_bytes, 0x28)[0] == 4, (
        "boot.img header_version at 0x28 is "
        f"{struct.unpack_from('<I', boot_bytes, 0x28)[0]}, expected 4"
    )


def test_real_boot_img_header_fields(boot_image):
    """The parser reads the real v4 header, including its sparse fields."""
    _require_file(BOOT_IMG)
    assert field(boot_image, "header_version", "version") == 4, (
        "header_version: image stores 4, parser reported "
        f"{field(boot_image, 'header_version', 'version')!r}"
    )
    assert field(boot_image, "header_size") == REAL_HEADER_SIZE, (
        f"header_size: image stores {REAL_HEADER_SIZE} at offset 0x14, parser "
        f"reported {field(boot_image, 'header_size')!r}"
    )
    # The v3/v4 header has no separate size attributes, so assert on the
    # recovered payload lengths: kernel_size is 18885902 and ramdisk_size is
    # genuinely 0 (this image ships no ramdisk).
    kernel = field(boot_image, "kernel", "kernel_data")
    assert kernel is not None, "kernel payload came back as None"
    assert len(kernel) == REAL_KERNEL_SIZE, (
        f"kernel_size: image stores {REAL_KERNEL_SIZE}, parser recovered a "
        f"{len(kernel)}-byte kernel"
    )
    ramdisk = field(boot_image, "ramdisk", "ramdisk_data")
    assert ramdisk is None, (
        "ramdisk_size: image stores 0, so the ramdisk must be None; parser "
        f"reported {ramdisk!r}"
    )
    # Empty name and cmdline. The bytes are all zero, so the parser must not
    # invent content for them.
    assert bytes(field(boot_image, "name", "board")).rstrip(b"\0") == b"", (
        f"name: image is empty, parser reported {field(boot_image, 'name', 'board')!r}"
    )
    assert (
        bytes(field(boot_image, "cmdline", "cmd_line", "args")).rstrip(b"\0") == b""
    ), (
        f"cmdline: image is empty, parser reported "
        f"{field(boot_image, 'cmdline', 'cmd_line', 'args')!r}"
    )


def test_real_boot_img_kernel_extracts_to_right_size(boot_image, boot_bytes: bytes):
    """The kernel payload extracts to exactly the declared size.

    Cross-checked against the raw file: the payload must equal
    ``boot_bytes[0x1000 : 0x1000 + kernel_size]``, not merely be the right
    length.
    """
    _require_file(BOOT_IMG)
    kernel = field(boot_image, "kernel", "kernel_data")
    assert kernel is not None, "kernel payload came back as None"
    assert len(kernel) == REAL_KERNEL_SIZE, (
        f"kernel extracted to {len(kernel)} bytes, expected {REAL_KERNEL_SIZE}"
    )
    expected = boot_bytes[REAL_KERNEL_OFFSET : REAL_KERNEL_OFFSET + REAL_KERNEL_SIZE]
    assert bytes(kernel) == expected, (
        "kernel payload does not match the bytes at "
        f"0x{REAL_KERNEL_OFFSET:x}..0x{REAL_KERNEL_OFFSET + REAL_KERNEL_SIZE:x}; "
        f"first difference at byte {_first_difference(bytes(kernel), expected)}"
    )
    # Sanity: a real ARM64 kernel image, so the extraction is plausible.
    assert bytes(kernel[:4]) == b"\x02\x21\x4c\x18", (
        f"kernel does not start with the expected ARM64 header: {bytes(kernel[:4])!r}"
    )


def test_real_boot_img_absent_ramdisk_is_none(boot_image):
    """An image with no ramdisk reports None, not empty bytes."""
    _require_file(BOOT_IMG)
    assert field(boot_image, "ramdisk", "ramdisk_data") is None, (
        "this boot.img has no ramdisk; expected None but got "
        f"{field(boot_image, 'ramdisk', 'ramdisk_data')!r}"
    )


def test_real_boot_img_repacks_to_identical_parse(boot_image):
    """The most valuable test in this suite: a true round trip on a real image.

    Original 64 MiB OEM bytes -> parse -> pack -> parse, and every header field
    and payload must come back unchanged. This is what catches a parser that
    only works on synthetic images.
    """
    _require_file(BOOT_IMG)
    repacked = pack_image(boot_image)
    again = parse_image(repacked, 4)

    assert field(again, "header_version", "version") == 4, (
        "header_version changed across a real-image round trip: 4 -> "
        f"{field(again, 'header_version', 'version')!r}"
    )
    assert field(again, "header_size") == REAL_HEADER_SIZE, (
        f"header_size changed across a real-image round trip: {REAL_HEADER_SIZE} -> "
        f"{field(again, 'header_size')!r}"
    )

    original_kernel = bytes(field(boot_image, "kernel", "kernel_data"))
    repacked_kernel = bytes(field(again, "kernel", "kernel_data"))
    assert len(repacked_kernel) == REAL_KERNEL_SIZE, (
        "kernel payload changed size across a real-image round trip: "
        f"{REAL_KERNEL_SIZE} -> {len(repacked_kernel)}"
    )
    assert repacked_kernel == original_kernel, (
        "kernel payload changed content across a real-image round trip; first "
        f"difference at byte {_first_difference(repacked_kernel, original_kernel)}"
    )


def test_real_boot_img_repack_is_deterministic(boot_image):
    """Packing the real image twice is byte-identical."""
    _require_file(BOOT_IMG)
    first = pack_image(boot_image)
    second = pack_image(boot_image)
    assert first == second, (
        "packing the same real boot.img twice produced different bytes "
        f"({len(first)} vs {len(second)} bytes)"
    )


def test_real_boot_img_truncated_raises_clean_error(boot_bytes: bytes):
    """A truncated real image is reported cleanly, not a raw struct/index error."""
    _require_file(BOOT_IMG)
    truncated = boot_bytes[: REAL_KERNEL_OFFSET + 1024]
    with pytest.raises(Exception) as excinfo:
        parse_image(truncated, 4)
    assert_clean_error(excinfo.value, context="truncated real boot.img")


def test_real_boot_img_corrupt_magic_raises_clean_error(boot_bytes: bytes):
    """A real image with a clobbered magic is reported, not mis-parsed."""
    _require_file(BOOT_IMG)
    # Exactly 8 bytes so the image differs only in the magic.
    corrupted = b"NOTBOOT!" + boot_bytes[8:]
    with pytest.raises(Exception) as excinfo:
        parse_image(corrupted, 4)
    assert_clean_error(excinfo.value, context="corrupt-magic real boot.img")
    message = str(excinfo.value).lower()
    assert "magic" in message or "android" in message, (
        f"corrupt-magic error message is not actionable: {excinfo.value!r}"
    )


def test_real_boot_img_oversized_kernel_size_raises_clean_error(boot_bytes: bytes):
    """A kernel_size larger than the file is rejected, not read out of bounds."""
    _require_file(BOOT_IMG)
    buf = bytearray(boot_bytes[: REAL_KERNEL_OFFSET + 4096])
    struct.pack_into("<I", buf, 0x08, len(boot_bytes) * 4)
    with pytest.raises(Exception) as excinfo:
        parse_image(bytes(buf), 4)
    assert_clean_error(excinfo.value, context="oversized-kernel-size real boot.img")


# --------------------------------------------------------------------------
# vendor_boot.img -- the non-standard variant.
#
# These are used as opaque round-trip targets only. They are NOT asserted
# against the AOSP VNDRBOOT field layout, because they provably do not follow
# it: the nominal cmdline_size/name_size/header_size offsets hold ASCII cmdline
# text rather than little-endian integers. See the module docstring.
# --------------------------------------------------------------------------


def _is_nonstandard_vendor(data: bytes) -> bool:
    """Detect the non-standard vendor_boot layout.

    True when the bytes at the nominal cmdline_size offset (0x1C) are printable
    ASCII, which is the signature of cmdline text having been written over the
    header's size fields.
    """
    region = data[VENDOR_NONSTANDARD_CMDLINE_OFFSET:0x100]
    return bool(region) and all(0x20 <= b < 0x7F or b == 0 for b in region if b)


@pytest.mark.parametrize(
    "fixture",
    [VENDOR_BOOT_IMG, VENDOR_BOOT_ALT_IMG],
    ids=["vendor_boot", "vendor_boot_alt"],
)
def test_vendor_boot_is_v4_vnrdboot(fixture: Path) -> None:
    """The vendor_boot fixtures are 64 MiB VNDRBOOT v4 images."""
    _require_file(fixture)
    data = _read(fixture)
    assert len(data) == REAL_BOOT_SIZE, (
        f"{fixture.name} is {len(data)} bytes, expected {REAL_BOOT_SIZE}"
    )
    assert data[0:8] == VENDOR_MAGIC, (
        f"{fixture.name} magic is {data[0:8]!r}, expected {VENDOR_MAGIC!r}"
    )
    assert struct.unpack_from("<I", data, 0x08)[0] == 4, (
        f"{fixture.name} header_version at 0x08 is "
        f"{struct.unpack_from('<I', data, 0x08)[0]}, expected 4"
    )
    # page_size at 0x0C is one of the few fields this variant does populate.
    assert struct.unpack_from("<I", data, 0x0C)[0] == REAL_PAGE_SIZE, (
        f"{fixture.name} page_size at 0x0C is "
        f"{struct.unpack_from('<I', data, 0x0C)[0]}, expected {REAL_PAGE_SIZE}"
    )


@pytest.mark.parametrize(
    "fixture",
    [VENDOR_BOOT_IMG, VENDOR_BOOT_ALT_IMG],
    ids=["vendor_boot", "vendor_boot_alt"],
)
def test_vendor_boot_parses_without_crashing(fixture: Path) -> None:
    """The parser must not crash on a real vendor_boot image.

    This is the contract for the non-standard variant: parse it, extract a
    vendor ramdisk, and survive. No field values are asserted, because this
    layout does not populate the AOSP fields the way AOSP defines them.
    """
    _require_file(fixture)
    data = _read(fixture)

    image = parse_vendor(data)
    ramdisk = field(image, "vendor_ramdisk", "ramdisk", "ramdisk_data")
    assert ramdisk is not None, "vendor ramdisk came back as None"
    assert len(ramdisk) > 0, "vendor ramdisk extracted as empty"

    # Opaque round trip: the bytes must survive a pack/parse cycle intact.
    repacked = parse_vendor(pack_vendor(image))
    repacked_ramdisk = field(repacked, "vendor_ramdisk", "ramdisk", "ramdisk_data")
    assert bytes(repacked_ramdisk) == bytes(ramdisk), (
        f"{fixture.name}: vendor ramdisk changed across a pack/parse cycle; "
        f"first difference at byte "
        f"{_first_difference(bytes(repacked_ramdisk), bytes(ramdisk))}"
    )


@pytest.mark.parametrize(
    "fixture",
    [VENDOR_BOOT_IMG, VENDOR_BOOT_ALT_IMG],
    ids=["vendor_boot", "vendor_boot_alt"],
)
def test_vendor_boot_nonstandard_layout_is_detected(fixture: Path) -> None:
    """Document the non-standard layout rather than assert the AOSP one.

    If a future fixture *does* follow AOSP, this test tells us the variant
    changed so the module docstring can be corrected. It never asserts the
    AOSP field values, which these images provably do not carry.
    """
    _require_file(fixture)
    data = _read(fixture)

    assert _is_nonstandard_vendor(data), (
        f"{fixture.name} no longer has the non-standard layout: the bytes at "
        f"0x{VENDOR_NONSTANDARD_CMDLINE_OFFSET:x} are no longer ASCII cmdline "
        f"text (got {data[VENDOR_NONSTANDARD_CMDLINE_OFFSET : VENDOR_NONSTANDARD_CMDLINE_OFFSET + 32]!r}). "
        f"Update this module's docstring and consider adding real AOSP-layout "
        f"assertions."
    )
    # The nominal size fields really are cmdline text, not integers.
    for offset, name in (
        (0x1C, "cmdline_size"),
        (0x24, "name_size"),
        (0x28, "header_size"),
    ):
        value = struct.unpack_from("<I", data, offset)[0]
        assert value > 0xFFFF, (
            f"{fixture.name}: expected the nominal {name} field at {offset:#x} to "
            f"hold cmdline text (a nonsensical u32), got {value}"
        )
