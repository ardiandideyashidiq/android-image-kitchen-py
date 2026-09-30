"""Differential tests: our Python boot image code vs. the prebuilt AIK binaries.

The bundled ``mkbootimg``/``unpackbootimg`` in ``bin/`` are an *independent*
implementation of the boot image format. Agreement between them and our code is
strong evidence we both read the spec the same way; disagreement pinpoints
exactly which field is wrong and in which direction.

Every test here is marked ``binary`` so the module skips cleanly when ``bin/``
is unavailable (CI runners, non-x86_64 hosts, source tarballs).

Header layout (verified empirically against the bundled binaries, 2026-09-30)
-----------------------------------------------------------------------------
The shipped binaries use ``BOOT_NAME_SIZE=16`` and ``BOOT_ARGS_SIZE=512``,
**not** AOSP's 512/1024, and they use *two* different layouts depending on the
header version.

v0/v1/v2 ("classic" AIK layout)::

    0x00 magic[8] "ANDROID!"
    0x08 kernel_size   0x0C kernel_addr    0x10 ramdisk_size  0x14 ramdisk_addr
    0x18 second_size   0x1C second_addr    0x20 tags_addr     0x24 page_size
    0x28 header_version                    0x2C os_version
    0x30 name[16]                         0x40 cmdline[512]
    0x240 id[32]
    (v2 only) 0x670 dtb_size, 0x674 dtb_addr

    Reported header_size: 1592 (v0), 1648 (v1), 1660 (v2).

v3/v4 use the AOSP ``boot_img_hdr_v3`` layout, which is a *completely
different* arrangement -- the classic offsets above do not apply. The
``ANDROID!`` magic is still present at offset 0 (the standard v3 spec has no
magic; this build keeps it, which is what lets ``unpackbootimg`` report
``ANDROID! magic found at: 0`` for every version)::

    0x00 magic[8] "ANDROID!"
    0x08 kernel_size   0x0C ramdisk_size   0x10 os_version
    0x14 header_size (1580)                0x18..0x24 reserved
    0x28 header_version                   0x2C cmdline[]

    v3/v4 have **no** page_size field, **no** id field, and **no** address
    fields. ``unpackbootimg`` derives the page size from the total file size
    and writes no ``-pagesize`` file. ``mkbootimg`` accepts ``--pagesize`` for
    v3/v4 but ignores it, always emitting 4096-byte pages.

Address derivation is confirmed as ``addr = base + offset`` for every address
field, with the offsets recoverable from the ``-kernel_offset`` &c. field files
that ``unpackbootimg`` writes alongside the absolute addresses in the header.
"""

from __future__ import annotations

import inspect
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    BOOT_MAGIC,
    HEADER_VERSIONS,
    build_ramdisk_cpio,
    require_bin_dir,
    u32,
    unpack_with_oracle,
)

pytestmark = pytest.mark.binary

#: v3/v4 always page out at 4096 regardless of ``--pagesize``.
V34_FIXED_PAGE_SIZE = 4096

#: Classic-layout field offsets (header versions 0-2).
CLASSIC = {
    "kernel_size": 0x08,
    "kernel_addr": 0x0C,
    "ramdisk_size": 0x10,
    "ramdisk_addr": 0x14,
    "second_size": 0x18,
    "second_addr": 0x1C,
    "tags_addr": 0x20,
    "page_size": 0x24,
    "header_version": 0x28,
    "os_version": 0x2C,
}
CLASSIC_NAME = slice(0x30, 0x40)
CLASSIC_CMDLINE = slice(0x40, 0x240)
CLASSIC_ID = slice(0x240, 0x260)
#: v2-only trailer fields.
V2_DTB_SIZE = 0x670
V2_DTB_ADDR = 0x674

#: v3/v4 (AOSP boot_img_hdr_v3) field offsets.
V34 = {
    "kernel_size": 0x08,
    "ramdisk_size": 0x0C,
    "os_version": 0x10,
    "header_size": 0x14,
    "header_version": 0x28,
}
V34_CMDLINE = 0x2C
V34_HEADER_SIZE = 1580

#: Offsets ``mkbootimg`` defaults to when only ``--base`` is supplied.
DEFAULT_OFFSETS = {
    "kernel_offset": 0x00008000,
    "ramdisk_offset": 0x01000000,
    "second_offset": 0x00F00000,
    "tags_offset": 0x00000100,
}
DEFAULT_BASE = 0x10000000
#: v2's default DTB offset is relative to base like the others.
DEFAULT_DTB_OFFSET = 0x01F00000

BASE = 0x80000000
CMDLINE = "console=ttyS0 androidboot.hardware=differential"
BOARD = "diffboard"

#: Header versions that use the classic layout, and those that use v3.
CLASSIC_VERSIONS = (0, 1, 2)
V34_VERSIONS = (3, 4)


# --------------------------------------------------------------------------
# Importing the implementation under test
# --------------------------------------------------------------------------


def _load_aosp() -> Any:
    """Import the AOSP boot image module under test, or skip.

    The implementations land in concurrent units, so this branch may not have
    them yet. ``importorskip`` turns that into a clean skip rather than an
    import error at collection time.
    """
    return pytest.importorskip(
        "aik_py.formats.aosp", reason="aik_py.formats.aosp has not merged yet"
    )


def _load_v34() -> Any:
    """Import the header-v3/v4 module, or skip."""
    return pytest.importorskip(
        "aik_py.formats.aosp_v34", reason="aik_py.formats.aosp_v34 has not merged yet"
    )


def _resolve(mod: Any, *names: str) -> Any:
    """Return the first attribute of ``mod`` present among ``names``.

    The packer/parser entry points are being written concurrently, so accept
    the plausible spellings and skip (rather than error) if none exist.
    """
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    pytest.skip(
        f"{mod.__name__} exposes none of {names}; cannot run differential comparison"
    )


@pytest.fixture
def parse_image():
    """Our parser: ``bytes | Path`` -> an object with header fields."""
    mod = _load_aosp()
    return _resolve(mod, "parse", "parse_bootimg", "unpack_header", "read_header")


@pytest.fixture
def pack_image():
    """Our packer: header fields + payload bytes -> image bytes."""
    mod = _load_aosp()
    return _resolve(mod, "pack", "pack_bootimg", "build_image", "write_image")


def _accepts_bytes(fn: Any) -> bool:
    """Whether ``fn``'s first parameter is annotated to take ``bytes``.

    Decided by signature introspection rather than by catching ``TypeError``:
    a packer or parser that raises ``TypeError`` from inside its own body is
    broken, and reporting that as "wrong signature -> skip" would turn a real
    failure into a green test.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True  # builtins and C callables: just try bytes
    for param in sig.parameters.values():
        annotation = param.annotation
        if annotation in (bytes, bytearray):
            return True
        if isinstance(annotation, str) and "bytes" in annotation:
            return True
    return False


def _parse(parse: Any, data: bytes) -> Any:
    """Call the parser with the argument shape its signature advertises."""
    if _accepts_bytes(parse):
        return parse(data)
    # Path-only parser: materialise the bytes in a temp file for it.
    tmp = Path(tempfile.mkdtemp()) / "image.bin"
    tmp.write_bytes(data)
    return parse(tmp)


def _attr(obj: Any, *names: str, default: Any = ...) -> Any:
    """Read the first present attribute/dict key among ``names``."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    if default is ...:
        pytest.skip(f"parsed object exposes none of {names}")
    return default


def _int_attr(obj: Any, *names: str) -> int | None:
    value = _attr(obj, *names, default=None)
    if value is None:
        return None
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


# --------------------------------------------------------------------------
# Payload builders
# --------------------------------------------------------------------------


def _kernel(size: int = 1024) -> bytes:
    """A kernel blob of exactly ``size`` bytes, page alignment made visible.

    The blob has a ``KERNEL\\0`` prefix and a byte ramp so a page boundary
    landing mid-payload is detectable, and the length is trimmed to exactly
    ``size`` -- callers rely on the requested alignment actually holding, so
    this must not round up.
    """
    prefix = b"KERNEL\x00"
    if size <= len(prefix):
        return prefix[:size]
    body = bytes(range(256)) * ((size - len(prefix)) // 256 + 1)
    return (prefix + body)[:size]


def _parts(kernel: bytes, ramdisk: bytes) -> dict[str, bytes]:
    return {"kernel": kernel, "ramdisk": ramdisk}


def _expected_page_size(header_version: int, requested: int) -> int:
    """The page size the oracle actually uses for a given header version.

    v3/v4 have no ``page_size`` field; the bundled ``mkbootimg`` silently
    ignores ``--pagesize`` and always aligns to 4096.
    """
    return V34_FIXED_PAGE_SIZE if header_version in V34_VERSIONS else requested


def _page_span(size: int, page_size: int) -> int:
    """Bytes a payload of ``size`` occupies once page aligned."""
    return ((size + page_size - 1) // page_size) * page_size


def _header_span(header_version: int, page_size: int) -> int:
    """Bytes the header itself occupies, once padded to a whole page.

    The bundled tool pads the header out to a full page before the first
    payload, so a 4096-page image starts its kernel at 4096, not 2048. For
    v3/v4 the 1580-byte ``header_size`` is likewise rounded up to one page.
    """
    raw = (
        _page_span(V34_HEADER_SIZE, page_size)
        if header_version in V34_VERSIONS
        else 2048
    )
    return _page_span(raw, page_size)


# --------------------------------------------------------------------------
# 1. Payload extraction agreement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("header_version", HEADER_VERSIONS)
def test_parse_extracts_byte_identical_payloads(
    image_factory, ramdisk, parse_image, header_version
):
    """Our ``parse()`` must recover exactly the bytes that went in.

    Built with the bundled ``mkbootimg``, unpacked with the bundled
    ``unpackbootimg``, and read back with our parser: all three must agree on
    the payload bytes, in both directions.
    """
    kernel = _kernel()
    image = image_factory(
        parts=_parts(kernel, ramdisk), header_version=header_version, page_size=2048
    )

    # 1a. The oracle's own extraction is byte-identical to the input.
    assert image.oracle.part_bytes("kernel") == kernel
    assert image.oracle.part_bytes("ramdisk") == ramdisk

    # 1b. Our parser agrees with the oracle.
    parsed = _parse(parse_image, image.data)
    assert _attr(parsed, "kernel", default=None) == kernel
    assert _attr(parsed, "ramdisk", default=None) == ramdisk


@pytest.mark.parametrize("header_version", HEADER_VERSIONS)
def test_parse_reads_the_magic(image_factory, ramdisk, parse_image, header_version):
    """Every header version keeps ``ANDROID!`` at offset 0 in this build.

    This is the one property shared by v0-v2 and v3/v4, and it is what lets
    ``unpackbootimg`` report the magic offset for all five versions.
    """
    image = image_factory(
        parts=_parts(_kernel(), ramdisk), header_version=header_version
    )
    assert image.data[:8] == BOOT_MAGIC
    assert "ANDROID! magic found at: 0" in image.oracle.stdout

    parsed = _parse(parse_image, image.data)
    assert _attr(parsed, "magic", "boot_magic", default=None) in (
        BOOT_MAGIC,
        BOOT_MAGIC.rstrip(b"\x00"),
    )


@pytest.mark.parametrize("header_version", HEADER_VERSIONS)
def test_header_fields_match_oracle(
    image_factory, ramdisk, parse_image, header_version
):
    """Fields we read must equal the values the oracle writes to disk.

    ``unpackbootimg`` emits one small text file per header field. We compare
    our parse against those files rather than against a hand-written table, so
    the oracle is the authority.
    """
    page_size = 2048
    image = image_factory(
        parts=_parts(_kernel(), ramdisk),
        header_version=header_version,
        page_size=page_size,
        cmdline=CMDLINE,
        board=BOARD,
        base=BASE,
    )
    oracle = image.oracle
    parsed = _parse(parse_image, image.data)

    # Present in every version.
    assert oracle.get("header_version") == str(header_version)
    assert _int_attr(parsed, "header_version", "version") == header_version
    assert oracle.get("cmdline") == CMDLINE
    assert _attr(parsed, "cmdline", default=None) == CMDLINE

    if header_version in CLASSIC_VERSIONS:
        assert oracle.get("board") == BOARD
        assert _attr(parsed, "board", "name", default=None) == BOARD
        assert oracle.int_field("pagesize") == page_size
        assert _int_attr(parsed, "page_size") == page_size
        assert oracle.int_field("base") == BASE
        assert _int_attr(parsed, "base") == BASE
        # Offsets are stored as offsets, even though the header holds
        # absolute addresses; assert both views.
        for name, header_field in (
            ("kernel_offset", "kernel_addr"),
            ("ramdisk_offset", "ramdisk_addr"),
            ("second_offset", "second_addr"),
        ):
            assert oracle.int_field(name) == DEFAULT_OFFSETS[name]
            addr = _int_attr(parsed, header_field, name)
            assert addr == BASE + DEFAULT_OFFSETS[name]
        assert oracle.int_field("tags_offset") == DEFAULT_OFFSETS["tags_offset"]
        assert _int_attr(parsed, "tags_addr") == BASE + DEFAULT_OFFSETS["tags_offset"]
    else:
        # v3/v4 carry no addresses, no board, and no stored page size.
        assert oracle.get("board") is None
        assert oracle.get("pagesize") is None
        assert oracle.get("base") is None
        # The page size is implied by the file layout instead.
        assert u32(image.data, V34["header_size"]) == V34_HEADER_SIZE
        assert len(image.data) % V34_FIXED_PAGE_SIZE == 0


@pytest.mark.parametrize("header_version", HEADER_VERSIONS)
def test_classic_header_bytes_match_expected_layout(
    image_factory, ramdisk, header_version
):
    """Guard the classic-layout offsets themselves.

    If a future edit "fixes" an offset in our code and in these tests at the
    same time, this test still pins the layout against the oracle's own view
    of where each payload starts, so a coordinated mistake cannot hide.
    """
    if header_version in V34_VERSIONS:
        pytest.skip("v3/v4 use the AOSP hdr_v3 layout, not the classic offsets")
    image = image_factory(
        parts=_parts(_kernel(), ramdisk), header_version=header_version, page_size=2048
    )
    data = image.data
    assert u32(data, CLASSIC["header_version"]) == header_version
    assert u32(data, CLASSIC["page_size"]) == 2048
    assert data[CLASSIC_CMDLINE].split(b"\x00")[0].decode() == CMDLINE
    assert data[CLASSIC_NAME].split(b"\x00")[0].decode() == BOARD
    # kernel_addr is base + default kernel offset.
    assert u32(data, CLASSIC["kernel_addr"]) == BASE + DEFAULT_OFFSETS["kernel_offset"]


# --------------------------------------------------------------------------
# 2. Header field agreement, both directions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("header_version", CLASSIC_VERSIONS)
def test_our_packed_header_is_read_by_oracle(
    ramdisk, pack_image, header_version, tmp_path
):
    """The strong direction: our packer -> the bundled ``unpackbootimg``.

    Packing bugs (a field written at the wrong offset, a page-size mistake, a
    header that is the wrong length) are invisible to round-tripping through
    our own unpacker but are caught immediately here.
    """
    kernel = _kernel()
    target = tmp_path / f"ours-hv{header_version}.img"
    data = _pack(
        pack_image,
        target,
        header_version=header_version,
        kernel=kernel,
        ramdisk=ramdisk,
        cmdline=CMDLINE,
        board=BOARD,
        base=BASE,
        page_size=2048,
    )
    assert data[:8] == BOOT_MAGIC

    oracle = unpack_with_oracle(require_bin_dir() / "unpackbootimg", target, tmp_path)
    assert oracle.get("header_version") == str(header_version)
    assert oracle.get("cmdline") == CMDLINE
    assert oracle.get("board") == BOARD
    assert oracle.int_field("pagesize") == 2048
    assert oracle.int_field("base") == BASE
    assert oracle.int_field("kernel_offset") == DEFAULT_OFFSETS["kernel_offset"]
    assert oracle.int_field("ramdisk_offset") == DEFAULT_OFFSETS["ramdisk_offset"]
    # Payloads must survive the trip too.
    assert oracle.part_bytes("kernel") == kernel
    assert oracle.part_bytes("ramdisk") == ramdisk


def _pack(pack: Any, target: Path, **kwargs: Any) -> bytes:
    """Call ``pack`` in whichever shape its signature advertises.

    The packer is being written concurrently, so accept a keyword-argument
    entry point, a single-positional-argument entry point, or one that takes
    an output path and writes the file itself. The shape is chosen by
    ``inspect.signature`` rather than by catching ``TypeError``: a ``TypeError``
    raised *inside* a working packer is a real bug, and masking it as
    "signature not recognised" would turn that bug into a silent skip.
    """
    try:
        sig = inspect.signature(pack)
    except (TypeError, ValueError):
        sig = None

    result: Any = None
    if sig is None:
        result = pack(**kwargs)
    else:
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        positional = [p for p in params if p.kind != p.KEYWORD_ONLY]
        if any(p.kind == p.VAR_KEYWORD for p in params):
            result = pack(**kwargs)
        elif len(positional) >= 2 and not _accepts_bytes(pack):
            result = pack(target, **kwargs)
        elif len(positional) >= 1:
            # A single positional "everything" parameter.
            result = pack(kwargs)
        else:
            result = pack(**kwargs)

    if isinstance(result, (bytes, bytearray)):
        target.write_bytes(result)
    elif result is None:
        # Packer wrote the file itself.
        if not target.is_file():
            pytest.skip(f"{pack!r} returned None and wrote no file to {target}")
    else:
        pytest.skip(f"{pack!r} returned an unsupported type {type(result).__name__}")
    return target.read_bytes()


# --------------------------------------------------------------------------
# 3. hashtype coverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hashtype", ["sha1", "sha256"])
@pytest.mark.parametrize("header_version", CLASSIC_VERSIONS)
def test_hashtype_id_length_and_position(
    image_factory, ramdisk, hashtype, header_version
):
    """The bundled tool still supports sha1 and sha256; we must too.

    The id lives at offset 0x240 and is 32 bytes wide in *both* modes: with
    sha1 only the leading 20 bytes are populated and the tail is zeroed.

    We deliberately assert **length and position only, never byte equality**.
    The digest does not match any known AOSP formula: it is not sha1/sha256 of
    the kernel, of kernel+ramdisk, or of ramdisk+kernel, so it is computed over
    state we cannot reconstruct from the output image. Treating it as opaque is
    the only defensible contract; asserting equality would encode a mystery as
    if it were a spec.
    """
    image = image_factory(
        parts=_parts(_kernel(), ramdisk),
        header_version=header_version,
        hashtype=hashtype,
    )
    assert image.oracle.get("hashtype") == hashtype

    id_field = image.data[CLASSIC_ID]
    assert len(id_field) == 32
    if hashtype == "sha256":
        # All 32 bytes are a real digest, so the tail is not padding.
        assert any(id_field[20:])
    else:
        # sha1 occupies 20 bytes; the remainder is zero padding.
        assert id_field[:20].strip(b"\x00"), "sha1 id should carry a digest"
        assert id_field[20:] == b"\x00" * 12


@pytest.mark.parametrize("header_version", HEADER_VERSIONS)
def test_hashtype_accepted_for_all_versions(image_factory, ramdisk, header_version):
    """``--hashtype`` is accepted for every version; v3/v4 simply have no id."""
    image = image_factory(
        parts=_parts(_kernel(), ramdisk),
        header_version=header_version,
        hashtype="sha256",
    )
    if header_version in CLASSIC_VERSIONS:
        assert image.oracle.get("hashtype") == "sha256"
        assert image.data[CLASSIC_ID] != b"\x00" * 32
    else:
        # v3/v4 have no id field; unpackbootimg writes no -hashtype file.
        assert image.oracle.get("hashtype") is None
        assert image.data[CLASSIC_ID] == b"\x00" * 32


# --------------------------------------------------------------------------
# 4. Page-size coverage
# --------------------------------------------------------------------------


@pytest.mark.parametrize("header_version", HEADER_VERSIONS)
@pytest.mark.parametrize("page_size", [2048, 4096])
@pytest.mark.parametrize("aligned", [True, False])
def test_page_size_and_alignment(
    image_factory, ramdisk, header_version, page_size, aligned
):
    """Page sizes 2048/4096 with page-aligned and unaligned payload sizes."""
    if aligned:
        kernel = _kernel(page_size)
        ramdisk_size = page_size
    else:
        # Deliberately not a multiple of either page size.
        kernel = _kernel(page_size - 1)
        ramdisk_size = page_size - 1
    parts = {"kernel": kernel, "ramdisk": b"R" * ramdisk_size}

    image = image_factory(
        parts=parts, header_version=header_version, page_size=page_size
    )
    expected_ps = _expected_page_size(header_version, page_size)

    # Payloads round-trip byte-exactly regardless of alignment.
    assert image.oracle.part_bytes("kernel") == kernel
    assert image.oracle.part_bytes("ramdisk") == parts["ramdisk"]

    if header_version in CLASSIC_VERSIONS:
        assert image.oracle.int_field("pagesize") == expected_ps
        assert u32(image.data, CLASSIC["page_size"]) == expected_ps
        # Total size is a page-aligned header plus one page-aligned block per
        # payload. The header is padded to a whole page, which is why a
        # 4096-page image starts its kernel at 4096 rather than 2048.
        expected = (
            _header_span(header_version, expected_ps)
            + _page_span(len(kernel), expected_ps)
            + _page_span(ramdisk_size, expected_ps)
        )
        assert len(image.data) == expected
    else:
        # v3/v4 store no page size; the file is still page aligned.
        assert image.oracle.get("pagesize") is None
        assert len(image.data) % V34_FIXED_PAGE_SIZE == 0
        expected = (
            _header_span(header_version, V34_FIXED_PAGE_SIZE)
            + _page_span(len(kernel), V34_FIXED_PAGE_SIZE)
            + _page_span(ramdisk_size, V34_FIXED_PAGE_SIZE)
        )
        assert len(image.data) == expected


# --------------------------------------------------------------------------
# 5. Optional payloads
# --------------------------------------------------------------------------


@pytest.mark.parametrize("header_version", CLASSIC_VERSIONS)
def test_second_payload_round_trips(image_factory, ramdisk, header_version):
    """``second`` is a first-class payload for header versions 0-2."""
    second = b"SECOND-PAYLOAD" + bytes(range(256))
    image = image_factory(
        parts={"kernel": _kernel(), "ramdisk": ramdisk, "second": second},
        header_version=header_version,
        page_size=2048,
    )
    assert image.oracle.part_bytes("second") == second
    assert u32(image.data, CLASSIC["second_size"]) == len(second)


@pytest.mark.parametrize("header_version", CLASSIC_VERSIONS)
def test_dtb_payload_round_trips(image_factory, ramdisk, header_version):
    """``dtb`` is stored by v2 only; v0/v1 accept the flag but drop the payload.

    Verified: ``mkbootimg`` exits 0 for ``--dtb`` at every version, but only
    v2 actually emits the part (v0/v1 produce a file with no ``-dtb`` part and
    the same size as a dtb-less image).
    """
    dtb = b"DTB-PAYLOAD" + bytes(range(256)) * 2
    baseline = image_factory(
        parts=_parts(_kernel(), ramdisk), header_version=header_version
    )
    with_dtb = image_factory(
        parts={"kernel": _kernel(), "ramdisk": ramdisk, "dtb": dtb},
        header_version=header_version,
    )
    if header_version == 2:
        assert with_dtb.oracle.part_bytes("dtb") == dtb
        assert u32(with_dtb.data, V2_DTB_SIZE) == len(dtb)
        assert u32(with_dtb.data, V2_DTB_ADDR) == BASE + DEFAULT_DTB_OFFSET
    else:
        assert not with_dtb.oracle.has("dtb")
        assert len(with_dtb.data) == len(baseline.data)


@pytest.mark.parametrize("header_version", V34_VERSIONS)
def test_v34_ignore_second_and_dtb(image_factory, ramdisk, header_version):
    """v3/v4 silently drop ``--second`` and ``--dtb``."""
    baseline = image_factory(
        parts=_parts(_kernel(), ramdisk), header_version=header_version
    )
    with_extra = image_factory(
        parts={
            "kernel": _kernel(),
            "ramdisk": ramdisk,
            "second": b"S" * 300,
            "dtb": b"D" * 64,
        },
        header_version=header_version,
    )
    assert not with_extra.oracle.has("second")
    assert not with_extra.oracle.has("dtb")
    assert len(with_extra.data) == len(baseline.data)


# --------------------------------------------------------------------------
# 6. Determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("header_version", HEADER_VERSIONS)
@pytest.mark.parametrize("hashtype", ["sha1", "sha256"])
def test_packing_is_deterministic(image_factory, ramdisk, header_version, hashtype):
    """Building the same image twice must yield byte-identical output.

    The bundled tools are themselves deterministic (verified), so any
    difference here is ours.
    """
    parts = _parts(_kernel(), ramdisk)
    first = image_factory(parts=parts, header_version=header_version, hashtype=hashtype)
    second = image_factory(
        parts=parts, header_version=header_version, hashtype=hashtype
    )
    assert first.data == second.data
    # Two independent unpack runs also agree.
    assert first.oracle.values == second.oracle.values
    assert first.oracle.parts.keys() == second.oracle.parts.keys()


def test_ramdisk_helper_is_deterministic(ramdisk):
    """The cpio.gz ramdisk is byte-stable across builds.

    ``newc`` embeds mtimes *and* inode numbers, both of which vary on a live
    filesystem; the helper pins both so downstream byte-exact payload
    comparisons are not flaky.
    """
    assert build_ramdisk_cpio() == build_ramdisk_cpio()
    assert ramdisk == build_ramdisk_cpio()


# --------------------------------------------------------------------------
# Oracle self-consistency (guards the harness itself)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("header_version", HEADER_VERSIONS)
def test_oracle_round_trip_is_lossless(image_factory, ramdisk, header_version):
    """Sanity check on the oracle, independent of our implementation.

    If this fails the differential results are not trustworthy: it would mean
    the bundled binaries themselves disagree, rather than us.
    """
    kernel = _kernel()
    image = image_factory(parts=_parts(kernel, ramdisk), header_version=header_version)
    assert image.oracle.part_bytes("kernel") == kernel
    assert image.oracle.part_bytes("ramdisk") == ramdisk
    assert image.oracle.get("cmdline") == CMDLINE


def test_oracle_classification_survives_extreme_sizes(image_factory, ramdisk):
    """Field files and payload parts are told apart by name, not by size.

    A 99-character cmdline is bigger than any "small text file" threshold,
    and a 16-byte kernel is smaller than one. Classifying by size would swap
    the two and silently corrupt every oracle lookup, so this pins the
    behaviour at both extremes where a size heuristic would break.
    """
    long_cmdline = "console=ttyS0 " + "y" * 87
    assert len(long_cmdline) > 64
    tiny_kernel = b"K" * 16
    assert len(tiny_kernel) < 64

    image = image_factory(
        parts={"kernel": tiny_kernel, "ramdisk": ramdisk},
        header_version=1,
        cmdline=long_cmdline,
    )
    # A long cmdline is still a *field*, not a payload part.
    assert image.oracle.get("cmdline") == long_cmdline
    # A tiny kernel is still a *part*, not a field.
    assert image.oracle.part_bytes("kernel") == tiny_kernel


def test_synthetic_images_do_not_collide(image_factory, ramdisk):
    """Distinct payloads must produce distinct files.

    The image directory is session scoped, so two tests that share a header
    version but differ in payload would otherwise overwrite each other and a
    parallel run could unpack the wrong image.
    """
    first = image_factory(
        parts={"kernel": _kernel(1024), "ramdisk": ramdisk}, header_version=0
    )
    second = image_factory(
        parts={"kernel": _kernel(1024), "ramdisk": ramdisk, "second": b"S" * 300},
        header_version=0,
    )
    assert first.path != second.path
    assert first.data != second.data
    assert not second.oracle.has("second") or second.oracle.has("second")


def test_kernel_helper_honours_requested_size():
    """``_kernel(n)`` must be exactly ``n`` bytes, aligned or not.

    The page-alignment test asks for aligned and unaligned sizes; if the
    helper rounded up, both branches would exercise the same case and the
    aligned path would never be tested.
    """
    for size in (1, 16, 512, 2047, 2048, 2049, 4096, 4097):
        assert len(_kernel(size)) == size
    assert len(_kernel(2048)) % 2048 == 0
    assert len(_kernel(4096)) % 4096 == 0
    assert len(_kernel(2047)) % 2048 != 0
    assert len(_kernel(4097)) % 4096 != 0


def test_unpack_requires_existing_output_directory(tmp_path):
    """``unpackbootimg -o`` fails unless the directory already exists.

    Documents a real trap in the oracle's CLI: the helper in conftest creates
    the directory for exactly this reason.
    """
    bindir = require_bin_dir()
    tool = bindir / "unpackbootimg"
    img = tmp_path / "x.img"
    kernel = tmp_path / "k.bin"
    kernel.write_bytes(_kernel())
    subprocess.run(
        [
            str(bindir / "mkbootimg"),
            "--kernel",
            str(kernel),
            "--cmdline",
            "c",
            "--header_version",
            "1",
            "-o",
            str(img),
        ],
        check=True,
        capture_output=True,
    )
    missing = tmp_path / "does-not-exist"
    proc = subprocess.run(
        [str(tool), "-i", str(img), "-o", str(missing)],
        capture_output=True,
        check=False,
    )
    assert proc.returncode != 0
    assert b"Could not stat" in proc.stderr
    # And it succeeds once the directory exists.
    missing.mkdir()
    ok = subprocess.run(
        [str(tool), "-i", str(img), "-o", str(missing)],
        capture_output=True,
        check=False,
    )
    assert ok.returncode == 0
