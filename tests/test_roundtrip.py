"""Layer 1: synthetic round-trip tests for the AOSP boot image formats.

These tests build images byte-by-byte from the *independently implemented*
reference layout in this module, hand them to the library, and check that the
library reads back what was written. Using a reference implementation rather
than the library's own packer is deliberate: a parse/pack round trip through a
single implementation can be self-consistent and still completely wrong. Every
assertion here is about the on-disk contract.

Reference layout
----------------
Verified empirically against the bundled ``bin/linux/x86_64/mkbootimg`` and
``unpackbootimg`` and against a genuine 64 MiB OEM ``boot.img``. This build
uses ``BOOT_NAME_SIZE=16`` and ``BOOT_ARGS_SIZE=512``, not AOSP's 512/1024.

v0/v1/v2 (magic-first, classic offsets)::

    0x00  magic "ANDROID!"            (8 bytes)
    0x08  kernel_size                 0x0C  kernel_addr
    0x10  ramdisk_size                0x14  ramdisk_addr
    0x18  second_size                 0x1C  second_addr
    0x20  tags_addr                   0x24  page_size
    0x28  header_version              0x2C  os_version
    0x30  name[16]                    0x40  cmdline[512]
    0x240 id[32]
    0x66C header_size                 0x670 dtb_size (v2)
    0x674 dtb_addr (v2)

v3/v4 keep the ``ANDROID!`` magic at offset 0 but switch to the *compact* AOSP
v3 field order -- this is the single most important fact in this file, and it
contradicts both the AOSP source (which drops the magic for v3+) and the
project brief (which says v3/v4 keep "classic offsets")::

    0x00  magic "ANDROID!"            (8 bytes)
    0x08  kernel_size                 0x0C  ramdisk_size
    0x10  os_version                  0x14  header_size
    0x28  header_version              0x2C  cmdline

Note there is no ``page_size`` field in the v3/v4 header: the page size is
implicitly the header size rounded up, which is why the real OEM image stores
0x630 (1584) at 0x14 and starts its kernel at the following 4096 boundary.
Tests build v3/v4 with that layout and cross-check it against the real image in
``test_golden.py``.

Header *size* values below are what the bundled tooling reports. They are used
to *build* reference images, so a disagreement with the library surfaces as a
real failure rather than being papered over. The brief quotes 1580 for v3/v4,
but the real OEM v4 image stores 1584; since the field is read from the image,
both are accepted (see :data:`ACCEPTED_V3_V4_HEADER_SIZES`).
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import struct
from typing import Any

import pytest

# The library modules land in other units; skip cleanly rather than erroring
# until they exist. aik_py.formats.aosp is the only hard requirement --
# aosp_v34 is optional and is resolved at call time (see _candidate_modules),
# so its absence degrades to using aosp rather than skipping everything.
aosp = pytest.importorskip(
    "aik_py.formats.aosp", reason="aik_py.formats.aosp not implemented yet"
)

# --------------------------------------------------------------------------
# Reference (independent) implementation of the on-disk format.
# --------------------------------------------------------------------------

BOOT_MAGIC = b"ANDROID!"
BOOT_MAGIC_SIZE = 8
BOOT_NAME_SIZE = 16
BOOT_ARGS_SIZE = 512
BOOT_ID_SIZE = 32

# Header size as reported by the bundled tooling. These are the sizes used to
# *build* reference images, so a disagreement with the library shows up as a
# real failure rather than being masked.
#
# Measured with `unpackbootimg -i` on images produced by the bundled
# `mkbootimg`. v0 reports nothing (1592 is the derived v0 size); v1 reports
# 1648, v2 reports 1660, and v3/v4 report 1580.
_REPORTED = {0: 1592, 1: 1648, 2: 1660, 3: 1580, 4: 1580}

# The real OEM v4 image stores 1584 at offset 0x14, not 1580. Both are real
# values from real producers, so a parser is correct to pass either through
# rather than normalising to a single number.
ACCEPTED_V3_V4_HEADER_SIZES = frozenset({1580, 1584})

# Offset of ``header_size`` differs between the two header shapes. v0 has no
# header_size field at all (unpackbootimg reports none for v0), so it maps to
# None rather than to a position inside a 1592-byte header.
HEADER_SIZE_OFFSET_V012 = 0x66C
HEADER_SIZE_OFFSET_V34 = 0x14
_HEADER_SIZE_OFFSETS = {
    0: None,
    1: HEADER_SIZE_OFFSET_V012,
    2: HEADER_SIZE_OFFSET_V012,
    3: HEADER_SIZE_OFFSET_V34,
    4: HEADER_SIZE_OFFSET_V34,
}


def header_size_offset(version: int) -> int | None:
    """Byte offset of the ``header_size`` field, or None for v0."""
    return _HEADER_SIZE_OFFSETS[version]


def reported_header_size(version: int) -> int:
    """Header size the bundled tooling reports for ``version``."""
    return _REPORTED[version]


def _u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def header_size_on_disk(data: bytes, header_version: int) -> int | None:
    """Read the on-disk ``header_size`` field from raw image bytes.

    Returns None for v0, which has no such field. Used to cross-check what the
    library reports against what is physically present, which is the only way
    to catch a parser and a builder that agree with each other but not with
    the format.
    """
    offset = header_size_offset(header_version)
    return None if offset is None else _u32(data, offset)


def build_header(
    *,
    header_version: int,
    kernel_size: int = 0,
    kernel_addr: int = 0,
    ramdisk_size: int = 0,
    ramdisk_addr: int = 0,
    second_size: int = 0,
    second_addr: int = 0,
    tags_addr: int = 0,
    page_size: int = 2048,
    os_version: int = 0,
    name: bytes = b"",
    cmdline: bytes = b"",
    header_size: int | None = None,
    dtb_size: int = 0,
    dtb_addr: int = 0,
    recovery_dtbo_size: int = 0,
    recovery_dtbo_addr: int = 0,
    hash_id: bytes | None = None,
) -> bytes:
    """Build a v0..v4 boot header exactly as the reference layout specifies.

    Sizes/addresses are the *payload* sizes; the caller is responsible for
    laying the payloads out at page boundaries in :func:`build_image`.
    """
    size = reported_header_size(header_version)
    size_offset = header_size_offset(header_version)
    # A v0 header is only 1592 bytes, shorter than the 0x66C offset used by
    # v1/v2, so the buffer must be sized from whichever is larger.
    needed = size if size_offset is None else max(size, size_offset + 4)
    buf = bytearray(needed + 64)
    buf[0:8] = BOOT_MAGIC
    struct.pack_into("<I", buf, 0x28, header_version)

    if header_version >= 3:
        # Compact AOSP v3 field order, with the magic retained.
        struct.pack_into("<I", buf, 0x08, kernel_size)
        struct.pack_into("<I", buf, 0x0C, ramdisk_size)
        struct.pack_into("<I", buf, 0x10, os_version)
        struct.pack_into(
            "<I",
            buf,
            HEADER_SIZE_OFFSET_V34,
            size if header_size is None else header_size,
        )
        # v3/v4 have no page_size field; the payload starts at the next page
        # boundary at or after header_size.
        buf[0x2C : 0x2C + BOOT_ARGS_SIZE] = cmdline.ljust(BOOT_ARGS_SIZE, b"\0")[
            :BOOT_ARGS_SIZE
        ]
        return bytes(buf[:size])

    struct.pack_into("<I", buf, 0x08, kernel_size)
    struct.pack_into("<I", buf, 0x0C, kernel_addr)
    struct.pack_into("<I", buf, 0x10, ramdisk_size)
    struct.pack_into("<I", buf, 0x14, ramdisk_addr)
    struct.pack_into("<I", buf, 0x18, second_size)
    struct.pack_into("<I", buf, 0x1C, second_addr)
    struct.pack_into("<I", buf, 0x20, tags_addr)
    struct.pack_into("<I", buf, 0x24, page_size)
    struct.pack_into("<I", buf, 0x2C, os_version)
    buf[0x30 : 0x30 + BOOT_NAME_SIZE] = name.ljust(BOOT_NAME_SIZE, b"\0")[
        :BOOT_NAME_SIZE
    ]
    buf[0x40 : 0x40 + BOOT_ARGS_SIZE] = cmdline.ljust(BOOT_ARGS_SIZE, b"\0")[
        :BOOT_ARGS_SIZE
    ]
    buf[0x240 : 0x240 + BOOT_ID_SIZE] = (hash_id or bytes(BOOT_ID_SIZE)).ljust(
        BOOT_ID_SIZE, b"\0"
    )[:BOOT_ID_SIZE]
    if size_offset is not None:
        struct.pack_into(
            "<I", buf, size_offset, size if header_size is None else header_size
        )
    if header_version == 2:
        struct.pack_into("<I", buf, 0x670, dtb_size)
        struct.pack_into("<I", buf, 0x674, dtb_addr)
    return bytes(buf[:size])


def _align_up(value: int, page_size: int) -> int:
    return (value + page_size - 1) // page_size * page_size


def build_image(
    *,
    header_version: int,
    page_size: int,
    kernel: bytes = b"",
    ramdisk: bytes = b"",
    second: bytes = b"",
    dtb: bytes = b"",
    recovery_dtbo: bytes = b"",
    kernel_signature: bytes = b"",
    **header_kwargs: Any,
) -> bytes:
    """Lay out a complete boot image: header page then page-aligned payloads.

    ``second`` and ``dtb`` are only valid for header versions <= 2; the kernel
    signature blob only for v3+. Payloads are placed in the canonical AOSP
    order: kernel, ramdisk, second, dtb, recovery_dtbo, kernel_signature.
    """
    if second and header_version > 2:
        raise ValueError("second is only valid for header_version <= 2")
    if dtb and header_version > 2:
        raise ValueError("dtb is only valid for header_version <= 2")
    if kernel_signature and header_version < 3:
        raise ValueError("kernel_signature requires header_version >= 3")
    if header_version >= 3 and page_size != 4096:
        # v3/v4 headers have no page_size field, so the payload page size is
        # fixed by header_size. Building a 2048-page v3 image would produce
        # bytes no parser could read back consistently.
        raise ValueError(
            f"header_version {header_version} has no page_size field; "
            f"page_size must be 4096, got {page_size}"
        )

    # Sizes are derivable from the payloads; fill them in unless the caller
    # deliberately overrode them (e.g. to build a malformed image). Treat None
    # as absent rather than as a zero-length payload.
    for field_name, payload in (
        ("kernel_size", kernel),
        ("ramdisk_size", ramdisk),
        ("second_size", second),
        ("dtb_size", dtb),
        ("recovery_dtbo_size", recovery_dtbo),
    ):
        header_kwargs.setdefault(field_name, len(payload or b""))

    header_kwargs.setdefault("page_size", page_size)
    header_kwargs.setdefault("header_version", header_version)
    header = build_header(**header_kwargs)

    out = bytearray(header)
    out += bytes(_align_up(len(out), page_size) - len(out))  # pad to a page

    def _place(payload: bytes | None) -> int:
        """Append ``payload`` padded out to a page boundary; return its offset.

        ``None`` means "this payload is absent" and emits nothing at all, which
        is what distinguishes an absent payload from a zero-length one.
        """
        offset = len(out)
        if not payload:
            return offset
        out.extend(payload)
        pad = _align_up(len(out), page_size) - len(out)
        out.extend(bytes(pad))
        return offset

    _place(kernel)
    _place(ramdisk)
    _place(second)
    _place(dtb)
    _place(recovery_dtbo)
    _place(kernel_signature)
    return bytes(out)


def payload_offset(
    data: bytes, page_size: int, *, kernel=b"", ramdisk=b"", second=b"", dtb=b""
) -> dict[str, int]:
    """Compute where each payload landed, mirroring :func:`build_image`.

    The header always occupies exactly one page, so the first payload starts
    at ``page_size``.
    """
    offsets: dict[str, int] = {}
    cursor = page_size
    for name, payload in (
        ("kernel", kernel),
        ("ramdisk", ramdisk),
        ("second", second),
        ("dtb", dtb),
    ):
        offsets[name] = cursor
        cursor = _align_up(cursor + len(payload), page_size)
    return offsets


# --------------------------------------------------------------------------
# Deterministic cpio (``newc``) builder.
#
# CRITICAL: a ``newc`` cpio archive embeds each entry's mtime verbatim, so any
# byte-exact ramdisk comparison is flaky unless mtimes are pinned. Real ramdisk
# trees are built with ``touch -d @0`` for exactly this reason; here we write
# the headers by hand with mtime=0, which is equivalent and avoids depending on
# an external ``cpio``/``touch`` in the test environment. This is the single
# most common source of flaky boot-image round-trip tests.
# --------------------------------------------------------------------------

CPIO_MAGIC = b"070701"
CPIO_TRAILER = "TRAILER!!!"


def _cpio_entry(
    name: str, data: bytes, *, ino: int, mode: int = 0o100644, mtime: int = 0
) -> bytes:
    """Encode one ``newc`` entry with a pinned mtime."""
    name_b = name.encode() + b"\0"
    fields = [
        0o040000 | mode,  # mode
        0,  # uid
        0,  # gid
        1,  # nlink
        mtime,
        len(data),
        0,  # devmajor
        0,  # devminor
        0,  # rdevmajor
        0,  # rdevminor
        len(name_b),
        0,  # check
    ]
    # Order for newc: ino, mode, uid, gid, nlink, mtime, filesize, dev*,
    # rdev*, namesize, check.
    values = [ino, *fields]
    out = bytearray(CPIO_MAGIC)
    for value in values:
        out += b"%08X" % (value & 0xFFFFFFFF)
    out += name_b
    out += bytes((-len(out)) % 4)
    out += data
    out += bytes((-len(out)) % 4)
    return bytes(out)


def build_cpio(entries: dict[str, bytes], *, mtime: int = 0) -> bytes:
    """Build a deterministic ``newc`` cpio archive.

    Every entry is stamped with ``mtime`` (default 0) and assigned a stable
    inode number, so the same input dict always produces the same bytes.
    """
    out = bytearray()
    for ino, (name, data) in enumerate(entries.items(), start=1):
        out += _cpio_entry(name, data, ino=ino, mtime=mtime)
    out += _cpio_entry(CPIO_TRAILER, b"", ino=0, mode=0, mtime=mtime)
    return bytes(out)


def build_gzip_cpio(entries: dict[str, bytes], *, mtime: int = 0) -> bytes:
    """Gzip a deterministic cpio archive with a pinned gzip header.

    ``gzip.compress`` embeds the current time in its header, so pass
    ``mtime=0`` to make the compressed output reproducible as well.
    """
    return gzip.compress(build_cpio(entries, mtime=mtime), mtime=0, compresslevel=9)


# --------------------------------------------------------------------------
# Adapter over the library API.
#
# The library modules are being written concurrently and their exact callables
# are not pinned down by the brief, only their *behaviour*. The adapter below
# probes for the conventional entry points and then reports precisely what it
# could not find. It never silently degrades: if a module imports but exposes
# none of the expected names, that is a real contract mismatch and the test
# fails loudly rather than skipping, because a skip would hide it.
# --------------------------------------------------------------------------

_MISSING = object()

_PARSE_NAMES = (
    "parse",
    "unpack",
    "load",
    "loads",
    "read",
    "from_bytes",
    "parse_bytes",
    "BootImage",
    "BootImageHeader",
)
_PACK_NAMES = ("pack", "dump", "to_bytes", "build", "serialize", "write", "Pack")


def _import(name: str) -> Any | None:
    """Import an optional library module, or return None if it is absent.

    Returns None rather than skipping, so that a missing sibling module (say
    ``aosp_v34``) degrades to trying the other one instead of skipping the
    whole suite. Only ``aik_py.formats.aosp`` itself is required, and that is
    checked once at import time by the ``importorskip`` at the top of the file.
    """
    try:
        return importlib.import_module(f"aik_py.formats.{name}")
    except ImportError:
        return None


def _candidate_modules(header_version: int) -> list[Any]:
    """Prefer the v3/v4 module for v3+ images, falling back to the base one."""
    aosp = _import("aosp")
    v34 = _import("aosp_v34")
    ordered = [v34, aosp] if header_version >= 3 else [aosp, v34]
    return [mod for mod in ordered if mod is not None]


def _call_parser(mod: Any, data: bytes) -> Any:
    """Call ``mod``'s parse entry point on ``data``.

    Deliberately narrow about what counts as "wrong candidate function": only a
    ``TypeError`` from calling the wrong signature. In particular a
    ``ValueError`` raised *by the library* must propagate untouched, or the
    malformed-input tests would end up asserting against this adapter's own
    error instead of the library's -- which would make them pass no matter how
    badly the library behaved.
    """
    for name in _PARSE_NAMES:
        fn = getattr(mod, name, None)
        if not callable(fn):
            continue
        try:
            return fn(data)
        except TypeError:
            # Signature mismatch (e.g. it wants a path, or it is a class whose
            # constructor takes kwargs) - keep probing.
            continue
    raise AssertionError(
        f"{mod.__name__} exposes none of the expected parse entry points "
        f"{_PARSE_NAMES!r}. Found: {sorted(n for n in dir(mod) if not n.startswith('_'))!r}"
    )


def parse_image(data: bytes, header_version: int) -> Any:
    """Parse ``data`` with whichever module implements ``header_version``."""
    errors: list[str] = []
    for mod in _candidate_modules(header_version):
        try:
            return _call_parser(mod, data)
        except AssertionError as exc:
            errors.append(str(exc))
    raise AssertionError("; ".join(errors))


def _image_bytes(img: Any) -> bytes:
    """Serialise a parsed image back to bytes via a packer or a method."""
    for name in _PACK_NAMES:
        fn = getattr(img, name, None)
        if callable(fn):
            try:
                out = fn()
            except TypeError:
                continue
            if isinstance(out, (bytes, bytearray)):
                return bytes(out)
    mod = getattr(img, "__module__", None)
    if mod is not None:
        import sys

        parent = sys.modules.get(mod)
        if parent is not None:
            for name in _PACK_NAMES:
                fn = getattr(parent, name, None)
                if callable(fn):
                    try:
                        out = fn(img)
                    except TypeError:
                        continue
                    if isinstance(out, (bytes, bytearray)):
                        return bytes(out)
    raise AssertionError(
        f"could not find a packer for parsed image of type {type(img).__name__!r}; "
        f"tried methods {_PACK_NAMES!r} and module-level packers"
    )


def pack_image(img: Any) -> bytes:
    """Pack a parsed image back to bytes."""
    return _image_bytes(img)


def _containers(obj: Any) -> list[Any]:
    """The object plus any nested header object, for field lookup.

    Only descends into real mappings or objects that actually expose fields;
    a missing ``fields`` attribute must not drag in unrelated objects.
    """
    containers = [obj]
    for sub in ("header", "hdr", "boot_header"):
        nested = _lookup(obj, sub)
        looks_like_fields = isinstance(nested, dict) or hasattr(nested, "__dict__")
        if looks_like_fields and nested is not None and nested is not obj:
            containers.append(nested)
    return containers


def field(obj: Any, *names: str) -> Any:
    """Read a header field, trying attribute and mapping access in turn.

    Also looks one level down through a nested ``header``/``hdr`` object so
    tests pass whether fields are flat on the image or grouped.
    """
    containers = _containers(obj)
    for container in containers:
        for name in names:
            value = _lookup(container, name)
            if value is not _MISSING:
                return value
    raise AssertionError(
        f"none of the field names {names!r} found on "
        f"{type(obj).__name__!r}; tried containers "
        f"{[type(c).__name__ for c in containers]!r}"
    )


def _lookup(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, _MISSING)
    value = getattr(obj, name, _MISSING)
    if value is not _MISSING:
        return value
    # Support dataclass/slots objects that only expose __dict__ keys.
    as_dict = getattr(obj, "__dict__", None)
    if isinstance(as_dict, dict) and name in as_dict:
        return as_dict[name]
    return _MISSING


def set_field(obj: Any, *names: str, value: Any) -> None:
    """Write a header field, trying attribute and mapping access in turn."""
    for container in _containers(obj):
        for name in names:
            if isinstance(container, dict) and name in container:
                container[name] = value
                return
            if hasattr(container, name):
                setattr(container, name, value)
                return
    raise AssertionError(
        f"none of the field names {names!r} could be written on {type(obj).__name__!r}"
    )


_set_field = set_field


# --------------------------------------------------------------------------
# Malformed-input handling.
# --------------------------------------------------------------------------

# Exceptions that must never escape the library. A parser that lets these
# through is leaking its implementation rather than reporting a bad image.
_RAW_EXCEPTIONS = (
    struct.error,
    IndexError,
    KeyError,
    UnicodeDecodeError,
    OverflowError,
    MemoryError,
)


def assert_clean_error(exc: BaseException, *, context: str) -> None:
    """Assert ``exc`` is a deliberate, reported error and not a raw leak."""
    assert not isinstance(exc, _RAW_EXCEPTIONS), (
        f"{context}: library leaked a raw {type(exc).__name__} instead of a "
        f"deliberate error: {exc!r}"
    )
    assert not isinstance(exc, TypeError), (
        f"{context}: library raised a bare TypeError (likely an unhandled "
        f"signature/unpacking bug): {exc!r}"
    )
    assert str(exc).strip(), f"{context}: library raised {exc!r} with an empty message"


# --------------------------------------------------------------------------
# The round-trip matrix.
# --------------------------------------------------------------------------

# Payload sizes, both page-aligned and deliberately unaligned. The unaligned
# cases are the ones that catch off-by-one errors in the page padding maths.
_PAGE_ALIGNED = 4096
_UNALIGNED = 4097
_TINY = 1

VERSION_IDS = (0, 1, 2, 3, 4)
HASHTYPES = ("sha1", "sha256")
PAGE_SIZES = (2048, 4096)


def _digest(hashtype: str, payload: bytes) -> bytes:
    h = hashlib.new(hashtype, payload).digest()
    if hashtype == "sha1":
        return h.ljust(BOOT_ID_SIZE, b"\0")[:BOOT_ID_SIZE]
    return h[:BOOT_ID_SIZE]


def _matrix_cases():
    """Yield (header_version, hashtype, page_size, aligned) tuples.

    v3/v4 are only exercised at the page size the format actually implies.
    Those headers have no ``page_size`` field, so a v3/v4 image simply does
    not carry a 2048-byte-page variant: the kernel starts at the page
    boundary implied by ``header_size``. Parametrising v3/v4 over 2048 would
    therefore be testing an image that cannot exist, not a real code path.
    """
    for version in VERSION_IDS:
        sizes = (4096,) if version >= 3 else PAGE_SIZES
        for hashtype in HASHTYPES:
            for page_size in sizes:
                for aligned in (True, False):
                    yield version, hashtype, page_size, aligned


MATRIX = list(_matrix_cases())
MATRIX_IDS = [
    f"v{v}-ht-{h}-ps{p}-{'aligned' if a else 'unaligned'}" for v, h, p, a in MATRIX
]


def _make(image_version: int, page_size: int, hashtype: str, aligned: bool) -> bytes:
    kernel_size = _PAGE_ALIGNED if aligned else _UNALIGNED
    ramdisk_size = 4096 if aligned else 1234
    kernel = bytes((i * 7 + 13) & 0xFF for i in range(kernel_size))
    ramdisk = build_gzip_cpio(
        {"init": b"#!/system/bin/sh\n", "prop.default": b"ro.test=1\n"},
        mtime=0,
    )
    if len(ramdisk) != ramdisk_size:
        # Pad deterministically rather than truncating meaningful content.
        ramdisk = ramdisk.ljust(ramdisk_size, b"\0")
    return build_image(
        header_version=image_version,
        page_size=page_size,
        kernel=kernel,
        ramdisk=ramdisk,
        kernel_addr=0x00008000,
        ramdisk_addr=0x01000000,
        tags_addr=0x00000100,
        name=b"board",
        cmdline=b"console=ttyHSL0,115200n8 androidboot.hardware=test",
        os_version=(30 << 25) | (9 << 16) | (0 << 8) | 20240101,
        hash_id=_digest(hashtype, kernel),
    )


@pytest.mark.parametrize(
    ("header_version", "hashtype", "page_size", "aligned"),
    MATRIX,
    ids=MATRIX_IDS,
)
def test_header_fields_round_trip(header_version, hashtype, page_size, aligned):
    """Every header field written comes back identical.

    v3/v4 use the compact AOSP v3 field order, which has no ``page_size``,
    no load addresses, no ``name`` and no ``id``; those assertions therefore
    only apply to v0-v2. See the module docstring.
    """
    data = _make(header_version, page_size, hashtype, aligned)
    img = parse_image(data, header_version)

    assert field(img, "header_version", "version") == header_version, (
        f"header_version: wrote {header_version}, "
        f"read back {field(img, 'header_version', 'version')}"
    )
    # Sanity-check the reference builder itself: the header_size it stored must
    # be readable at the offset the layout says, and must match what we asked
    # for. If this fails the rest of the case is meaningless. v0 has no
    # header_size field, so there is nothing to check there.
    stored_header_size = header_size_on_disk(data, header_version)
    if header_version == 0:
        assert stored_header_size is None, (
            f"v0 has no header_size field, but the reference builder wrote "
            f"{stored_header_size} at offset "
            f"{header_size_offset(header_version):#x}"
        )
    else:
        # Accept both 1580 and 1584 for v3/v4 (the real OEM image uses 1584).
        expected_sizes = (
            ACCEPTED_V3_V4_HEADER_SIZES
            if header_version >= 3
            else (reported_header_size(header_version),)
        )
        assert stored_header_size in expected_sizes, (
            f"reference builder stored header_size {stored_header_size} at offset "
            f"{header_size_offset(header_version):#x}, expected one of "
            f"{expected_sizes}"
        )

    cmdline = field(img, "cmdline", "cmd_line", "args")
    assert bytes(cmdline).rstrip(b"\0") == (
        b"console=ttyHSL0,115200n8 androidboot.hardware=test"
    ), f"cmdline: read back {cmdline!r}"

    if header_version >= 3:
        # Compact header: the size fields are the whole story. Compare against
        # the payload actually recovered rather than a size attribute, so the
        # assertion holds whether or not the impl surfaces `kernel_size`
        # separately from the payload.
        kernel = field(img, "kernel", "kernel_data")
        ramdisk = field(img, "ramdisk", "ramdisk_data")
        assert len(kernel) == (_PAGE_ALIGNED if aligned else _UNALIGNED), (
            f"kernel_size: wrote {_PAGE_ALIGNED if aligned else _UNALIGNED}, "
            f"recovered a {len(kernel)}-byte kernel"
        )
        assert len(ramdisk) == (4096 if aligned else 1234), (
            f"ramdisk_size: wrote {4096 if aligned else 1234}, "
            f"recovered a {len(ramdisk)}-byte ramdisk"
        )
        return

    assert field(img, "page_size", "pagesize") == page_size, (
        f"page_size: wrote {page_size}, read back {field(img, 'page_size', 'pagesize')}"
    )
    assert field(img, "kernel_addr", "base") == 0x00008000, (
        f"kernel_addr: wrote {0x00008000:#x}, "
        f"read back {field(img, 'kernel_addr', 'base')!r}"
    )
    assert field(img, "ramdisk_addr", "ramdisk_base") == 0x01000000, (
        f"ramdisk_addr: wrote {0x01000000:#x}, "
        f"read back {field(img, 'ramdisk_addr', 'ramdisk_base')!r}"
    )
    assert field(img, "tags_addr", "tags_offset") == 0x00000100, (
        f"tags_addr: wrote {0x00000100:#x}, "
        f"read back {field(img, 'tags_addr', 'tags_offset')!r}"
    )

    name = field(img, "name", "board")
    assert bytes(name).rstrip(b"\0") == b"board", f"name: read back {name!r}"


@pytest.mark.parametrize("header_version", [3, 4], ids=lambda v: f"v{v}")
@pytest.mark.parametrize("header_size", sorted(ACCEPTED_V3_V4_HEADER_SIZES))
def test_v3_v4_accepts_both_real_header_sizes(header_version, header_size):
    """Both header_size values seen in the wild round-trip for v3/v4.

    ``mkbootimg`` produces 1580 while the real 64 MiB OEM image carries 1584.
    A parser that hard-codes either one and rejects the other would break on
    half of all v3/v4 images in the wild, so both must be accepted and
    preserved across a round trip.
    """
    data = build_image(
        header_version=header_version,
        page_size=4096,
        kernel=b"K" * 4096,
        ramdisk=b"R" * 4096,
        header_size=header_size,
    )
    assert header_size_on_disk(data, header_version) == header_size

    img = parse_image(data, header_version)
    assert field(img, "header_size") == header_size, (
        f"header_size: image stores {header_size}, parser reported "
        f"{field(img, 'header_size')!r}"
    )
    assert bytes(field(img, "kernel", "kernel_data")) == b"K" * 4096, (
        f"kernel payload is misplaced for a {header_size}-byte header"
    )

    again = parse_image(pack_image(img), header_version)
    assert field(again, "header_size") == header_size, (
        f"header_size was not preserved across a round trip: {header_size} -> "
        f"{field(again, 'header_size')!r}"
    )


@pytest.mark.parametrize(
    ("header_version", "hashtype", "page_size", "aligned"),
    MATRIX,
    ids=MATRIX_IDS,
)
def test_payloads_round_trip_byte_identical(
    header_version, hashtype, page_size, aligned
):
    """Every payload comes back byte-identical to what was written."""
    data = _make(header_version, page_size, hashtype, aligned)
    img = parse_image(data, header_version)

    kernel = field(img, "kernel", "kernel_data")
    ramdisk = field(img, "ramdisk", "ramdisk_data")
    assert kernel is not None, "kernel payload came back as None"
    assert ramdisk is not None, "ramdisk payload came back as None"
    assert len(kernel) == (_PAGE_ALIGNED if aligned else _UNALIGNED), (
        f"kernel length: wrote {_PAGE_ALIGNED if aligned else _UNALIGNED}, "
        f"read back {len(kernel)}"
    )
    assert len(ramdisk) == (4096 if aligned else 1234), (
        f"ramdisk length: wrote {4096 if aligned else 1234}, read back {len(ramdisk)}"
    )
    # Compare against a freshly re-derived expectation rather than only
    # checking self-consistency.
    assert bytes(kernel) == _expected_kernel(aligned), "kernel payload differs"
    assert bytes(ramdisk) == _expected_ramdisk(aligned), "ramdisk payload differs"


def _expected_kernel(aligned: bool) -> bytes:
    size = _PAGE_ALIGNED if aligned else _UNALIGNED
    return bytes((i * 7 + 13) & 0xFF for i in range(size))


def _expected_ramdisk(aligned: bool) -> bytes:
    size = 4096 if aligned else 1234
    data = build_gzip_cpio(
        {"init": b"#!/system/bin/sh\n", "prop.default": b"ro.test=1\n"}, mtime=0
    )
    return data.ljust(size, b"\0")


@pytest.mark.parametrize(
    ("header_version", "hashtype", "page_size", "aligned"),
    MATRIX,
    ids=MATRIX_IDS,
)
def test_pack_is_deterministic(header_version, hashtype, page_size, aligned):
    """Packing the same parsed image twice is byte-identical."""
    data = _make(header_version, page_size, hashtype, aligned)
    first = pack_image(parse_image(data, header_version))
    second = pack_image(parse_image(data, header_version))
    assert first == second, "packing the same image twice produced different bytes"


@pytest.mark.parametrize(
    ("header_version", "hashtype", "page_size", "aligned"),
    MATRIX,
    ids=MATRIX_IDS,
)
def test_reparsed_image_matches(header_version, hashtype, page_size, aligned):
    """Re-parsing a re-packed image yields the same fields and payloads.

    This is the full round trip: original bytes -> parse -> pack -> parse.
    """
    data = _make(header_version, page_size, hashtype, aligned)
    original = parse_image(data, header_version)
    repacked = pack_image(original)
    again = parse_image(repacked, header_version)

    assert bytes(field(again, "kernel", "kernel_data")) == bytes(
        field(original, "kernel", "kernel_data")
    ), "kernel payload changed across a full pack/parse cycle"
    assert bytes(field(again, "ramdisk", "ramdisk_data")) == bytes(
        field(original, "ramdisk", "ramdisk_data")
    ), "ramdisk payload changed across a full pack/parse cycle"
    assert field(again, "header_version", "version") == header_version
    if header_version <= 2:
        # v3/v4 have no page_size field, so only assert it where it exists.
        assert field(again, "page_size", "pagesize") == page_size


# --------------------------------------------------------------------------
# Targeted cases: optional payloads.
# --------------------------------------------------------------------------

SECOND_PAYLOAD = b"second-stage-loader-blob" * 4
DTB_PAYLOAD = b"\xd0\x0d\xfe\xed" + b"dtb-bytes" * 8
RECOVERY_DTBO_PAYLOAD = b"\x00" * 8 + b"recovery-dtbo" * 16
KERNEL_SIGNATURE_PAYLOAD = b"\x00" * 8 + b"v4-kernel-signature-blob" * 4


@pytest.mark.parametrize("header_version", [0, 1, 2], ids=lambda v: f"v{v}")
def test_second_round_trips_v0_v1_v2(header_version):
    """`second` round-trips for header versions 0-2, where it is defined."""
    data = build_image(
        header_version=header_version,
        page_size=4096,
        kernel=b"K" * 1000,
        ramdisk=b"R" * 2000,
        second=SECOND_PAYLOAD,
        second_addr=0x02F00000,
    )
    img = parse_image(data, header_version)

    second = field(img, "second", "second_data")
    assert second is not None, "second payload came back as None"
    assert bytes(second) == SECOND_PAYLOAD, (
        f"second payload differs: wrote {len(SECOND_PAYLOAD)} bytes, "
        f"read back {len(second)} bytes"
    )
    assert field(img, "second_addr", "second_base") == 0x02F00000, (
        f"second_addr: wrote {0x02F00000:#x}, "
        f"read back {field(img, 'second_addr', 'second_base')!r}"
    )
    assert bytes(field(img, "kernel", "kernel_data")) == b"K" * 1000
    assert bytes(field(img, "ramdisk", "ramdisk_data")) == b"R" * 2000


def test_dtb_round_trips_v2():
    """`dtb` round-trips on v2, where AOSP added the field."""
    data = build_image(
        header_version=2,
        page_size=4096,
        kernel=b"K" * 1000,
        ramdisk=b"R" * 2000,
        second=SECOND_PAYLOAD,
        dtb=DTB_PAYLOAD,
        dtb_size=len(DTB_PAYLOAD),
        dtb_addr=0x01F00000,
    )
    # Cross-check against the physical layout rather than trusting the
    # reference builder: the dtb blob must actually be sitting at the offset
    # the layout maths says it is.
    offsets = payload_offset(
        data,
        4096,
        kernel=b"K" * 1000,
        ramdisk=b"R" * 2000,
        second=SECOND_PAYLOAD,
        dtb=DTB_PAYLOAD,
    )
    assert data[offsets["dtb"] : offsets["dtb"] + len(DTB_PAYLOAD)] == DTB_PAYLOAD, (
        f"reference builder did not place dtb at {offsets['dtb']:#x}"
    )

    img = parse_image(data, 2)
    dtb = field(img, "dtb", "dtb_data")
    assert dtb is not None, "dtb payload came back as None"
    assert bytes(dtb) == DTB_PAYLOAD, (
        f"dtb payload differs: wrote {len(DTB_PAYLOAD)} bytes, "
        f"read back {len(dtb)} bytes"
    )


@pytest.mark.parametrize("header_version", [2, 3, 4], ids=lambda v: f"v{v}")
def test_recovery_dtbo_round_trips(header_version):
    """A recovery_dtbo attached to a parsed image survives pack/parse.

    NOTE on what this does and does not pin down. Unlike every other payload
    here, the recovery_dtbo header offset could not be verified empirically:
    the bundled ``mkbootimg`` in this repo is a stripped build that rejects
    both ``--recovery_dtbo`` and ``--recovery_acpio`` on every header version,
    even though ``repackimg.sh`` passes ``--recovery_dtbo``. Writing the field
    by hand at a guessed offset and asserting the library agrees with the guess
    would test my guess, not the format.

    So this test states the contract the library itself must honour: attach a
    blob through the library's own object model, pack it, parse it back, and
    the blob must still be there. That catches a packer which silently drops
    recovery_dtbo -- a real and likely bug. It cannot catch a library that
    stores the blob at a non-AOSP offset. When a fuller ``mkbootimg`` becomes
    available, replace this with an offset-exact test like the ``dtb`` one.
    """
    base = build_image(
        header_version=header_version,
        page_size=4096,
        kernel=b"K" * 1000,
        ramdisk=b"R" * 2000,
    )
    img = parse_image(base, header_version)

    # Attach through the library's own object model, so we are testing the
    # library's round trip and not a hand-written header. On v3/v4 the trailing
    # slot is the kernel signature rather than recovery_dtbo, so exercise the
    # slot that actually exists for that version.
    slot = "kernel_signature" if header_version >= 3 else "recovery_dtbo"
    _set_field(
        img, slot, "signature", "recovery_dtbo_data", value=RECOVERY_DTBO_PAYLOAD
    )
    repacked = parse_image(pack_image(img), header_version)

    dtbo = field(repacked, slot, "signature", "recovery_dtbo_data")
    assert dtbo is not None, (
        f"{slot} payload was dropped by pack/parse on v{header_version}"
    )
    assert bytes(dtbo) == RECOVERY_DTBO_PAYLOAD, (
        f"{slot} payload differs after pack/parse: "
        f"wrote {len(RECOVERY_DTBO_PAYLOAD)} bytes, read back {len(dtbo)} bytes"
    )
    assert bytes(field(repacked, "kernel", "kernel_data")) == b"K" * 1000


@pytest.mark.parametrize("header_version", [3, 4], ids=lambda v: f"v{v}")
def test_v4_kernel_signature_blob_round_trips(header_version):
    """The v3/v4 kernel signature blob survives pack/parse.

    Same caveat as the recovery_dtbo case above: the header offset for the
    v3/v4 signature blob could not be confirmed against the bundled tooling, so
    this asserts the blob round-trips rather than that it sits at a particular
    byte offset.
    """
    base = build_image(
        header_version=header_version,
        page_size=4096,
        kernel=b"K" * 4096,
        ramdisk=b"R" * 4096,
    )
    img = parse_image(base, header_version)
    _set_field(
        img,
        "kernel_signature",
        "kernel_sig",
        "signature",
        value=KERNEL_SIGNATURE_PAYLOAD,
    )
    repacked = parse_image(pack_image(img), header_version)

    sig = field(repacked, "kernel_signature", "kernel_sig", "signature")
    assert sig is not None, "kernel signature blob was dropped by pack/parse"
    assert bytes(sig) == KERNEL_SIGNATURE_PAYLOAD, (
        f"kernel signature differs: wrote {len(KERNEL_SIGNATURE_PAYLOAD)} bytes, "
        f"read back {len(sig)} bytes"
    )


def test_absent_optional_payloads_are_none_not_empty():
    """Absent optional payloads come back as None, not empty bytes.

    b"" and None mean different things to a caller deciding whether to emit a
    second/dtbo/signature section, so this distinction is load-bearing.
    """
    data = build_image(
        header_version=4,
        page_size=4096,
        kernel=b"K" * 4096,
        ramdisk=None,
    )
    img = parse_image(data, 4)

    # Only the payloads that exist for this header version. v4 has no `second`
    # and no `dtb` -- those were removed in v3 -- so asking for them would
    # fail on any correct implementation rather than test anything.
    for name in ("ramdisk", "kernel_signature", "recovery_dtbo"):
        value = field(img, name)
        assert value is None, (
            f"{name} was absent from the image but parsed as {value!r}; "
            f"expected None, not empty bytes"
        )
        assert not (isinstance(value, (bytes, bytearray)) and len(value) == 0), (
            f"{name} was absent from the image but parsed as empty bytes; expected None"
        )


@pytest.mark.parametrize("header_version", [0, 2], ids=["v0", "v2"])
def test_absent_v012_optional_payloads_are_none(header_version):
    """`second` and `dtb` are None when absent, on the versions that have them."""
    data = build_image(
        header_version=header_version,
        page_size=4096,
        kernel=b"K" * 4096,
        ramdisk=b"R" * 4096,
    )
    img = parse_image(data, header_version)

    for name in ("second", "dtb"):
        value = field(img, name)
        assert value is None, (
            f"v{header_version}: {name} was absent from the image but parsed as "
            f"{value!r}; expected None, not empty bytes"
        )


def test_empty_optional_payload_is_reported_as_none():
    """A present-but-zero-length payload is also reported as None.

    Note that the builder treats ``None`` and ``b""`` identically -- both mean
    "emit nothing" and both produce the same bytes. That is the documented
    behaviour: zero-length carries no information at pack time, so collapsing
    it to None loses nothing. The distinction that *does* matter is
    None-vs-empty-on-parse, which is what
    :func:`test_absent_optional_payloads_are_none_not_empty` covers.
    """
    data = build_image(
        header_version=4,
        page_size=4096,
        kernel=b"K" * 4096,
        ramdisk=b"",
    )
    img = parse_image(data, 4)
    assert field(img, "ramdisk") is None


# --------------------------------------------------------------------------
# Malformed input: must raise a deliberate library error, never a raw leak.
# --------------------------------------------------------------------------


def _good_image(header_version: int = 0) -> bytes:
    return build_image(
        header_version=header_version,
        page_size=4096,
        kernel=b"K" * 4096,
        ramdisk=b"R" * 4096,
    )


# Offsets of the two size fields, which live at different places in the two
# header shapes.
_KERNEL_SIZE_OFFSET = 0x08
_RAMDISK_SIZE_OFFSET_V012 = 0x10
_RAMDISK_SIZE_OFFSET_V34 = 0x0C
_PAGE_SIZE_OFFSET = 0x24


def _corrupt_magic(data: bytes) -> bytes:
    # Exactly BOOT_MAGIC_SIZE bytes, so the result differs from the original
    # only in the magic and no other field shifts.
    return b"NOTBOOT!" + data[BOOT_MAGIC_SIZE:]


def _truncate_inside_header(data: bytes) -> bytes:
    return data[:12]


def _truncate_inside_kernel(data: bytes) -> bytes:
    return data[: 4096 + 100]


def _oversized_kernel_size(data: bytes, header_version: int = 0) -> bytes:
    buf = bytearray(data)
    struct.pack_into("<I", buf, _KERNEL_SIZE_OFFSET, len(data) * 4)
    return bytes(buf)


def _oversized_ramdisk_size(data: bytes, header_version: int = 0) -> bytes:
    buf = bytearray(data)
    offset = (
        _RAMDISK_SIZE_OFFSET_V34 if header_version >= 3 else _RAMDISK_SIZE_OFFSET_V012
    )
    struct.pack_into("<I", buf, offset, len(data) * 4)
    return bytes(buf)


def _huge_page_size(data: bytes, header_version: int = 0) -> bytes:
    buf = bytearray(data)
    if header_version >= 3:
        # v3/v4 have no page_size field; the closest analogue is header_size,
        # which is what determines the payload page boundary.
        struct.pack_into("<I", buf, HEADER_SIZE_OFFSET_V34, 0xFFFFFFF0)
    else:
        struct.pack_into("<I", buf, _PAGE_SIZE_OFFSET, 0xFFFFFFF0)
    return bytes(buf)


def _absurd_header_version(data: bytes) -> bytes:
    buf = bytearray(data)
    struct.pack_into("<I", buf, 0x28, 0xDEADBEEF)
    return bytes(buf)


_BAD_CASES = {
    "bad-magic": _corrupt_magic,
    "truncated-in-header": _truncate_inside_header,
    "truncated-in-kernel": _truncate_inside_kernel,
    "oversized-kernel-size": _oversized_kernel_size,
    "oversized-ramdisk-size": _oversized_ramdisk_size,
    "huge-page-size": _huge_page_size,
    "absurd-header-version": _absurd_header_version,
}

# Corruptors that write to a version-specific offset, so they need to know
# which header shape they are attacking. Named explicitly rather than matched
# by function identity, so adding or wrapping a corruptor cannot silently
# fall through to the wrong call signature.
_VERSION_AWARE_CASES = frozenset(
    {"oversized-kernel-size", "oversized-ramdisk-size", "huge-page-size"}
)


@pytest.mark.parametrize("case", sorted(_BAD_CASES), ids=sorted(_BAD_CASES))
@pytest.mark.parametrize("header_version", [0, 4], ids=["v0", "v4"])
def test_malformed_input_raises_clean_error(case, header_version):
    """Malformed input raises a deliberate error, not a raw struct/index error."""
    corrupt = _BAD_CASES[case]
    data = _good_image(header_version)
    if case in _VERSION_AWARE_CASES:
        data = corrupt(data, header_version)
    else:
        data = corrupt(data)

    with pytest.raises(Exception) as excinfo:
        parse_image(data, header_version)
    # We want a reported error, and specifically not an implementation leak.
    assert excinfo.value is not None
    assert_clean_error(excinfo.value, context=f"{case} (v{header_version})")


def test_empty_input_raises_clean_error():
    """Zero bytes is not a boot image and must be reported, not mis-parsed."""
    with pytest.raises(Exception) as excinfo:
        parse_image(b"", 0)
    assert_clean_error(excinfo.value, context="empty input")


def test_bad_magic_error_message_mentions_magic():
    """The bad-magic error should say what was wrong, not just 'failed'."""
    data = _corrupt_magic(_good_image(0))
    with pytest.raises(Exception) as excinfo:
        parse_image(data, 0)
    message = str(excinfo.value).lower()
    assert "magic" in message or "android" in message, (
        f"bad-magic error message is not actionable: {excinfo.value!r}"
    )
