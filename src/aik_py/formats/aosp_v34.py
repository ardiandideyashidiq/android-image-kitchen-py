"""AOSP boot image header versions 3 and 4.

Layout verified empirically against the bundled ``mkbootimg`` / ``unpackbootimg``
in ``bin/linux/x86_64`` and cross-checked with AOSP's ``boot_img_hdr_v3`` /
``boot_img_hdr_v4``::

    0x00   8      magic, always ``ANDROID!``
    0x08   4      kernel_size
    0x0C   4      ramdisk_size
    0x10   4      os_version (OS version + patch level, bit packed)
    0x14   4      header_size
    0x18   16     reserved[4]
    0x28   4      header_version
    0x2C   1536   cmdline (asciiz)
    0x62C  4      signature_size (v4 only)

``header_size`` is the offset of the first payload byte.  AOSP's
``boot_img_hdr_v4`` -- and every real v4 image, including the 64 MiB OEM sample
in ``~/Downloads/boot.img`` -- uses 1584, but the bundled ``mkbootimg`` stamps
1580 even for ``--header_version 4`` because it never emits the
``signature_size`` word.  Both are honoured: the value found on disk is what
gets re-emitted, so images from either producer round-trip byte for byte.

Consequently a ``signature_size`` word is only read when ``header_size``
actually reaches 1584.  On a 1580-byte v4 header those four bytes are padding,
and reading them as a length would turn arbitrary fill (``0xff`` is common)
into a bogus signature size.

Payloads follow in the order kernel, ramdisk, signature, each zero-padded up to
the page boundary.

Two points contradict the AOSP prose documentation but match the binaries, so
they are implemented as observed rather than as documented:

* AOSP describes v3/v4 as having no ``ANDROID!`` magic and the header version
  in the first u32.  The bundled tooling keeps the magic at offset 0 and puts
  ``header_version`` at 0x28, so v3/v4 images remain magic-addressed and the
  version dispatch reads a field the classic layout also uses.  A header
  version outside 3..8 is therefore rejected rather than mis-parsed.
* AOSP fixes the v3/v4 page size at 4096 and defines no ``page_size`` field for
  these versions.  There is none here either -- see :data:`PAGE_SIZE`.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

try:  # pragma: no cover - the shared module lands in a sibling unit
    from aik_py.errors import AikError
except ImportError:  # pragma: no cover

    class AikError(Exception):
        """Local stand-in so this unit stands alone."""


BOOT_MAGIC = b"ANDROID!"
BOOT_MAGIC_SIZE = 8
CMDLINE_SIZE = 1536  # BOOT_ARGS_SIZE + BOOT_EXTRA_ARGS_SIZE

MAGIC_OFFSET = 0x00
KERNEL_SIZE_OFFSET = 0x08
RAMDISK_SIZE_OFFSET = 0x0C
OS_VERSION_OFFSET = 0x10
HEADER_SIZE_OFFSET = 0x14
RESERVED_OFFSET = 0x18
HEADER_VERSION_OFFSET = 0x28
CMDLINE_OFFSET = 0x2C
SIGNATURE_SIZE_OFFSET = 0x62C

HEADER_SIZE_V3 = CMDLINE_OFFSET + CMDLINE_SIZE  # 1580
HEADER_SIZE_V4 = SIGNATURE_SIZE_OFFSET + 4  # 1584

#: v3/v4 have no page_size field; AOSP fixes it at 4096 and the bundled
#: mkbootimg honours that regardless of ``--pagesize``.  Briefed as a 16384
#: fallback for a zero page_size field, but there is no field to read and the
#: bundled unpackbootimg was measured slicing payloads at 4096 -- a kernel
#: hand-placed on the 16384 boundary came back corrupt.  4096 it is.
PAGE_SIZE = 4096

#: The bundled tooling computes ``version - 3`` unsigned and compares against a
#: small bound, routing 3..8 through the v3/v4 branch and 0/1/2 through the
#: classic one.  Confirmed by building versions 0..9 and watching which path the
#: unpacker takes.  Only 3 and 4 are AOSP-defined; 5..8 are accepted because
#: that is what the tooling dispatches, and are written with the v4 layout.
MIN_HEADER_VERSION = 3
MAX_HEADER_VERSION = 8

MAX_PAGE_SIZE = 65536
MAX_HEADER_SIZE = 1 << 20

_U32 = struct.Struct("<I")
_U32X4 = struct.Struct("<4I")


class BootImageError(AikError):
    """Raised when a v3/v4 boot image cannot be parsed or packed."""


@dataclass(slots=True)
class BootImageV34:
    """A parsed boot image with header version 3 or 4."""

    header_version: int = 4
    kernel: bytes = b""
    ramdisk: bytes = b""
    signature: bytes = b""
    cmdline: bytes = b""
    os_version: tuple[int, int, int] = (0, 0, 0)
    os_patch_level: tuple[int, int] = (2000, 0)
    reserved: tuple[int, int, int, int] = (0, 0, 0, 0)
    page_size: int = PAGE_SIZE
    #: Offset of the first payload byte.  0 means "derive from header_version",
    #: which is what a freshly built image wants; :func:`parse` sets it to the
    #: value actually found so non-canonical headers round-trip unchanged.
    header_size: int = 0
    raw_header: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def magic(self) -> bytes:
        return BOOT_MAGIC

    @property
    def kernel_size(self) -> int:
        return len(self.kernel)

    @property
    def ramdisk_size(self) -> int:
        return len(self.ramdisk)

    @property
    def signature_size(self) -> int:
        return len(self.signature)


def uses_v34_layout(header_version: int) -> bool:
    """Return True when ``header_version`` selects the v3/v4 layout."""
    return MIN_HEADER_VERSION <= header_version <= MAX_HEADER_VERSION


def _coerce_reserved(reserved: object) -> tuple[int, int, int, int]:
    """Validate the reserved[4] tuple so ``pack`` never leaks a struct.error."""
    if not isinstance(reserved, tuple | list) or len(reserved) != 4:
        raise BootImageError(f"reserved must be a sequence of 4 ints, got {reserved!r}")
    words = []
    for i, value in enumerate(reserved):
        if not isinstance(value, int) or isinstance(value, bool):
            raise BootImageError(f"reserved[{i}] must be an int, got {value!r}")
        if not 0 <= value <= 0xFFFF_FFFF:
            raise BootImageError(f"reserved[{i}] {value} out of range 0..0xffffffff")
        words.append(value)
    return tuple(words)


def encode_os_version(
    os_version: tuple[int, int, int], os_patch_level: tuple[int, int]
) -> int:
    """Pack ``A.B.C`` and ``YYYY-MM`` into the single v3/v4 os_version u32.

    The bit layout was derived bit by bit from ``mkbootimg`` output and matches
    AOSP's ``SetOsVersion``/``SetOsPatchLevel``::

        A[31:25] B[24:18] C[17:11] (YYYY-2000)[10:4] M[3:0]

    The year offset is 2000.  A zeroed field therefore means "unset" rather than
    "1980"; that is why a zero decodes to month 0 / year 2000 here and to
    ``0-00`` when the bundled unpacker prints it.
    """
    if not isinstance(os_version, tuple | list) or len(os_version) != 3:
        raise BootImageError(f"malformed os_version {os_version!r}: expected 3 ints")
    if not isinstance(os_patch_level, tuple | list) or len(os_patch_level) != 2:
        raise BootImageError(
            f"malformed os_patch_level {os_patch_level!r}: expected 2 ints"
        )
    major, minor, patch = os_version
    year, month = os_patch_level
    for name, value in (
        ("major", major),
        ("minor", minor),
        ("patch", patch),
        ("os_patch_level year", year),
        ("os_patch_level month", month),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise BootImageError(f"{name} must be an int, got {value!r}")
    for name, value in (("major", major), ("minor", minor), ("patch", patch)):
        if not 0 <= value <= 0x7F:
            raise BootImageError(f"os_version {name} {value} out of range 0..127")
    if not 0 <= year - 2000 <= 0x7F:
        raise BootImageError(f"os_patch_level year {year} out of range 2000..2127")
    if not 0 <= month <= 15:
        # Deliberately wider than parse_os_patch_level's 0..12: this is the
        # raw 4-bit field, and images with months 13-15 exist in the wild.
        # Reading and re-emitting one must not corrupt it, so the bit-faithful
        # range is accepted here and the calendar check lives in the string
        # parser where a human is the one supplying the value.
        raise BootImageError(f"os_patch_level month {month} out of range 0..15")
    return (
        ((major & 0x7F) << 25)
        | ((minor & 0x7F) << 18)
        | ((patch & 0x7F) << 11)
        | (((year - 2000) & 0x7F) << 4)
        | (month & 0xF)
    )


def decode_os_version(value: int) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """Split the os_version u32 into ``(A, B, C)`` and ``(YYYY, MM)``."""
    return (
        ((value >> 25) & 0x7F, (value >> 18) & 0x7F, (value >> 11) & 0x7F),
        (((value >> 4) & 0x7F) + 2000, value & 0xF),
    )


def format_os_version(os_version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in os_version)


def format_os_patch_level(os_patch_level: tuple[int, int]) -> str:
    """Render as ``YYYY-MM``, always zero-padding the month.

    The bundled tools disagree with each other: ``mkbootimg`` accepts an
    unpadded ``2020-9`` while ``unpackbootimg`` always prints ``2020-09``.  The
    padding is cosmetic -- the field holds only 4 bits of month either way -- so
    we always emit the padded form.  That makes unpack/repack byte-stable and
    confines the asymmetry to normalising what a user typed.
    """
    year, month = os_patch_level
    return f"{year}-{month:02d}"


def _all_ascii_digits(parts: list[str]) -> bool:
    # isdigit() alone accepts superscripts and other non-ASCII numerals, which
    # int() then rejects with a bare ValueError.
    return all(p and p.isascii() and p.isdigit() for p in parts)


def parse_os_version(text: str) -> tuple[int, int, int]:
    parts = text.strip().split(".")
    if len(parts) != 3 or not _all_ascii_digits(parts):
        raise BootImageError(f'malformed os_version {text!r}: expected "A.B.C"')
    major, minor, patch = (int(p) for p in parts)
    for name, value in (("major", major), ("minor", minor), ("patch", patch)):
        if value > 0x7F:
            raise BootImageError(f"os_version {name} {value} out of range 0..127")
    return major, minor, patch


def parse_os_patch_level(text: str) -> tuple[int, int]:
    """Parse ``YYYY-M`` or ``YYYY-MM``; padding is optional on input.

    Month 0 is accepted because it is what a zeroed header decodes to, and the
    ``format_os_patch_level`` output for an unset field must be re-parseable.
    Months 13-15 are reachable through the raw 4-bit field but are not valid
    calendar months, so they are rejected here.
    """
    parts = text.strip().split("-")
    if len(parts) != 2 or not _all_ascii_digits(parts):
        raise BootImageError(f'malformed os_patch_level {text!r}: expected "YYYY-MM"')
    year, month = int(parts[0]), int(parts[1])
    if not 2000 <= year <= 2127:
        raise BootImageError(f"os_patch_level year {year} out of range 2000..2127")
    if not 0 <= month <= 12:
        raise BootImageError(f"os_patch_level month {month} out of range 0..12")
    return year, month


def _require_magic(data: bytes) -> None:
    if len(data) < BOOT_MAGIC_SIZE:
        raise BootImageError(
            f"truncated image: need {BOOT_MAGIC_SIZE} bytes for magic, got {len(data)}"
        )
    if data[:BOOT_MAGIC_SIZE] != BOOT_MAGIC:
        raise BootImageError(
            f"bad magic: expected {BOOT_MAGIC!r}, got {data[:BOOT_MAGIC_SIZE]!r}"
        )


def _read_u32(data: bytes, offset: int, what: str) -> int:
    if len(data) < offset + 4:
        raise BootImageError(
            f"truncated image: need {offset + 4} bytes to read {what}, got {len(data)}"
        )
    return _U32.unpack_from(data, offset)[0]


def _take(data: bytes, offset: int, size: int, what: str) -> bytes:
    if size == 0:
        return b""
    end = offset + size
    if end > len(data):
        raise BootImageError(
            f"{what} of {size} bytes at offset {offset} runs past file size {len(data)}"
        )
    return data[offset:end]


def _align(size: int, page_size: int) -> int:
    return (size + page_size - 1) // page_size * page_size


def _check_page_size(page_size: int) -> int:
    if not 0 < page_size <= MAX_PAGE_SIZE:
        raise BootImageError(
            f"implausible page_size {page_size}: must be 1..{MAX_PAGE_SIZE}"
        )
    return page_size


def _resolve_header_size(raw: int, header_version: int) -> int:
    """Prefer the stored header_size, falling back to the version default.

    A zero field is the one benign case: OEM images in the wild zero it, and
    rejecting those would lose otherwise-valid files.  A non-zero but absurd
    value is corruption and is reported by the caller's bounds check rather
    than silently replaced.
    """
    expected = HEADER_SIZE_V4 if header_version >= 4 else HEADER_SIZE_V3
    return expected if raw == 0 else raw


def dispatch_header_version(data: bytes) -> int:
    """Validate the magic and return the header_version u32 at 0x28.

    Only meaningful once the magic checks out, because 0x28 holds different
    content in the classic v0-v2 layout.
    """
    _require_magic(data)
    return _read_u32(data, HEADER_VERSION_OFFSET, "header_version")


def parse(data: bytes, *, page_size: int | None = None) -> BootImageV34:
    """Parse a v3/v4 boot image.

    Always raises :class:`BootImageError` for bad magic, a classic header
    version, truncation, absurd sizes, or payloads running off the end --
    never ``struct.error`` or ``IndexError``.

    ``page_size`` must be supplied when the image was written with a non-4096
    alignment, since the header records no page size.  It defaults to
    :data:`PAGE_SIZE`, which is what every real v3/v4 image uses.
    """
    _require_magic(data)

    header_version = _read_u32(data, HEADER_VERSION_OFFSET, "header_version")
    if not uses_v34_layout(header_version):
        # 0x28 holds unrelated content in the classic layout, so without this
        # guard a v0-v2 image is mis-parsed rather than rejected: its kernel
        # addr would be read as kernel_size and its os_version string would
        # come back as a cmdline.
        raise BootImageError(
            f"header_version {header_version} is outside the v3/v4 range "
            f"{MIN_HEADER_VERSION}..{MAX_HEADER_VERSION}; "
            "this image uses the classic layout"
        )
    page_size = _check_page_size(PAGE_SIZE if page_size is None else page_size)

    kernel_size = _read_u32(data, KERNEL_SIZE_OFFSET, "kernel_size")
    ramdisk_size = _read_u32(data, RAMDISK_SIZE_OFFSET, "ramdisk_size")
    raw_os_version = _read_u32(data, OS_VERSION_OFFSET, "os_version")
    header_size = _resolve_header_size(
        _read_u32(data, HEADER_SIZE_OFFSET, "header_size"), header_version
    )
    if header_size > MAX_HEADER_SIZE:
        raise BootImageError(
            f"implausible header_size {header_size}: must be 0..{MAX_HEADER_SIZE}"
        )
    # A header_size below the cmdline would place the first payload inside the
    # header, yielding header padding as a "kernel" that loads as zeros.
    if header_size < CMDLINE_OFFSET + CMDLINE_SIZE:
        raise BootImageError(
            f"implausible header_size {header_size}: v3/v4 headers are at least "
            f"{CMDLINE_OFFSET + CMDLINE_SIZE} bytes"
        )
    if header_size > len(data):
        raise BootImageError(
            f"truncated image: header_size {header_size} exceeds file size {len(data)}"
        )

    # signature_size lives at 0x62C, which is only inside the header when
    # header_size reaches HEADER_SIZE_V4.  The bundled mkbootimg stamps
    # header_size 1580 on v4, leaving that word in the padding region; reading
    # it there would interpret arbitrary padding as a signature length.  So gate
    # on the header actually having room for the field, not on the version.
    signature_size = 0
    if header_version >= 4 and header_size >= HEADER_SIZE_V4:
        signature_size = _read_u32(data, SIGNATURE_SIZE_OFFSET, "signature_size")

    cmdline_field = _take(data, CMDLINE_OFFSET, CMDLINE_SIZE, "cmdline")
    nul = cmdline_field.find(b"\x00")
    cmdline = cmdline_field if nul < 0 else cmdline_field[:nul]

    pos = _align(header_size, page_size)
    kernel = _take(data, pos, kernel_size, "kernel")
    pos += _align(kernel_size, page_size)
    ramdisk = _take(data, pos, ramdisk_size, "ramdisk")
    pos += _align(ramdisk_size, page_size)
    signature = _take(data, pos, signature_size, "signature")

    os_version, os_patch_level = decode_os_version(raw_os_version)

    return BootImageV34(
        header_version=header_version,
        kernel=kernel,
        ramdisk=ramdisk,
        signature=signature,
        cmdline=cmdline,
        os_version=os_version,
        os_patch_level=os_patch_level,
        reserved=_U32X4.unpack_from(data, RESERVED_OFFSET),
        page_size=page_size,
        header_size=header_size,
        raw_header=bytes(data[:header_size]),
    )


def pack(image: BootImageV34) -> bytes:
    """Serialise ``image`` back to v3/v4 bytes.

    There is no page_size field in the v3/v4 header, so a non-4096
    ``image.page_size`` produces an image that :func:`parse` cannot read back
    unless it is given the same value.  Real v3/v4 images -- everything the
    bundled tooling and every OEM boot -- use 4096; the override exists for
    callers that must produce a differently aligned image and know how to read
    it back.
    """
    if not uses_v34_layout(image.header_version):
        raise BootImageError(
            f"header_version {image.header_version} is outside the v3/v4 range "
            f"{MIN_HEADER_VERSION}..{MAX_HEADER_VERSION}"
        )
    page_size = _check_page_size(image.page_size)
    if image.signature and image.header_version < 4:
        raise BootImageError("only v4 images carry a boot signature")
    if not isinstance(image.cmdline, bytes | bytearray):
        raise BootImageError(
            f"cmdline must be bytes, got {type(image.cmdline).__name__}"
        )
    if len(image.cmdline) >= CMDLINE_SIZE:
        raise BootImageError(
            f"cmdline is {len(image.cmdline)} bytes, maximum is {CMDLINE_SIZE - 1}"
        )
    reserved = _coerce_reserved(image.reserved)

    # A parsed image carries the header_size it was read with, which may be
    # non-canonical: the bundled mkbootimg stamps 1580 on v4 and some OEM
    # images use a larger header.  Re-emitting that verbatim is what makes
    # parse/pack byte-stable, so honour it.  Only the 0 sentinel means "derive
    # it"; an explicitly impossible size is a caller mistake worth reporting
    # rather than silently rewriting.
    canonical = HEADER_SIZE_V4 if image.header_version >= 4 else HEADER_SIZE_V3
    header_size = image.header_size or canonical
    if not HEADER_SIZE_V3 <= header_size <= MAX_HEADER_SIZE:
        raise BootImageError(
            f"header_size {header_size} cannot hold a v3/v4 header "
            f"(need {HEADER_SIZE_V3}..{MAX_HEADER_SIZE}); use 0 to derive it "
            f"from header_version"
        )
    if image.header_version >= 4 and image.signature and header_size < HEADER_SIZE_V4:
        raise BootImageError(
            f"header_size {header_size} cannot hold the {len(image.signature)}-byte "
            f"signature_size field a v4 image needs (requires {HEADER_SIZE_V4})"
        )
    header = bytearray(header_size)
    if image.raw_header is not None and len(image.raw_header) == header_size:
        # Carry the original header forward so padding we never interpret (an
        # OEM 0xFF fill, reserved bits we do not model) survives a round trip.
        header[:] = image.raw_header
    header[MAGIC_OFFSET : MAGIC_OFFSET + BOOT_MAGIC_SIZE] = BOOT_MAGIC
    _U32.pack_into(header, KERNEL_SIZE_OFFSET, len(image.kernel))
    _U32.pack_into(header, RAMDISK_SIZE_OFFSET, len(image.ramdisk))
    _U32.pack_into(
        header,
        OS_VERSION_OFFSET,
        encode_os_version(image.os_version, image.os_patch_level),
    )
    _U32.pack_into(header, HEADER_SIZE_OFFSET, header_size)
    _U32X4.pack_into(header, RESERVED_OFFSET, *reserved)
    _U32.pack_into(header, HEADER_VERSION_OFFSET, image.header_version)
    if image.header_version >= 4 and header_size >= HEADER_SIZE_V4:
        _U32.pack_into(header, SIGNATURE_SIZE_OFFSET, len(image.signature))
    header[CMDLINE_OFFSET : CMDLINE_OFFSET + len(image.cmdline)] = image.cmdline

    out = bytearray(header)
    out += bytes(-len(out) % page_size)
    for payload in (image.kernel, image.ramdisk, image.signature):
        out += payload
        out += bytes(-len(payload) % page_size)
    return bytes(out)


def compute_signature(image: BootImageV34) -> bytes:
    """Return a SHA-256 digest over the image's header, kernel and ramdisk.

    AOSP's v4 layout appends a ``boot_signature`` blob whose ``signature_size``
    is the last header field.  The bundled tooling never emits or verifies one
    -- it accepts a hand-built signed image but does not report the field, and
    no oracle was available to pin the digest construction -- so this helper
    produces a deterministic digest for callers that need one, and is not a
    mirror of any observed signature bytes.  Repacking an image does not call
    it: parse/pack carry the signature bytes through untouched.

    The digest is taken over the image as it would be written with no signature
    and no payload sizes, so a v3 image and a v4 image with identical payloads
    do not collide.  Only v4 has a signature_size word to zero out.
    """
    unsigned = bytearray(pack(replace(image, signature=b"", header_size=0)))
    if image.header_version >= 4:
        _U32.pack_into(unsigned, SIGNATURE_SIZE_OFFSET, 0)
    _U32.pack_into(unsigned, KERNEL_SIZE_OFFSET, 0)
    _U32.pack_into(unsigned, RAMDISK_SIZE_OFFSET, 0)
    return hashlib.sha256(bytes(unsigned)).digest()


def replace(image: BootImageV34, **changes: object) -> BootImageV34:
    """Return a copy of ``image`` with ``changes`` applied."""
    from dataclasses import replace as _replace

    return _replace(image, **changes)
