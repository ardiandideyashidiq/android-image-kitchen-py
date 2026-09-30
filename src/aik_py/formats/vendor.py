"""AOSP ``vendor_boot`` (VNDRBOOT) image support.

The vendor_boot image carries only the vendor ramdisk, the device tree blobs and
the vendor command line. There is deliberately **no kernel** and **no hashtype**
on this path -- the Bash original runs ``unpackbootimg`` on VNDRBOOT and never
receives a ``-kernel`` or ``-hashtype`` argument, and there is no ``os_version``
field either. Those concepts are modelled here by their absence: do not expect
``image.kernel`` to exist.

Page-0 layout (all integers little-endian)
------------------------------------------

The offsets below are **verified empirically**, not taken from the header table
in ``system/tools/mkbootimg/include/bootimg/bootimg.h`` -- that table describes
an older draft with ``cmdline_size``/``name_size`` at 0x1C/0x24 which no actual
AOSP ``mkbootimg`` ever emits.

===================  ======  =====================================================
offset               size    field
===================  ======  =====================================================
0x000                8       magic ``VNDRBOOT``
0x008                4       header_version (3 or 4)
0x00C                4       page_size
0x010                4       kernel_addr (load hint only; no kernel payload)
0x014                4       ramdisk_addr
0x018                4       vendor_ramdisk_size
0x01C                2048    vendor_cmdline, NUL-padded
0x81C                4       tags_addr
0x820                16      board name, NUL-padded
0x830                4       header_size
0x834                4       dtb_size
0x838                8       dtb_addr (64-bit! -- a frequent mis-assumption)
0x840                4       vendor_ramdisk_table_size (v4 only)
0x844                4       vendor_ramdisk_table_entry_num (v4 only)
0x848                4       vendor_ramdisk_table_entry_size (v4 only)
0x84C                4       bootconfig_size (v4 only)
===================  ======  =====================================================

The cmdline and name fields are **fixed-width and NUL-padded**; they carry no
length prefix. That is why ``header_size`` exists at all: it tells the loader how
far into page 0 the populated fields actually reach, so the remaining bytes can
be treated as padding. ``header_size`` is 2112 (0x840) for v3 and 2128 (0x850)
for v4, the latter gaining the four v4-only words above.

Payload ordering
----------------

After the header, everything is aligned up to ``page_size``:

* v3: vendor ramdisk, then dtb.
* v4: vendor ramdisk fragments (concatenated in table order), then dtb, then the
  vendor ramdisk fragment table itself, then the bootconfig.

AOSP's ``mkbootimg`` writes the v3 layout even when asked for ``--header_version 4``
with a single ``--vendor_ramdisk``, which is why a v4 image built that way has a
zero entry count and no fragment table. Both shapes parse.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field

try:  # pragma: no cover - exercised only when the shared layer is present
    from aik_py.errors import (
        ImageFormatError,
        ImageTruncatedError,
        ImageValueError,
    )
except ImportError:  # pragma: no cover - this unit must stand alone

    class ImageFormatError(ValueError):
        """The bytes are not a recognisable image of this format."""

    class ImageTruncatedError(ImageFormatError):
        """The image ends before a field or payload it declares."""

    class ImageValueError(ImageFormatError):
        """A field holds a value the format cannot represent."""


__all__ = [
    "MAGIC",
    "SUPPORTED_VERSIONS",
    "VendorBootImage",
    "VendorRamdiskFragment",
    "detect",
    "pack",
    "parse",
]

MAGIC = b"VNDRBOOT"
SUPPORTED_VERSIONS = (3, 4)

CMDLINE_SIZE = 2048
NAME_SIZE = 16
BOARD_ID_SIZE = 16  # four uint32 board identifiers per v4 table entry

HEADER_SIZE_V3 = 2112
HEADER_SIZE_V4 = 2128
# AOSP's own mkbootimg writes HEADER_SIZE_V3 into a header_version 4 image when
# the image has no fragment table and no bootconfig, since the four v4-only
# words are then never populated. Such images are still spec-valid, so v4 must
# accept the smaller value and simply leave the fragment fields at zero.
MIN_HEADER_SIZE = {3: HEADER_SIZE_V3, 4: HEADER_SIZE_V3}
DEFAULT_HEADER_SIZE = {3: HEADER_SIZE_V3, 4: HEADER_SIZE_V4}

DEFAULT_PAGE_SIZE = 4096
MAX_CMDLINE = CMDLINE_SIZE - 1
MAX_NAME = NAME_SIZE - 1

# Fragment types from AOSP ``libbootimg.py``.
RAMDISK_TYPE_NAMES = {
    0: "none",
    1: "platform",
    2: "recovery",
    3: "dlkm",
}

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


@dataclass(frozen=True, slots=True)
class VendorRamdiskFragment:
    """One entry of the v4 vendor ramdisk fragment table.

    The ``size``/``offset``/``type``/``name``/``board_id`` field set is taken
    from AOSP ``mkbootimg.py``'s ``VendorRamdiskTableBuilder``. Multi-fragment
    images could not be verified against a real file -- the bundled ``mkbootimg``
    predates ``--vendor_ramdisk_fragment`` -- so this is **spec-derived, not
    empirically verified**. Single-entry tables *are* verified: both real images
    on this machine carry exactly one 108-byte entry.
    """

    data: bytes
    ramdisk_type: int = 1
    name: bytes = b""
    board_id: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def type_name(self) -> str:
        return RAMDISK_TYPE_NAMES.get(self.ramdisk_type, f"unknown-{self.ramdisk_type}")


@dataclass(slots=True)
class VendorBootImage:
    """A parsed ``vendor_boot`` image.

    ``vendor_ramdisk`` is the whole ramdisk region; for a v4 image with a real
    fragment table it is the concatenation of the fragments and ``fragments``
    holds the individual pieces. For v3, and for v4 images built with a single
    ``--vendor_ramdisk``, ``fragments`` is empty and only ``vendor_ramdisk`` is
    meaningful.
    """

    header_version: int = 3
    page_size: int = DEFAULT_PAGE_SIZE
    kernel_addr: int = 0
    ramdisk_addr: int = 0
    tags_addr: int = 0
    dtb_addr: int = 0
    vendor_cmdline: bytes = b""
    name: bytes = b""
    vendor_ramdisk: bytes = b""
    dtb: bytes | None = None
    bootconfig: bytes | None = None
    fragments: tuple[VendorRamdiskFragment, ...] = ()
    header_size: int | None = None
    # Trailing bytes the format does not describe. Real images carry an AVB
    # vbmeta partition here; kept verbatim so a parse/pack round-trip is lossless.
    trailing: bytes = field(default=b"", repr=False)

    @property
    def cmdline(self) -> str:
        return self.vendor_cmdline.decode("utf-8", "replace")

    @property
    def board(self) -> str:
        return self.name.decode("utf-8", "replace")

    @property
    def vendor_ramdisk_size(self) -> int:
        return len(self.vendor_ramdisk)

    @property
    def dtb_size(self) -> int:
        return len(self.dtb) if self.dtb else 0


def _align(value: int, page_size: int) -> int:
    return value + (-value % page_size)


def _u32(data: bytes, offset: int, what: str) -> int:
    if offset + 4 > len(data):
        raise ImageTruncatedError(
            f"truncated image: {what} needs 4 bytes at offset {offset:#x}, "
            f"only {len(data)} bytes available"
        )
    return _U32.unpack_from(data, offset)[0]


def _u64(data: bytes, offset: int, what: str) -> int:
    if offset + 8 > len(data):
        raise ImageTruncatedError(
            f"truncated image: {what} needs 8 bytes at offset {offset:#x}, "
            f"only {len(data)} bytes available"
        )
    return _U64.unpack_from(data, offset)[0]


def _read_padded(data: bytes, offset: int, width: int, what: str) -> bytes:
    """Read a fixed-width NUL-padded field, stripping at the first NUL."""
    if offset + width > len(data):
        raise ImageTruncatedError(
            f"truncated image: {what} needs {width} bytes at offset {offset:#x}, "
            f"only {len(data)} bytes available"
        )
    raw = data[offset : offset + width]
    nul = raw.find(b"\x00")
    return raw if nul < 0 else raw[:nul]


def _check_region(
    data: bytes, offset: int, size: int, what: str, page_size: int
) -> None:
    """Validate that a page-aligned region of ``size`` bytes fits in the file."""
    if size < 0 or offset < 0:
        raise ImageFormatError(f"negative {what} ({size}) at offset {offset:#x}")
    end = offset + size
    if end > len(data):
        raise ImageTruncatedError(
            f"truncated image: {what} declares {size} bytes at offset {offset:#x} "
            f"but the image is only {len(data)} bytes"
        )
    if offset % page_size:
        raise ImageFormatError(f"{what} offset {offset:#x} is not page-aligned")


def detect(data: bytes) -> bool:
    """True when ``data`` starts with the vendor_boot magic."""
    return data[:8] == MAGIC


def parse(data: bytes) -> VendorBootImage:
    """Parse a ``vendor_boot`` image.

    Raises:
        ImageFormatError: bad magic, unsupported version, or a field that cannot
            describe the file it is embedded in.
        ImageTruncatedError: the image ends before a declared field or payload.
    """
    if len(data) < 8:
        raise ImageTruncatedError(
            f"truncated image: need at least 8 bytes for the magic, got {len(data)}"
        )
    if data[:8] != MAGIC:
        raise ImageFormatError(
            f"bad magic: expected {MAGIC!r}, got {bytes(data[:8])!r}"
        )

    header_version = _u32(data, 0x08, "header_version")
    if header_version not in SUPPORTED_VERSIONS:
        raise ImageFormatError(
            f"unsupported vendor_boot header_version {header_version}; "
            f"expected one of {SUPPORTED_VERSIONS}"
        )

    page_size = _u32(data, 0x0C, "page_size")
    if page_size <= 0 or page_size & (page_size - 1):
        raise ImageFormatError(
            f"invalid page_size {page_size}; expected a power of two"
        )

    kernel_addr = _u32(data, 0x10, "kernel_addr")
    ramdisk_addr = _u32(data, 0x14, "ramdisk_addr")
    vendor_ramdisk_size = _u32(data, 0x18, "vendor_ramdisk_size")
    vendor_cmdline = _read_padded(data, 0x1C, CMDLINE_SIZE, "vendor_cmdline")
    tags_addr = _u32(data, 0x81C, "tags_addr")
    name = _read_padded(data, 0x820, NAME_SIZE, "board name")
    header_size = _u32(data, 0x830, "header_size")
    dtb_size = _u32(data, 0x834, "dtb_size")
    dtb_addr = _u64(data, 0x838, "dtb_addr")

    table_size = table_num = table_entry_size = bootconfig_size = 0
    table_entries: bytes = b""
    if header_size < MIN_HEADER_SIZE[header_version]:
        raise ImageFormatError(
            f"header_size {header_size} is smaller than the "
            f"{MIN_HEADER_SIZE[header_version]}-byte minimum for "
            f"header_version {header_version}"
        )

    if header_version >= 4:
        if HEADER_SIZE_V3 < header_size < HEADER_SIZE_V4:
            # Producers write exactly 2112 (no v4 extras) or 2128 (v4 words
            # present). Anything between leaves the fragment-table words'
            # presence ambiguous, and guessing "absent" would silently drop a
            # real fragment table into the trailing bytes.
            raise ImageFormatError(
                f"header_version 4 with header_size {header_size} is ambiguous; "
                f"expected {HEADER_SIZE_V3} or at least {HEADER_SIZE_V4}"
            )
        if header_size >= HEADER_SIZE_V4:
            table_size = _u32(data, 0x840, "vendor_ramdisk_table_size")
            table_num = _u32(data, 0x844, "vendor_ramdisk_table_entry_num")
            table_entry_size = _u32(data, 0x848, "vendor_ramdisk_table_entry_size")
            bootconfig_size = _u32(data, 0x84C, "bootconfig_size")

    payload_start = _align(header_size, page_size)

    ramdisk = b""
    fragments: tuple[VendorRamdiskFragment, ...] = ()
    dtb: bytes | None = None
    bootconfig: bytes | None = None
    cursor = payload_start

    _check_region(data, cursor, vendor_ramdisk_size, "vendor ramdisk", page_size)
    ramdisk = data[cursor : cursor + vendor_ramdisk_size]
    cursor = _align(cursor + vendor_ramdisk_size, page_size)

    if dtb_size:
        _check_region(data, cursor, dtb_size, "dtb", page_size)
        dtb = data[cursor : cursor + dtb_size]
        cursor = _align(cursor + dtb_size, page_size)

    # The fragment table physically follows the ramdisks and the dtb, even though
    # it describes where each fragment starts inside the ramdisk region.
    if table_num:
        entry_size_min = 44 + 4 * BOARD_ID_SIZE
        if table_entry_size < entry_size_min:
            raise ImageFormatError(
                f"vendor ramdisk table entry_size {table_entry_size} is too small; "
                f"need at least {entry_size_min} bytes"
            )
        expected_table = table_num * table_entry_size
        if table_size != expected_table:
            raise ImageFormatError(
                f"vendor ramdisk table_size {table_size} does not match "
                f"{table_num} entries of {table_entry_size} bytes "
                f"(expected {expected_table})"
            )
        _check_region(data, cursor, table_size, "vendor ramdisk table", page_size)
        table_entries = data[cursor : cursor + table_size]
        cursor = _align(cursor + table_size, page_size)
        fragments = _slice_fragments(
            ramdisk, table_entries, table_entry_size, table_num
        )

    if header_version >= 4 and bootconfig_size:
        _check_region(data, cursor, bootconfig_size, "bootconfig", page_size)
        bootconfig = data[cursor : cursor + bootconfig_size]
        cursor = _align(cursor + bootconfig_size, page_size)

    return VendorBootImage(
        header_version=header_version,
        page_size=page_size,
        kernel_addr=kernel_addr,
        ramdisk_addr=ramdisk_addr,
        tags_addr=tags_addr,
        dtb_addr=dtb_addr,
        vendor_cmdline=vendor_cmdline,
        name=name,
        vendor_ramdisk=ramdisk,
        dtb=dtb,
        bootconfig=bootconfig,
        fragments=fragments,
        header_size=header_size,
        trailing=data[cursor:],
    )


def _slice_fragments(
    ramdisk: bytes, table: bytes, entry_size: int, count: int
) -> tuple[VendorRamdiskFragment, ...]:
    """Turn raw v4 table bytes into fragments carved out of ``ramdisk``.

    Entry layout (AOSP ``VendorRamdiskTableBuilder``): size, offset, type as
    uint32 each, then a 32-byte NUL-padded name, then 16 uint32 board ids.
    """
    fragments: list[VendorRamdiskFragment] = []
    for index in range(count):
        entry = table[index * entry_size : (index + 1) * entry_size]
        size, offset, ramdisk_type = struct.unpack_from("<III", entry, 0)
        name = entry[12:44].split(b"\x00", 1)[0]
        board_id = struct.unpack_from(f"<{BOARD_ID_SIZE}I", entry, 44)
        end = offset + size
        if end > len(ramdisk):
            raise ImageFormatError(
                f"vendor ramdisk fragment {index} spans {offset:#x}..{end:#x} "
                f"but the ramdisk is only {len(ramdisk)} bytes"
            )
        fragments.append(
            VendorRamdiskFragment(
                data=ramdisk[offset:end],
                ramdisk_type=ramdisk_type,
                name=name,
                # Keep every slot: a 0 board id means "this slot is unused",
                # so dropping it would shift the remaining ids to the wrong
                # boards on repack.
                board_id=tuple(board_id),
            )
        )
    return tuple(fragments)


def _pack_fixed_field(value: bytes, width: int, what: str) -> bytes:
    # AOSP pads with ``pack(f'{SIZE}s', value)``, which accepts a completely
    # full field; _read_padded returns all `width` bytes in that case, so pack
    # must accept the same range or a valid image could not be re-emitted.
    if len(value) > width:
        raise ImageValueError(
            f"{what} is {len(value)} bytes but its fixed-width field is {width}"
        )
    return value.ljust(width, b"\x00")


def pack(image: VendorBootImage) -> bytes:
    """Serialise a :class:`VendorBootImage` back into ``vendor_boot`` bytes.

    The result parses back through :func:`parse` with identical field values and
    byte-identical payloads.
    """
    version = image.header_version
    if version not in SUPPORTED_VERSIONS:
        raise ImageValueError(
            f"cannot pack header_version {version}; expected one of {SUPPORTED_VERSIONS}"
        )
    page_size = image.page_size
    if not 0 < page_size <= 0xFFFFFFFF or page_size & (page_size - 1):
        raise ImageValueError(
            f"invalid page_size {page_size}; expected a power of two that fits in a "
            f"32-bit field"
        )

    header_size = image.header_size
    if header_size is None:
        # A v4 image only needs the extra 16 bytes when it actually carries a
        # fragment table or a bootconfig; otherwise match AOSP and reuse 2112.
        has_v4_extras = bool(image.fragments) or bool(image.bootconfig)
        header_size = (
            HEADER_SIZE_V4 if (version == 4 and has_v4_extras) else HEADER_SIZE_V3
        )
    if header_size < MIN_HEADER_SIZE[version]:
        raise ImageValueError(
            f"header_size {header_size} is smaller than the "
            f"{MIN_HEADER_SIZE[version]}-byte minimum for header_version {version}"
        )
    if version == 4 and HEADER_SIZE_V3 < header_size < HEADER_SIZE_V4:
        raise ImageValueError(
            f"header_version 4 with header_size {header_size} is ambiguous; "
            f"use {HEADER_SIZE_V3} or at least {HEADER_SIZE_V4}"
        )

    for addr, what in (
        (image.kernel_addr, "kernel_addr"),
        (image.ramdisk_addr, "ramdisk_addr"),
        (image.tags_addr, "tags_addr"),
    ):
        if not 0 <= addr <= 0xFFFFFFFF:
            raise ImageValueError(f"{what} {addr} does not fit in a 32-bit field")

    # dtb_addr is the one 64-bit header field; capping it at 32 bits would reject
    # images that parse() accepts, breaking the parse/pack round trip.
    if not 0 <= image.dtb_addr <= 0xFFFFFFFFFFFFFFFF:
        raise ImageValueError(
            f"dtb_addr {image.dtb_addr} does not fit in a 64-bit field"
        )

    ramdisk = image.vendor_ramdisk
    if image.fragments:
        # The fragment list is authoritative; rebuild the flat region from it so
        # the table offsets and vendor_ramdisk_size stay consistent.
        ramdisk = b"".join(fragment.data for fragment in image.fragments)

    out = bytearray()
    out += MAGIC
    out += _U32.pack(version)
    out += _U32.pack(page_size)
    out += _U32.pack(image.kernel_addr)
    out += _U32.pack(image.ramdisk_addr)
    out += _U32.pack(len(ramdisk))
    out += _pack_fixed_field(image.vendor_cmdline, CMDLINE_SIZE, "vendor_cmdline")
    out += _U32.pack(image.tags_addr)
    out += _pack_fixed_field(image.name, NAME_SIZE, "board name")
    out += _U32.pack(header_size)
    out += _U32.pack(len(image.dtb) if image.dtb else 0)
    out += _U64.pack(image.dtb_addr)

    table_entries: bytes = b""
    bootconfig_size = 0
    # The four v4-only words live past 0x850. A 2112-byte header has no room for
    # them, and such an image cannot describe fragments or a bootconfig either.
    if version >= 4 and header_size >= HEADER_SIZE_V4:
        if image.fragments:
            entry_size = 44 + 4 * BOARD_ID_SIZE
            table_entries = _pack_fragments(image.fragments, entry_size)
        else:
            entry_size = 0
        out += _U32.pack(len(table_entries))
        out += _U32.pack(len(image.fragments) if image.fragments else 0)
        out += _U32.pack(entry_size)
        bootconfig_size = len(image.bootconfig) if image.bootconfig else 0
        out += _U32.pack(bootconfig_size)
    elif image.fragments or image.bootconfig:
        raise ImageValueError(
            f"header_size {header_size} has no room for the header_version 4 "
            f"fragment/bootconfig fields; use {HEADER_SIZE_V4} or larger"
        )

    if len(out) > header_size:
        raise ImageValueError(
            f"header needs {len(out)} bytes but header_size is only {header_size}"
        )
    out += b"\x00" * (header_size - len(out))
    out += b"\x00" * (-len(out) % page_size)

    out += ramdisk
    out += b"\x00" * (-len(out) % page_size)

    if image.dtb:
        out += image.dtb
        out += b"\x00" * (-len(out) % page_size)

    if version >= 4:
        if table_entries:
            out += table_entries
            out += b"\x00" * (-len(out) % page_size)
        if image.bootconfig:
            out += image.bootconfig
            out += b"\x00" * (-len(out) % page_size)

    out += image.trailing
    return bytes(out)


def _pack_fragments(
    fragments: Sequence[VendorRamdiskFragment], entry_size: int
) -> bytes:
    table = bytearray()
    offset = 0
    for fragment in fragments:
        if not 0 <= fragment.ramdisk_type <= 0xFFFFFFFF:
            raise ImageValueError(
                f"ramdisk_type {fragment.ramdisk_type} does not fit in a 32-bit field"
            )
        if len(fragment.name) > 32:
            raise ImageValueError(
                f"fragment name {fragment.name!r} exceeds the 32-byte table field"
            )
        if len(fragment.board_id) > BOARD_ID_SIZE:
            raise ImageValueError(
                f"board_id has {len(fragment.board_id)} entries; "
                f"at most {BOARD_ID_SIZE} are allowed"
            )
        table += _U32.pack(fragment.size)
        table += _U32.pack(offset)
        table += _U32.pack(fragment.ramdisk_type)
        table += fragment.name.ljust(32, b"\x00")
        board_id = list(fragment.board_id) + [0] * (
            BOARD_ID_SIZE - len(fragment.board_id)
        )
        for value in board_id:
            if not 0 <= value <= 0xFFFFFFFF:
                raise ImageValueError(
                    f"board_id value {value} does not fit in a uint32"
                )
            table += _U32.pack(value)
        offset += fragment.size
    if len(table) != entry_size * len(fragments):
        raise ImageValueError(
            f"fragment table is {len(table)} bytes but {len(fragments)} entries of "
            f"{entry_size} bytes need {entry_size * len(fragments)}"
        )
    return bytes(table)
