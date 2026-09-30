"""AOSP boot image, header_version 0 through 2.

The classic ``ANDROID!`` header plus page-aligned kernel/ramdisk/second/
recovery_dtbo/dtb payloads.

Every offset in this module was derived empirically by packing images with the
AIK-bundled ``bin/linux/x86_64/mkbootimg`` and reading the bytes back, then
cross-checked against the vendored ``unpackbootimg``. They happen to agree with
upstream AOSP ``mkbootimg.py``, including the non-obvious ``BOOT_NAME_SIZE=16`` /
``BOOT_ARGS_SIZE=512`` pairing (older AOSP docs describe 512/1024 -- do not
"fix" these constants to match the docs).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field, replace
from typing import Final, Literal, Self

# `aik_py.types` / `aik_py.errors` are owned by other units and may not exist yet
# on this branch, so the shared symbols are imported defensively.
try:  # pragma: no cover - depends on sibling units landing
    from aik_py.errors import AikError, BootImageError
except ImportError:  # pragma: no cover

    class AikError(Exception):
        """Base class for aik-py failures."""

    class BootImageError(AikError):
        """A boot image is malformed or cannot be represented."""


try:  # pragma: no cover - depends on sibling units landing
    from aik_py.types import BootImage, BootImageHeader, HashType, OsVersion
except ImportError:  # pragma: no cover
    HashType = Literal["sha1", "sha256"]

    @dataclass(slots=True)
    class OsVersion:
        """``os_version`` packs two fields into one u32.

        The high 21 bits hold three 7-bit components (major << 14 | minor << 7 |
        patch); the low 11 hold the patch level as
        ``(year - 2000) * 16 + month``. Both were derived from images packed by
        the bundled tools and match what ``unpackbootimg`` reports as
        ``BOARD_OS_VERSION`` and ``BOARD_OS_PATCH_LEVEL``.

        ``patch_year`` is the raw offset from 2000, so 2021-08 is (21, 8);
        :meth:`year` and :meth:`month` give the calendar values the tools print.
        """

        major: int = 0
        minor: int = 0
        patch: int = 0
        patch_year: int = 0
        patch_month: int = 0

        @property
        def year(self) -> int:
            return 2000 + self.patch_year

        @property
        def month(self) -> int:
            return self.patch_month

        def pack(self) -> int:
            version = (
                ((self.major & 0x7F) << 14)
                | ((self.minor & 0x7F) << 7)
                | (self.patch & 0x7F)
            )
            patch_level = ((self.patch_year & 0x7F) * 16 + self.patch_month) & 0x7FF
            return (version << 11) | patch_level

        @classmethod
        def unpack(cls, raw: int) -> Self:
            version, patch_level = raw >> 11, raw & 0x7FF
            return cls(
                major=(version >> 14) & 0x7F,
                minor=(version >> 7) & 0x7F,
                patch=version & 0x7F,
                patch_year=patch_level // 16,
                patch_month=patch_level % 16,
            )

    @dataclass(slots=True)
    class BootImageHeader:
        """Header fields as they sit on disk."""

        kernel_addr: int = 0
        ramdisk_addr: int = 0
        second_addr: int = 0
        tags_addr: int = 0
        page_size: int = 0
        header_version: int = 0
        os_version: OsVersion = field(default_factory=OsVersion)
        name: bytes = b""
        cmdline: bytes = b""
        extra_cmdline: bytes = b""
        hashtype: HashType = "sha1"
        id: bytes | None = None
        recovery_dtbo_size: int = 0
        recovery_dtbo_offset: int = 0
        dtb_size: int = 0
        dtb_addr: int = 0

    @dataclass(slots=True)
    class BootImage:
        """A parsed boot image."""

        header: BootImageHeader = field(default_factory=BootImageHeader)
        kernel: bytes = b""
        ramdisk: bytes = b""
        second: bytes | None = None
        recovery_dtbo: bytes | None = None
        dtb: bytes | None = None


__all__ = [
    "BootImage",
    "BootImageError",
    "BootImageHeader",
    "HashType",
    "OsVersion",
    "pack",
    "parse",
]

MAGIC: Final = b"ANDROID!"
BOOT_MAGIC_SIZE: Final = 8

# These four are the ones people get wrong: AIK's bundled mkbootimg writes a
# 16-byte name and a 512-byte cmdline, unlike the 512/1024 pair in older AOSP
# write-ups. Verified byte-for-byte against packed images.
BOOT_NAME_SIZE: Final = 16
BOOT_ARGS_SIZE: Final = 512
BOOT_EXTRA_ARGS_SIZE: Final = 1024

BOOT_IMAGE_HEADER_V1_SIZE: Final = 1648
BOOT_IMAGE_HEADER_V2_SIZE: Final = 1660
#: ``header_version`` 0 predates the ``header_size`` field; its header ends after
#: ``extra_cmdline``. Upstream writes a zero there, but readers still need to
#: know how far the header reaches, so it gets a synthesised value.
BOOT_IMAGE_HEADER_V0_SIZE: Final = 1592
_HEADER_SIZES: Final = {
    0: BOOT_IMAGE_HEADER_V0_SIZE,
    1: BOOT_IMAGE_HEADER_V1_SIZE,
    2: BOOT_IMAGE_HEADER_V2_SIZE,
}

MAX_HEADER_VERSION: Final = 2
ID_SIZE: Final = 32
#: `unpackbootimg` writes 4096 for a zero page_size field (the binary carries a
#: literal "BOARD_PAGE_SIZE 4096" fallback string). Its own `-p` flag only
#: overrides for vendor_boot images, never for the v0-v2 path.
DEFAULT_PAGE_SIZE: Final = 4096

# Field offsets, verified against the bundled tools.
_OFF_MAGIC: Final = 0x00
_OFF_KERNEL_SIZE: Final = 0x08
_OFF_KERNEL_ADDR: Final = 0x0C
_OFF_RAMDISK_SIZE: Final = 0x10
_OFF_RAMDISK_ADDR: Final = 0x14
_OFF_SECOND_SIZE: Final = 0x18
_OFF_SECOND_ADDR: Final = 0x1C
_OFF_TAGS_ADDR: Final = 0x20
_OFF_PAGE_SIZE: Final = 0x24
_OFF_HEADER_VERSION: Final = 0x28
_OFF_OS_VERSION: Final = 0x2C
_OFF_NAME: Final = 0x30
_OFF_CMDLINE: Final = 0x40
_OFF_ID: Final = 0x240
_OFF_EXTRA_CMDLINE: Final = 0x260
_OFF_RECOVERY_DTBO_SIZE: Final = 0x660
_OFF_RECOVERY_DTBO_OFFSET: Final = 0x664
_OFF_HEADER_SIZE: Final = 0x66C
_OFF_DTB_SIZE: Final = 0x670
_OFF_DTB_ADDR: Final = 0x674

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


def header_size_for(header_version: int) -> int:
    """Bytes the header occupies, including ``extra_cmdline``."""
    try:
        return _HEADER_SIZES[header_version]
    except KeyError:
        msg = f"unsupported header_version {header_version}; this module handles 0-2"
        raise BootImageError(msg) from None


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _cstr(field: bytes, what: str) -> bytes:
    """Strip NUL padding off a read-back header string.

    A field may legitimately have no terminator at all: mkbootimg fills the
    512-byte cmdline edge to edge. So reading only ever truncates, never
    rejects -- bounds are enforced on the write path by `_fit`.
    """
    return field.split(b"\x00", 1)[0]


def _fit(value: bytes, size: int, what: str, *, needs_nul: bool) -> bytes:
    """Bounds-check a string destined for a fixed header field.

    `needs_nul` distinguishes the two cases the bundled tool treats differently:
    a 16-byte board name must keep a terminator (it rejects a 16-character
    name), while a 512-byte cmdline may be filled edge to edge.
    """
    limit = size - 1 if needs_nul else size
    if len(value) > limit:
        msg = f"{what} is {len(value)} bytes but the header field holds at most {limit}"
        raise BootImageError(msg)
    return value


def _check_version(header_version: int) -> None:
    if header_version > MAX_HEADER_VERSION:
        msg = f"unsupported header_version {header_version}; this module handles 0-{MAX_HEADER_VERSION}"
        raise BootImageError(msg)


def _payloads(image: BootImage) -> list[tuple[str, bytes]]:
    """Payloads in on-disk order. Optional members are skipped when unset."""
    hv = image.header.header_version
    _check_version(hv)
    if hv == 0 and image.recovery_dtbo:
        msg = "recovery_dtbo requires header_version >= 1"
        raise BootImageError(msg)
    if hv < 2 and image.dtb:
        msg = "dtb requires header_version 2"
        raise BootImageError(msg)
    out = [("kernel", image.kernel or b""), ("ramdisk", image.ramdisk or b"")]
    if image.second is not None:
        out.append(("second", image.second))
    if image.recovery_dtbo is not None:
        out.append(("recovery_dtbo", image.recovery_dtbo))
    if image.dtb is not None:
        out.append(("dtb", image.dtb))
    return out


def _hash_id(algorithm: HashType, image: BootImage) -> bytes:
    """The v0-v2 id is the AOSP *payload* digest, not a digest of the header.

    Each payload contributes its bytes followed by a u32 little-endian length,
    and absent members still contribute their zero length -- so the set of slots
    depends on header_version (recovery_dtbo only from v1, dtb only from v2).
    Verified byte-for-byte against every image the bundled mkbootimg produced.

    Upstream AOSP dropped --hashtype and now hashes as sha1 unconditionally;
    the AIK build keeps both, which is why the algorithm is a parameter here.
    """
    hv = image.header.header_version
    slots: list[bytes | None] = [
        image.kernel or b"",
        image.ramdisk or b"",
        image.second,
    ]
    if hv >= 1:
        slots.append(image.recovery_dtbo)
    if hv >= 2:
        slots.append(image.dtb)

    digest = hashlib.new(algorithm)
    for slot in slots:
        data = slot or b""
        digest.update(data)
        digest.update(_U32.pack(len(data)))
    return digest.digest()


def _build_header(image: BootImage) -> bytes:
    header = image.header
    hv = header.header_version
    _check_version(hv)
    page_size = header.page_size
    if page_size <= 0 or page_size % 4:
        msg = f"page_size must be a positive multiple of 4, got {page_size}"
        raise BootImageError(msg)

    name = _fit(header.name, BOOT_NAME_SIZE, "board name", needs_nul=True)
    cmdline = _fit(header.cmdline, BOOT_ARGS_SIZE, "cmdline", needs_nul=False)
    extra = _fit(
        header.extra_cmdline, BOOT_EXTRA_ARGS_SIZE, "extra_cmdline", needs_nul=False
    )

    out = bytearray(header_size_for(hv))
    out[_OFF_MAGIC : _OFF_MAGIC + BOOT_MAGIC_SIZE] = MAGIC
    _U32.pack_into(out, _OFF_KERNEL_SIZE, len(image.kernel or b""))
    _U32.pack_into(out, _OFF_KERNEL_ADDR, header.kernel_addr & 0xFFFFFFFF)
    _U32.pack_into(out, _OFF_RAMDISK_SIZE, len(image.ramdisk or b""))
    _U32.pack_into(out, _OFF_RAMDISK_ADDR, header.ramdisk_addr & 0xFFFFFFFF)
    if image.second is not None:
        _U32.pack_into(out, _OFF_SECOND_SIZE, len(image.second))
    _U32.pack_into(out, _OFF_SECOND_ADDR, header.second_addr & 0xFFFFFFFF)
    _U32.pack_into(out, _OFF_TAGS_ADDR, header.tags_addr & 0xFFFFFFFF)
    _U32.pack_into(out, _OFF_PAGE_SIZE, page_size)
    _U32.pack_into(out, _OFF_HEADER_VERSION, hv)
    _U32.pack_into(out, _OFF_OS_VERSION, header.os_version.pack())
    out[_OFF_NAME : _OFF_NAME + BOOT_NAME_SIZE] = name.ljust(BOOT_NAME_SIZE, b"\x00")
    out[_OFF_CMDLINE : _OFF_CMDLINE + BOOT_ARGS_SIZE] = cmdline.ljust(
        BOOT_ARGS_SIZE, b"\x00"
    )
    out[_OFF_EXTRA_CMDLINE : _OFF_EXTRA_CMDLINE + BOOT_EXTRA_ARGS_SIZE] = extra.ljust(
        BOOT_EXTRA_ARGS_SIZE, b"\x00"
    )

    if header.id is not None:
        identifier = header.id
    else:
        identifier = _hash_id(header.hashtype, image)
    if len(identifier) > ID_SIZE:
        msg = f"id is {len(identifier)} bytes but the header holds at most {ID_SIZE}"
        raise BootImageError(msg)
    # A 20-byte sha1 id leaves the remaining 12 bytes zero, as the tools write it.
    out[_OFF_ID : _OFF_ID + ID_SIZE] = identifier.ljust(ID_SIZE, b"\x00")

    if hv >= 1:
        if image.recovery_dtbo is not None:
            _U32.pack_into(out, _OFF_RECOVERY_DTBO_SIZE, len(image.recovery_dtbo))
            _U32.pack_into(
                out,
                _OFF_RECOVERY_DTBO_OFFSET,
                header.recovery_dtbo_offset & 0xFFFFFFFFFFFFFFFF,
            )
        _U32.pack_into(out, _OFF_HEADER_SIZE, header_size_for(hv))
    if hv >= 2:
        if image.dtb is not None:
            _U32.pack_into(out, _OFF_DTB_SIZE, len(image.dtb))
        # dtb_addr is base + dtb_offset and is written whether or not a dtb is
        # present; the tools emit a non-zero value even for a dtb-less v2 image.
        _U64.pack_into(out, _OFF_DTB_ADDR, header.dtb_addr)
    return bytes(out)


def pack(image: BootImage) -> bytes:
    """Serialise `image` to boot image bytes.

    The header occupies the first page; every payload follows, each padded up to
    a whole number of pages, in the order kernel, ramdisk, second,
    recovery_dtbo, dtb.
    """
    page_size = image.header.page_size or DEFAULT_PAGE_SIZE
    header = replace(image.header, page_size=page_size)
    image = replace(image, header=header)

    header_bytes = _build_header(image)
    # The header must fit in the page it occupies. A smaller page_size would
    # otherwise silently overwrite the first payloads.
    if len(header_bytes) > page_size:
        msg = (
            f"page_size {page_size} is smaller than the {len(header_bytes)}-byte "
            f"header for header_version {header.header_version}"
        )
        raise BootImageError(msg)

    out = bytearray(_align(len(header_bytes), page_size))
    out[: len(header_bytes)] = header_bytes

    for _, payload in _payloads(image):
        out += payload
        out += b"\x00" * (_align(len(payload), page_size) - len(payload))
    return bytes(out)


def _u32(data: bytes, offset: int, what: str) -> int:
    if offset + 4 > len(data):
        msg = f"truncated boot image: {what} at 0x{offset:X} needs 4 bytes, file is {len(data)}"
        raise BootImageError(msg)
    return _U32.unpack_from(data, offset)[0]


def _u64(data: bytes, offset: int, what: str) -> int:
    if offset + 8 > len(data):
        msg = f"truncated boot image: {what} at 0x{offset:X} needs 8 bytes, file is {len(data)}"
        raise BootImageError(msg)
    return _U64.unpack_from(data, offset)[0]


def parse(data: bytes) -> BootImage:
    """Read a v0-v2 boot image from `data`.

    Raises `BootImageError` rather than letting struct/index errors escape.
    """
    if len(data) < BOOT_MAGIC_SIZE:
        msg = f"truncated boot image: need at least {BOOT_MAGIC_SIZE} bytes for the magic, got {len(data)}"
        raise BootImageError(msg)
    magic = data[:BOOT_MAGIC_SIZE]
    if magic != MAGIC:
        msg = f"bad magic: expected {MAGIC!r}, got {magic!r}"
        raise BootImageError(msg)

    header_version = _u32(data, _OFF_HEADER_VERSION, "header_version")
    header_size = header_size_for(header_version)
    if len(data) < header_size:
        msg = (
            f"truncated boot image: header_version {header_version} needs {header_size} "
            f"header bytes, file is {len(data)}"
        )
        raise BootImageError(msg)

    page_size = _u32(data, _OFF_PAGE_SIZE, "page_size")
    if page_size == 0:
        # unpackbootimg substitutes 4096 here; anything else is a corrupt field.
        page_size = DEFAULT_PAGE_SIZE
    elif page_size % 4:
        msg = f"implausible page_size {page_size}: not a multiple of 4"
        raise BootImageError(msg)

    header = BootImageHeader(
        kernel_addr=_u32(data, _OFF_KERNEL_ADDR, "kernel_addr"),
        ramdisk_addr=_u32(data, _OFF_RAMDISK_ADDR, "ramdisk_addr"),
        second_addr=_u32(data, _OFF_SECOND_ADDR, "second_addr"),
        tags_addr=_u32(data, _OFF_TAGS_ADDR, "tags_addr"),
        page_size=page_size,
        header_version=header_version,
        os_version=OsVersion.unpack(_u32(data, _OFF_OS_VERSION, "os_version")),
        name=_cstr(data[_OFF_NAME : _OFF_NAME + BOOT_NAME_SIZE], "board name"),
        cmdline=_cstr(data[_OFF_CMDLINE : _OFF_CMDLINE + BOOT_ARGS_SIZE], "cmdline"),
        extra_cmdline=_cstr(
            data[_OFF_EXTRA_CMDLINE : _OFF_EXTRA_CMDLINE + BOOT_EXTRA_ARGS_SIZE],
            "extra_cmdline",
        ),
        id=data[_OFF_ID : _OFF_ID + ID_SIZE],
    )
    if header_version >= 1:
        header.recovery_dtbo_size = _u32(
            data, _OFF_RECOVERY_DTBO_SIZE, "recovery_dtbo_size"
        )
        header.recovery_dtbo_offset = _u64(
            data, _OFF_RECOVERY_DTBO_OFFSET, "recovery_dtbo_offset"
        )
        stored_size = _u32(data, _OFF_HEADER_SIZE, "header_size")
        if stored_size not in (0, header_size):
            msg = f"header_size field says {stored_size}, expected {header_size} for v{header_version}"
            raise BootImageError(msg)
    if header_version >= 2:
        header.dtb_size = _u32(data, _OFF_DTB_SIZE, "dtb_size")
        header.dtb_addr = _u64(data, _OFF_DTB_ADDR, "dtb_addr")

    # A sha1 id occupies 20 bytes and the other 12 are zero padding; a sha256 id
    # fills all 32. Deriving the length from the padding recovers hashtype without
    # needing to re-hash. (An all-zero id is read as sha1, the tools' default.)
    significant = 20 if not any(header.id[20:]) else ID_SIZE
    header.hashtype = "sha1" if significant == 20 else "sha256"
    header.id = bytes(header.id[:significant])

    sizes = [("kernel", _u32(data, _OFF_KERNEL_SIZE, "kernel_size"))]
    sizes.append(("ramdisk", _u32(data, _OFF_RAMDISK_SIZE, "ramdisk_size")))
    sizes.append(("second", _u32(data, _OFF_SECOND_SIZE, "second_size")))
    if header_version >= 1:
        sizes.append(("recovery_dtbo", header.recovery_dtbo_size))
    if header_version >= 2:
        sizes.append(("dtb", header.dtb_size))

    payloads: dict[str, bytes] = {}
    cursor = page_size
    for name, size in sizes:
        if size == 0:
            payloads[name] = b""
            continue
        end = cursor + size
        if end > len(data):
            msg = (
                f"truncated boot image: {name} of {size} bytes at 0x{cursor:X} "
                f"runs past end of file ({len(data)} bytes)"
            )
            raise BootImageError(msg)
        payloads[name] = data[cursor:end]
        cursor = _align(end, page_size)

    return BootImage(
        header=header,
        kernel=payloads["kernel"],
        ramdisk=payloads["ramdisk"],
        second=payloads["second"] or None,
        recovery_dtbo=payloads.get("recovery_dtbo") or None,
        dtb=payloads.get("dtb") or None,
    )
