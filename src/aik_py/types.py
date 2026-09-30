from dataclasses import dataclass, field
from enum import IntEnum

from aik_py.errors import CorruptImageError

MAGIC_BOOT = b"ANDROID!"
MAGIC_VENDOR = b"VNDRBOOT"

# Offsets below are verified empirically by packing real images with the mkbootimg
# bundled in AIK's bin/ tree and parsing the bytes. They deliberately do NOT match
# AOSP's documented layout: this build uses BOOT_NAME_SIZE=16 / BOOT_ARGS_SIZE=512,
# not AOSP's 512/1024, which shifts every field after cmdline.
BOOT_NAME_SIZE = 16
BOOT_ARGS_SIZE = 512
BOOT_ID_SIZE = 32

# v0 has no header_size field on the wire at all (mkbootimg reports none), so it
# gets no size constant. These are the sizes v1/v2/v3 actually report.
BOOT_HEADER_V1_SIZE = 1648
BOOT_HEADER_V2_SIZE = 1660
BOOT_HEADER_V3_SIZE = 1580

# Boot header v1/v2 trailing fields, verified on the wire: they sit past the id
# field rather than at the compact offsets AOSP documents.
BOOT_V1_RECOVERY_DTBO_SIZE = 0x660
BOOT_V1_RECOVERY_DTBO_ADDR = 0x664
BOOT_V1_HEADER_SIZE = 0x66C
BOOT_V1_DTB_SIZE = 0x670
BOOT_V1_DTB_ADDR = 0x674

# Boot v3/v4 is the sparse AOSP v3+ layout: name/cmdline/id are gone, the payload
# sizes sit up front and the only length reported is header_size.
BOOT_V3_KERNEL_SIZE = 0x08
BOOT_V3_RAMDISK_SIZE = 0x0C
BOOT_V3_HEADER_SIZE = 0x14
BOOT_V3_HEADER_VERSION = 0x28

# Vendor header offsets. This is the AOSP v3 vendor_boot header, and it is NOT the
# compact listing the boot header uses: cmdline and board name are trailing blobs
# with fixed maxima rather than sized fields, so the remaining numbers cluster at
# the end of the first page. Verified by packing --vendor_boot images and parsing.
VENDOR_HEADER_VERSION = 0x08
VENDOR_PAGE_SIZE = 0x0C
VENDOR_KERNEL_ADDR = 0x10
VENDOR_RAMDISK_ADDR = 0x14
VENDOR_VENDOR_RAMDISK_SIZE = 0x18
VENDOR_CMDLINE_OFFSET = 0x1C
VENDOR_TAGS_ADDR = 0x81C
VENDOR_NAME_OFFSET = 0x820
VENDOR_HEADER_SIZE = 0x830
VENDOR_DTB_SIZE = 0x834
VENDOR_DTB_ADDR = 0x838
VENDOR_HEADER_SIZE_VALUE = 2112
VENDOR_CMDLINE_MAX = 2048
VENDOR_NAME_MAX = 16


class HeaderVersion(IntEnum):
    V0 = 0
    V1 = 1
    V2 = 2
    V3 = 3
    V4 = 4


def _coerce_header_version(value: HeaderVersion | int, owner: str) -> HeaderVersion:
    try:
        return HeaderVersion(value)
    except ValueError:
        known = ", ".join(str(v.value) for v in HeaderVersion)
        raise CorruptImageError(
            f"{owner}: unknown header_version {value!r} (expected one of: {known})."
        ) from None


#: header_size reported on the wire per version. V0 is absent: that version has no
#: header_size field, and v3/v4 share the sparse struct.
HEADER_SIZE_BY_VERSION: dict[HeaderVersion, int] = {
    HeaderVersion.V1: BOOT_HEADER_V1_SIZE,
    HeaderVersion.V2: BOOT_HEADER_V2_SIZE,
    HeaderVersion.V3: BOOT_HEADER_V3_SIZE,
    HeaderVersion.V4: BOOT_HEADER_V3_SIZE,
}


class ImageType(IntEnum):
    """The ``imgtype`` tokens the AIK bash scripts write to ``*-imgtype``.

    Member names are valid Python; use :attr:`token` for the exact on-disk string,
    which the bash spells ``AOSP-PXA`` and ``U-Boot`` with hyphens. Matching the
    token matters: ``repackimg.sh`` dispatches on it and falls through to an abort
    for anything it does not recognise.
    """

    AOSP = 0
    AOSP_VNDR = 1
    AOSP_PXA = 2
    ELF = 3
    KRNL = 4
    OSIP = 5
    U_BOOT = 6

    @property
    def token(self) -> str:
        """Exact ``imgtype`` string as the bash scripts write and match it."""
        return _IMAGE_TYPE_TOKENS[self]


_IMAGE_TYPE_TOKENS: dict[ImageType, str] = {
    ImageType.AOSP: "AOSP",
    ImageType.AOSP_VNDR: "AOSP_VNDR",
    ImageType.AOSP_PXA: "AOSP-PXA",
    ImageType.ELF: "ELF",
    ImageType.KRNL: "KRNL",
    ImageType.OSIP: "OSIP",
    ImageType.U_BOOT: "U-Boot",
}


class Compression(IntEnum):
    """Ramdisk compression codecs, with the file extensions AIK's bash uses."""

    GZIP = 0
    BZIP2 = 1
    XZ = 2
    LZMA = 3
    LZ4 = 4
    LZ4_LEGACY = 5
    CPIO = 6
    LZOP = 7
    EMPTY = 8

    @property
    def extension(self) -> str:
        """File extension including the leading dot, or ``""`` when uncompressed.

        LZ4 and LZ4_LEGACY share ``.lz4``; the bash distinguishes them only by
        command-line flag (``-l`` for legacy), never by filename. EMPTY keeps
        ``.empty`` because that is the sentinel suffix the scripts write.
        """
        return _COMPRESSION_TABLE[self].extension

    @property
    def magic_name(self) -> str:
        """The ``ramdiskcomp`` token ``file(1)`` reports, which is what AIK matches on."""
        return _COMPRESSION_TABLE[self].magic_name


@dataclass(frozen=True, slots=True)
class _CompressionCodec:
    extension: str
    magic_name: str


# One row per codec so extension and magic name can never drift apart.
_COMPRESSION_TABLE: dict[Compression, _CompressionCodec] = {
    Compression.GZIP: _CompressionCodec(".gz", "gzip"),
    Compression.BZIP2: _CompressionCodec(".bz2", "bzip2"),
    Compression.XZ: _CompressionCodec(".xz", "xz"),
    Compression.LZMA: _CompressionCodec(".lzma", "lzma"),
    Compression.LZ4: _CompressionCodec(".lz4", "lz4"),
    Compression.LZ4_LEGACY: _CompressionCodec(".lz4", "lz4-l"),
    Compression.CPIO: _CompressionCodec("", "cpio"),
    Compression.LZOP: _CompressionCodec(".lzo", "lzop"),
    Compression.EMPTY: _CompressionCodec(".empty", "empty"),
}


@dataclass(slots=True)
class BootHeader:
    """The classic ``ANDROID!`` boot header (v0/v1/v2).

    Sizes and addresses are stored exactly as packed; no unit conversion happens
    here, so ``kernel_size`` is bytes and ``kernel_addr`` is the raw load address
    from the image. ``os_version`` is the packed AOSP bitfield, not a version tuple.

    The v1/v2 fields beyond ``id`` (recovery_dtbo_*, header_size, dtb_*) live in a
    second page rather than in the 16-byte-name layout above; see the
    ``BOOT_V1_*`` offset constants. v0 omits them entirely, and v3/v4 drop
    ``name``/``cmdline``/``id`` for the sparse AOSP v3+ struct.
    """

    kernel_size: int = 0
    kernel_addr: int = 0
    ramdisk_size: int = 0
    ramdisk_addr: int = 0
    second_size: int = 0
    second_addr: int = 0
    tags_addr: int = 0
    page_size: int = 0
    header_version: HeaderVersion = HeaderVersion.V0
    os_version: int = 0
    name: str = ""
    cmdline: str = ""
    id: str = ""
    dtb_size: int = 0
    dtb_addr: int = 0
    recovery_dtbo_size: int = 0
    recovery_dtbo_addr: int = 0
    # v0 has no header_size on the wire, so 0 means "not present".
    header_size: int = 0

    def __post_init__(self) -> None:
        self.header_version = _coerce_header_version(self.header_version, "BootHeader")
        if self.header_size == 0:
            self.header_size = HEADER_SIZE_BY_VERSION.get(self.header_version, 0)


@dataclass(slots=True)
class VendorBootHeader:
    """The ``VNDRBOOT`` header used by Android 9+ vendor_boot images.

    ``cmdline_size`` and ``name_size`` are not stored on the wire in this layout:
    the cmdline runs from 0x1C to the tags field and the board name sits at 0x820,
    both bounded by fixed maxima. They are kept as explicit fields so callers can
    carry the real lengths once the blobs are split out.
    """

    header_version: HeaderVersion = HeaderVersion.V3
    page_size: int = 0
    kernel_addr: int = 0
    ramdisk_addr: int = 0
    vendor_ramdisk_size: int = 0
    cmdline_size: int = 0
    tags_addr: int = 0
    name_size: int = VENDOR_NAME_MAX
    header_size: int = VENDOR_HEADER_SIZE_VALUE
    dtb_size: int = 0
    dtb_addr: int = 0

    def __post_init__(self) -> None:
        self.header_version = _coerce_header_version(
            self.header_version, "VendorBootHeader"
        )


@dataclass(slots=True)
class BootImage:
    """A parsed boot/recovery image: a header plus its payload blobs.

    Every payload is ``None`` when the image carries no such section, which is the
    common case for the second-stage ``bootstub`` and for vendor images missing a
    recovery dtbo.
    """

    header: BootHeader = field(default_factory=BootHeader)
    kernel: bytes | None = None
    ramdisk: bytes | None = None
    second: bytes | None = None
    dtb: bytes | None = None
    recovery_dtbo: bytes | None = None
    signature: bytes | None = None
    image_type: ImageType = ImageType.AOSP


@dataclass(slots=True)
class VendorBootImage:
    """A parsed ``VNDRBOOT`` image: header plus vendor payload blobs."""

    header: VendorBootHeader = field(default_factory=VendorBootHeader)
    vendor_ramdisk: bytes | None = None
    dtb: bytes | None = None
    cmdline: str = ""
    image_type: ImageType = ImageType.AOSP_VNDR
