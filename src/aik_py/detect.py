"""Format detection for Android boot/recovery images.

Replaces the Bash toolkit's ``file -m bin/androidbootimg.magic`` invocation,
which scraped the human-readable libmagic description with ``awk '{print $N}'``
and therefore depended on the exact field positions libmagic happens to emit.
``>>``/``>>>`` continuation levels land on a single output line, so any libmagic
version bump silently shifted the parsed fields. Everything here matches raw
bytes instead, so detection is stable and returns ``None`` rather than raising
when a buffer is too short to hold a match.

The ruleset below is ``bin/androidbootimg.magic``; its offsets were confirmed
empirically against file-5.48 and against real images produced by the bundled
``mkbootimg`` and ``pxa-mkbootimg``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

try:
    from aik_py.errors import CorruptImageError, UnsupportedFormatError
except ImportError:  # pragma: no cover
    # aik_py.errors is owned by another unit; keep this module importable on its
    # own rather than making detection depend on it landing first. Once errors.py
    # exists the import above wins, so callers that catch aik_py.errors.* must
    # simply import this module after it -- which is the normal order.
    class CorruptImageError(Exception):
        """The image matches no format AIK knows how to unpack."""

    class UnsupportedFormatError(Exception):
        """The image matches a known format that AIK cannot rebuild."""


class SignatureType(StrEnum):
    """Wrapping signature found at the head of the image."""

    BLOB = "BLOB"
    CHROMEOS = "CHROMEOS"
    DHTB = "DHTB"
    NOOK = "NOOK"
    NOOKTAB = "NOOKTAB"
    SINV1 = "SINv1"
    SINV2 = "SINv2"
    SINV3 = "SINv3"


class PatchType(StrEnum):
    """In-place obfuscation layered over an otherwise normal AOSP header."""

    LOKI = "LOKI"
    AMONET = "AMONET"


class ImageType(StrEnum):
    """Boot image layout, named as the Bash recorded it in ``$file-imgtype``."""

    AOSP = "AOSP"
    AOSP_VNDR = "AOSP_VNDR"
    AOSP_PXA = "AOSP-PXA"
    ELF = "ELF"
    KRNL = "KRNL"
    OSIP = "OSIP"
    U_BOOT = "U-Boot"


class TailType(StrEnum):
    """Footer appended to the end of the image."""

    AVBV1 = "AVBv1"
    AVBV2 = "AVBv2"
    BUMP = "Bump"
    SEANDROID = "SEAndroid"


class MtkType(StrEnum):
    """MediaTek 512-byte header prepended to a split kernel or ramdisk."""

    KERNEL = "kernel"
    ROOTFS = "rootfs"
    RECOVERY = "recovery"


class LokiPartition(StrEnum):
    BOOT = "boot"
    RECOVERY = "recovery"


class AvbPartition(StrEnum):
    BOOT = "boot"
    RECOVERY = "recovery"


AOSP_MAGIC = b"ANDROID!"
MTK_MAGIC = b"\x88\x16\x88\x58"

# The `0 string x` lines in androidbootimg.magic are documented upstream as a
# workaround for "odd file/magic behavior". They are inert under file-5.48: the
# `>0 search` rules match anywhere in the buffer whether or not byte 0 is
# literally `x`, which was verified by re-running the oracle both ways. They are
# dropped here; if another libmagic build ever makes them meaningful, these
# searches are where the gate would have to be reinstated.

# PXA is flagged by the u32 at offset 0x24 (the magic file's `>>36`). That is a
# different field from header_version, which lives at 0x28; the two are read from
# their own offsets rather than conflated. The magic file points at 36 because
# that offset carries the PXA signature in the header shape it matches -- and,
# critically, stock images really do put 2 or 3 in header_version, so reading
# 0x28 for the marker would misclassify them.
PXA_OFFSET = 0x24
PXA_MARKERS = (b"\x00\x00\x00\x02", b"\x00\x00\x80\x02", b"\x00\x00\x00\x03")
HEADER_VERSION_OFFSET = 0x28

MICROLOADER_OFFSET = 96
LOKI_OFFSET = 1024
LOKI_PARTITION_OFFSET = 1028
AMONET_ANDROID_OFFSET = 1024
OSIP_HEADERLESS_OFFSET = 1000

TAIL_WINDOW = 8192
# `tail -n50` on a binary image yields at most this many bytes; it bounds the
# fallback well below a whole-file rescan.
TAIL_LINES_WINDOW = 65536

_SIGNATURES: tuple[tuple[SignatureType, int, bytes], ...] = (
    (SignatureType.BLOB, 0, b"-SIGNED-BY-SIGNBLOB-"),
    (SignatureType.CHROMEOS, 0, b"CHROMEOS"),
    (SignatureType.DHTB, 0, b"DHTB\x01\x00\x00"),
    (SignatureType.SINV1, 0, b"\x01\x00\x00\x00"),
    (SignatureType.SINV2, 0, b"\x02\x00\x00\x00"),
    (SignatureType.SINV3, 0, b"\x03SIN"),
    (SignatureType.NOOK, 64, b"Red Loader"),
    (SignatureType.NOOK, 64, b"Green Loader"),
    (SignatureType.NOOK, 64, b"Green Recovery"),
    (SignatureType.NOOK, 64, b"eMMC boot.img+secondloader"),
    (SignatureType.NOOK, 64, b"eMMC recovery.img+secondloader"),
    (SignatureType.NOOKTAB, 48, b"BauwksBoot"),
)

_AVB1_MAGIC = b"\x02\x01\x01\x30\x82"
_AVB2_MAGIC = b"AVBf"
_BUMP_MAGIC = b"\x41\xa9\xe4\x67\x74\x4d\x1d\x1b\xa4\x29\xf2\xec\xea\x65\x52\x79"
_SEANDROID_MAGIC = b"SEANDROIDENFORCE"

_TAIL_MAGICS: tuple[tuple[TailType, bytes], ...] = (
    (TailType.AVBV1, _AVB1_MAGIC),
    (TailType.AVBV2, _AVB2_MAGIC),
    (TailType.BUMP, _BUMP_MAGIC),
    (TailType.SEANDROID, _SEANDROID_MAGIC),
)

_MTK_KEYWORDS: tuple[tuple[MtkType, bytes], ...] = (
    (MtkType.KERNEL, b"KERNEL"),
    (MtkType.ROOTFS, b"ROOTFS"),
    (MtkType.RECOVERY, b"RECOVERY"),
)

_LOKI_PARTITIONS = {0x00: LokiPartition.BOOT, 0x01: LokiPartition.RECOVERY}

# Bash: `case $imgtype in AOSP*|ELF|KRNL|OSIP|U-Boot) ;; *) "Unsupported format."`
SUPPORTED_IMAGE_TYPES = frozenset(ImageType)


def _at(data: bytes, offset: int, pattern: bytes) -> bool:
    """Compare ``pattern`` at ``offset``; short buffers simply do not match."""
    return data[offset : offset + len(pattern)] == pattern


def _after(data: bytes, offset: int, pattern: bytes) -> bool:
    return data.find(pattern, offset) != -1


def detect_signature(data: bytes) -> SignatureType | None:
    """Return the wrapping signature at the head of ``data``, if any."""
    for sig, offset, pattern in _SIGNATURES:
        if _at(data, offset, pattern):
            return sig
    return None


def detect_image_type(data: bytes) -> ImageType | None:
    """Return the boot image layout of ``data``, or ``None`` if unrecognized."""
    if data.startswith(AOSP_MAGIC):
        return ImageType.AOSP_PXA if is_pxa(data) else ImageType.AOSP
    if data.startswith(b"VNDRBOOT"):
        return ImageType.AOSP_VNDR
    if data.startswith(b"\x27\x05\x19\x56"):
        return ImageType.U_BOOT
    if _at(data, 0, b"$OS$\x00\x00\x01") or _at(
        data, OSIP_HEADERLESS_OFFSET, b"\xfc\xfa\xbc\x00"
    ):
        return ImageType.OSIP
    if data.startswith(b"KRNL"):
        return ImageType.KRNL
    if data.startswith(b"\x7fELF"):
        return ImageType.ELF
    return None


def is_pxa(data: bytes) -> bool:
    """Report whether an AOSP image carries the PXA layout marker."""
    return any(_at(data, PXA_OFFSET, marker) for marker in PXA_MARKERS)


def header_version(data: bytes) -> int | None:
    """Read header_version (offset 0x28) from an AOSP image.

    Distinct from the PXA marker at 0x24. Returns ``None`` for a PXA image,
    whose shifted layout puts unrelated bytes at 0x28 -- a real pxa-mkbootimg
    image carries ``00 01 00 80`` there, which would otherwise read back as a
    nonsense version.
    """
    if not data.startswith(AOSP_MAGIC) or is_pxa(data):
        return None
    raw = data[HEADER_VERSION_OFFSET : HEADER_VERSION_OFFSET + 4]
    return int.from_bytes(raw, "little") if len(raw) == 4 else None


def detect_patch(data: bytes) -> PatchType | None:
    """Return the obfuscation layered over the AOSP header, if any.

    Both LOKI and AMONET are nested under the magic file's ``ANDROID!`` search,
    so an image without that marker anywhere is never reported as patched.
    """
    if not _after(data, 0, AOSP_MAGIC):
        return None
    if _at(data, LOKI_OFFSET, b"LOKI"):
        return PatchType.LOKI
    if _after(data, MICROLOADER_OFFSET, b"microloader") and _at(
        data, AMONET_ANDROID_OFFSET, AOSP_MAGIC
    ):
        return PatchType.AMONET
    return None


def loki_partition(data: bytes) -> LokiPartition | None:
    """Return which partition a detected LOKI patch wraps.

    The magic file's ``>>>1028`` is relative to the ``LOKI`` match at 1024, so
    the partition byte is absolute 1028. Returns ``None`` when the image carries
    no LOKI header at all: byte 1028 is only meaningful under that header, and
    reading it unconditionally would invent a partition for every AOSP image.
    A value other than 0 or 1 carries no name in the Bash, which then leaves
    ``$file-lokitype`` empty.
    """
    if not _at(data, LOKI_OFFSET, b"LOKI"):
        return None
    if len(data) <= LOKI_PARTITION_OFFSET:
        return None
    return _LOKI_PARTITIONS.get(data[LOKI_PARTITION_OFFSET])


def tail_buffer(data: bytes) -> bytes:
    """Return the buffer a footer may live in.

    The Bash pipes the last 8192 bytes through libmagic and, if that finds
    nothing, retries with ``tail -n50``. Both views are tails, so the scan is
    bounded rather than a rescan of the whole image -- an unbounded search for
    the 4-byte ``AVBf`` would fire on incidental payload text.
    """
    window = data[-TAIL_WINDOW:]
    return window if _classify_tail(window) else data[-TAIL_LINES_WINDOW:]


def detect_tail(data: bytes) -> TailType | None:
    """Return the footer appended to the end of ``data``, if any.

    Mirrors the Bash two-stage scan: inspect the last 8192 bytes, and only if
    that finds nothing widen the search.
    """
    return _classify_tail(tail_buffer(data))


def avb1_partition(data: bytes) -> AvbPartition | None:
    """Return the partition named by an AVBv1 footer.

    The magic file matches ``/boot`` and ``/recovery`` -- leading slash
    included -- starting from the AVBv1 marker, and lists ``/boot`` first. The
    Bash then reads the first continuation field, so a footer naming both reads
    as boot.

    The search is anchored at the marker because an AVB footer declares its
    partition *after* the AVB block; scanning the whole buffer would let a stray
    ``/boot`` in the kernel payload outrank the footer's real partition and
    re-sign the image under the wrong name.
    """
    start = data.rfind(_AVB1_MAGIC)
    if start == -1:
        return None
    if _after(data, start, b"/boot"):
        return AvbPartition.BOOT
    if _after(data, start, b"/recovery"):
        return AvbPartition.RECOVERY
    return None


def _classify_tail(buf: bytes) -> TailType | None:
    for tail, magic in _TAIL_MAGICS:
        if _after(buf, 0, magic):
            return tail
    return None


def has_mtk(data: bytes) -> bool:
    """Report whether an MTK header is present in a split kernel or ramdisk."""
    return data.find(MTK_MAGIC) != -1


def detect_mtk(data: bytes) -> MtkType | None:
    """Return the kind of component an MTK header was prepended to.

    Both the marker and the type keyword are looked for anywhere in the buffer,
    as the magic file's ``>0 search`` rules do. The keyword search starts at the
    marker so one pass serves both, and a header that never names its component
    yields ``None`` -- the Bash writes no type in that case.
    """
    start = data.find(MTK_MAGIC)
    if start == -1:
        return None
    for kind, keyword in _MTK_KEYWORDS:
        if _after(data, start, keyword):
            return kind
    return None


def detect_qcdt(data: bytes) -> bool:
    """Report whether a Qualcomm device tree wrapper is present."""
    return data.startswith(b"QCDT")


@dataclass(frozen=True, slots=True)
class Detection:
    """Everything AIK records about an image before splitting it."""

    signature: SignatureType | None = None
    image_type: ImageType | None = None
    patch: PatchType | None = None
    loki_partition: LokiPartition | None = None
    tail: TailType | None = None
    avb_partition: AvbPartition | None = None
    header_version: int | None = None
    mtk: MtkType | None = None
    qcdt: bool = False


def describe(data: bytes) -> Detection:
    """Run every whole-image check and bundle the results.

    Mirrors the Bash ordering: it detects the signature first, but everything
    else describes the *unwrapped* payload. A still-wrapped buffer therefore
    reports its signature with ``image_type`` left unset rather than being
    rejected as unrecognized -- the layout is simply not knowable until the
    signature is stripped, which is the unpacker's job. Callers strip and then
    call this again on the payload.

    Raises :class:`CorruptImageError` for an unrecognized layout (the Bash
    "Unrecognized format." abort) and :class:`UnsupportedFormatError` for one it
    cannot rebuild.
    """
    signature = detect_signature(data)
    if signature is not None:
        return Detection(signature=signature)

    image_type = detect_image_type(data)
    if image_type is None:
        raise CorruptImageError("Unrecognized format.")
    check_imgtype(image_type)

    patch = detect_patch(data)
    tail = detect_tail(data)
    footer = tail_buffer(data)
    # The magic file appends ", MTK headers" under both the AOSP and ELF rules.
    return Detection(
        image_type=image_type,
        patch=patch,
        loki_partition=loki_partition(data) if patch is PatchType.LOKI else None,
        tail=tail,
        avb_partition=avb1_partition(footer) if tail is TailType.AVBV1 else None,
        header_version=header_version(data),
        mtk=detect_mtk(data),
        qcdt=detect_qcdt(data),
    )


def describe_mtk(kernel: bytes | None, ramdisk: bytes | None) -> MtkType | None:
    """Resolve the MTK header across an already-split kernel and ramdisk.

    The Bash branches on the *presence* of an MTK header, not on the type
    keyword, so a header with no keyword still counts: it prefers the ramdisk's
    type when both carry one and falls back to ``rootfs`` when only the kernel
    does, even if that kernel's own keyword is missing.
    """
    kernel_has = has_mtk(kernel) if kernel else False
    ramdisk_has = has_mtk(ramdisk) if ramdisk else False
    if not kernel_has and not ramdisk_has:
        return None
    ramdisk_type = detect_mtk(ramdisk) if ramdisk_has else None
    kernel_type = detect_mtk(kernel) if kernel_has else None
    return ramdisk_type or kernel_type or MtkType.ROOTFS


def check_imgtype(name: str | ImageType) -> ImageType:
    """Validate a recorded imgtype against the supported whitelist.

    Accepts the ``AOSP-PXA`` spelling the Bash concatenated by hand because
    libmagic never emits it, or an :class:`ImageType` directly.

    Every member of :class:`ImageType` is currently supported, so this raises
    only for a name outside the enum -- but the whitelist is kept explicit
    rather than assumed, so adding a member that AIK cannot rebuild starts
    failing here instead of silently producing an unpackable image.
    """
    if isinstance(name, ImageType):
        image_type = name
    else:
        try:
            image_type = ImageType(name)
        except ValueError:
            raise UnsupportedFormatError(f"Unsupported format: {name}") from None
    if image_type not in SUPPORTED_IMAGE_TYPES:
        raise UnsupportedFormatError(f"Unsupported format: {image_type}")
    return image_type
