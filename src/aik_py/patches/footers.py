"""Detection, stripping and re-application of boot-image footers.

Four footer kinds are handled:

``AVBv1``
    The AOSP ``boot_signer.jar`` signature block (Verified Boot 1.0).
    Identified by the magic ``02 01 01 30 82``, which is simply the head of a
    BER TLV stream: ``INTEGER 1`` followed by a ``SEQUENCE`` with a long-form
    length (``30 82``).  The block embeds the partition name (``/boot`` or
    ``/recovery``), which is the argument the signer needs on repack.

``AVBv2``
    The native Verified Boot 2.0 footer: a fixed 64-byte structure at the very
    end of the partition whose ``AVBf`` magic records the original image size
    plus the offset and size of the vbmeta blob.

``Bump``
    LG's 16-byte ``41 A9 E4 67 ...`` marker.

``SEAndroid``
    Samsung's 16-byte ASCII ``SEANDROIDENFORCE`` marker.

Closing the AVBv2 gap
---------------------
The Bash toolkit detected an AVBv2 footer but only recorded a signature type
for ``*v1`` (``unpackimg.sh:195-199``), and the repack path only handled
``AVBv1`` (``repackimg.sh:339-340``).  An AVBv2 image was therefore unpacked
and then silently repacked *unsigned*, which bricks the device.  This module
detects AVBv2, strips it cleanly, and routes re-signing to :mod:`aik_py.avb`
instead of dropping the signature.

DHTB interaction
----------------
A DHTB-signed image is signed by ``dhtbsign`` itself, which writes its own
DHTB footer over the tail of the partition.  The Bash toolkit accounted for
this by deleting the recorded tail type before appending (the ``DHTB`` branch
of the repack case removes ``split_img/*-tailtype``) so a Bump/SEAndroid
footer is not *also* appended and corrupt the result.  The session layer must
preserve that rule: when a DHTB signature is applied it must suppress the
footer stage entirely rather than call :func:`apply` with the recorded footer.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "AVB_V1_MAGIC",
    "AVB_V2_FOOTER_SIZE",
    "AVB_V2_MAGIC",
    "BUMP_MAGIC",
    "SEANDROID_MAGIC",
    "TAIL_SCAN_LINES",
    "TAIL_SCAN_SIZE",
    "CorruptImageError",
    "Footer",
    "FooterKind",
    "SignatureError",
    "apply",
    "detect",
    "detect_all",
    "strip",
]

try:  # pragma: no cover - the shared hierarchy lands with another unit
    from aik_py.errors import CorruptImageError, SignatureError
except ImportError:  # pragma: no cover - stand-alone fallback

    class FooterError(Exception):
        """Base class for footer handling failures."""

    class CorruptImageError(FooterError):
        """The image advertises a footer but its structure does not hold up."""

    class SignatureError(FooterError):
        """A footer could not be stripped or re-applied."""


# --- Magic values ----------------------------------------------------------
# Transcribed from ``bin/androidbootimg.magic``, the file(1) magic database
# the Bash toolkit uses for its tail detection.

#: ``02 01 01 30 82`` -- BER ``INTEGER 1`` + ``SEQUENCE`` long-form length.
AVB_V1_MAGIC = b"\x02\x01\x01\x30\x82"

#: ASCII ``AVBf`` -- the AVB Footer v2 magic.
AVB_V2_MAGIC = b"AVBf"

#: LG Bump footer, exactly 16 bytes.
BUMP_MAGIC = bytes.fromhex("41A9E467744D1D1BA429F2ECEA655279")

#: Samsung SEAndroid footer, exactly 16 bytes.
SEANDROID_MAGIC = b"SEANDROIDENFORCE"

#: AVB Footer v2 is a fixed 64-byte structure.  Confirmed against AOSP
#: ``avbtool.py`` (``AvbFooter.SIZE = 64``) and the ``AvbVBMetaImageFooter``
#: struct in ``libavb/avb_vbmeta_image.h``.
AVB_V2_FOOTER_SIZE = 64

# The Bash reads the last 8192 bytes and, when that yields only the generic
# word ``data``, retries with the last 50 lines (``unpackimg.sh:185-188``).
# Both stages are reproduced here: a wide byte window first, then a
# line-oriented window that can reach much further back into a partition.
TAIL_SCAN_SIZE = 8192
TAIL_SCAN_LINES = 50

# The AVB v1 block embeds a partition name.  The magic database only knows
# "/boot" and "/recovery", but by-name partitions ("/dev/block/.../vendor_boot")
# are real, so any path-shaped name is accepted rather than leaving an unusual
# device silently unsigned.
_MIN_PARTITION_PATH = "/"

# 64-byte AVB Footer v2, big-endian.  Field order and widths are taken from
# AOSP ``avbtool.py``'s ``AvbFooter.FORMAT_STRING`` (``!4s2LQQQ28x``); the
# 28-byte tail is reserved padding.  Note this is *not* the much wider field
# list the vbmeta *header* carries -- the footer only needs enough to locate
# the blob it describes.
_AVB_V2_STRUCT = struct.Struct(">4sIIQQQ28x")

# BER tag octets used while walking the AVB v1 stream.
_BER_SEQUENCE = 0x30
_BER_PRINTABLE_STRING = 0x13


class FooterKind(StrEnum):
    """The four footer flavours the toolkit knows how to handle."""

    AVBV1 = "AVBv1"
    AVBV2 = "AVBv2"
    BUMP = "Bump"
    SEANDROID = "SEAndroid"


@dataclass(frozen=True, slots=True)
class Footer:
    """A footer found at the tail of an image, with its metadata recovered.

    ``offset``/``length`` describe where the footer sits and how many bytes it
    occupies; for AVBv1 that length is *variable* (1314 bytes for ``/boot``
    versus 1318 for ``/recovery`` with the bundled test key, because the
    partition name is embedded), so it is parsed rather than assumed.
    """

    kind: FooterKind
    #: Absolute byte offset of the footer within the image it came from.
    offset: int = 0
    #: Bytes the footer occupies.
    length: int = 0
    #: AVBv1 only: the partition name recovered from the block, e.g. ``/boot``.
    partition: str | None = None
    #: AVBv2 only: size of the image before AVB metadata was appended.
    original_image_size: int | None = None
    #: AVBv2 only: the raw vbmeta blob the footer points at.
    vbmeta: bytes = b""
    #: Bump/SEAndroid only: the exact trailing bytes, re-emitted verbatim.
    magic: bytes = b""

    def describe(self) -> str:
        """Human-readable one-liner, mirroring the Bash's echo output."""
        if self.kind is FooterKind.AVBV1:
            return f"AVBv1 signing footer ({self.partition})"
        if self.kind is FooterKind.AVBV2:
            return f"AVBv2 signing footer ({len(self.vbmeta)} byte vbmeta)"
        return f"{self.kind} footer"


# --- AVB v1: BER TLV walking ----------------------------------------------


def _ber_tlv(buf: bytes, pos: int) -> tuple[int, int, int]:
    """Read one BER TLV at ``pos``, returning ``(tag, content_start, end)``."""
    if pos + 2 > len(buf):
        raise CorruptImageError("truncated AVBv1 footer")
    tag = buf[pos]
    first = buf[pos + 1]
    if first < 0x80:
        return tag, pos + 2, pos + 2 + first
    nbytes = first & 0x7F
    # 0x80 is BER-only indefinite length, invalid in DER and never emitted by
    # the signer; more than 4 length octets cannot describe a footer this size.
    if nbytes == 0 or nbytes > 4 or pos + 2 + nbytes > len(buf):
        raise CorruptImageError("unsupported BER length encoding in AVBv1 footer")
    length = int.from_bytes(buf[pos + 2 : pos + 2 + nbytes], "big")
    start = pos + 2 + nbytes
    return tag, start, start + length


def _looks_like_partition(raw: bytes) -> bool:
    if not raw.startswith(_MIN_PARTITION_PATH.encode()):
        return False
    text = raw.decode("ascii", "replace")
    return all(c.isalnum() or c in "/_.-" for c in text)


def _target_partition(data: bytes, cstart: int, end: int) -> str | None:
    """Return the partition name if this SEQUENCE is the target/length pair.

    Disambiguated by content -- a PrintableString child that looks like a
    partition path -- rather than by size, so long by-name paths are still
    recognised.
    """
    if cstart >= end:
        return None
    try:
        tag, tstart, tend = _ber_tlv(data, cstart)
    except CorruptImageError:
        return None
    if tag != _BER_PRINTABLE_STRING or tend > end:
        return None
    raw = data[tstart:tend]
    return raw.decode("ascii") if _looks_like_partition(raw) else None


def _sequence_start(data: bytes, start: int) -> int | None:
    """Return the offset of the SEQUENCE header that wraps the TLV at ``start``.

    boot_signer wraps the whole signature in a DER ``SEQUENCE`` whose tag and
    long-form length (``30 82 xx xx``) sit *immediately before* the magic.
    Confirmed empirically against real signer output.  Those 4 bytes are part
    of the footer: leaving them behind would append 4 stray bytes to the
    repacked image.  Returns ``None`` when the preceding octets are not such a
    header, in which case the footer starts at the magic itself.
    """
    if start < 4:
        return None
    if data[start - 4] != _BER_SEQUENCE or data[start - 3] != 0x82:
        return None
    declared = int.from_bytes(data[start - 2 : start], "big")
    if declared != len(data) - start:
        return None
    return start - 4


def _scan_avb_v1(data: bytes, win_start: int) -> Footer | None:
    """Locate and parse an AVBv1 footer ending exactly at the end of ``data``."""
    start = data.rfind(AVB_V1_MAGIC, win_start)
    while start != -1:
        pos = start
        partition: str | None = None
        try:
            # Inside the wrapper: INTEGER formatVersion, SEQUENCE
            # authenticatedAttributes, SEQUENCE algorithmIdentifier,
            # SEQUENCE { PRINTABLESTRING target, INTEGER length }, then the
            # signature OCTET STRING.  The stream runs to the end of the
            # footer with no padding, so the last element's end *is* the
            # footer length.
            while pos < len(data):
                tag, cstart, end = _ber_tlv(data, pos)
                if tag == _BER_SEQUENCE and partition is None:
                    partition = _target_partition(data, cstart, end)
                pos = end
        except CorruptImageError:
            partition, pos = None, start
        if pos == len(data) and partition is not None:
            wrapper = _sequence_start(data, start)
            begin = start if wrapper is None else wrapper
            return Footer(
                kind=FooterKind.AVBV1,
                offset=begin,
                length=len(data) - begin,
                partition=partition,
            )
        start = data.rfind(AVB_V1_MAGIC, win_start, start)
    return None


# --- AVB v2: fixed 64-byte footer ------------------------------------------


def _scan_avb_v2(data: bytes, win_start: int) -> Footer | None:
    """Parse a trailing 64-byte AVB Footer v2 out of ``data``."""
    end = len(data)
    if end - win_start < AVB_V2_FOOTER_SIZE:
        return None
    start = end - AVB_V2_FOOTER_SIZE
    try:
        magic, _major, _minor, original_size, vbmeta_offset, vbmeta_size = (
            _AVB_V2_STRUCT.unpack_from(data, start)
        )
    except struct.error:
        return None
    if magic != AVB_V2_MAGIC:
        return None
    # The blob has to sit ahead of the footer and inside the image.  Sanity
    # checking the recorded offsets stops a corrupt footer from pointing
    # strip() at an arbitrary truncation point.
    if vbmeta_offset + vbmeta_size > start:
        return None
    if original_size > vbmeta_offset:
        return None
    return Footer(
        kind=FooterKind.AVBV2,
        offset=start,
        length=AVB_V2_FOOTER_SIZE,
        original_image_size=original_size,
        vbmeta=data[vbmeta_offset : vbmeta_offset + vbmeta_size],
    )


# --- Detection -------------------------------------------------------------


def _scan(data: bytes, win_start: int) -> list[Footer]:
    """Every footer in ``data[win_start:]``, most specific first."""
    found: list[Footer] = []
    # The closest occurrence to the tail is the meaningful one: a Bump marker
    # can legitimately appear inside the kernel, and the file(1) scan the Bash
    # relies on has the same ambiguity.  rfind matches how file(1) reports the
    # strongest match for a tail buffer.
    for kind, magic in (
        (FooterKind.BUMP, BUMP_MAGIC),
        (FooterKind.SEANDROID, SEANDROID_MAGIC),
    ):
        at = data.rfind(magic, win_start)
        if at != -1 and at + len(magic) == len(data):
            found.append(
                Footer(
                    kind=kind,
                    offset=at,
                    length=len(magic),
                    magic=magic,
                )
            )
    v2 = _scan_avb_v2(data, win_start)
    if v2 is not None:
        found.append(v2)
    v1 = _scan_avb_v1(data, win_start)
    if v1 is not None:
        found.append(v1)
    return found


def _tail_windows(data: bytes) -> Iterator[int]:
    """Yield the start offsets of each scan window, narrowest reach first.

    Stage one covers the last :data:`TAIL_SCAN_SIZE` bytes.  Stage two
    reproduces ``tail -n50``: it reaches back to the start of the 50th
    newline-delimited line from the end, which on a padded partition can be
    megabytes deep but never scans the whole file.  A footer that is neither
    within 8 KiB of the tail nor within the last 50 lines is not found --
    matching the Bash, which is the behaviour that keeps this from mistaking
    incidental bytes in the middle of a ramdisk for a footer.
    """
    if len(data) <= TAIL_SCAN_SIZE:
        yield 0
        return
    yield len(data) - TAIL_SCAN_SIZE
    pos = len(data)
    for _ in range(TAIL_SCAN_LINES):
        found = data.rfind(b"\n", 0, pos)
        if found < 0:
            # ``tail -n50`` on a buffer with no newlines returns the whole
            # thing, so the window opens all the way to the start rather than
            # staying pinned at the tail and finding nothing.
            yield 0
            return
        pos = found
    yield pos


def detect(data: bytes) -> Footer | None:
    """Detect a footer at the tail of ``data``, or ``None`` if there is none."""
    for footer in detect_all(data):
        return footer
    return None


def detect_all(data: bytes) -> list[Footer]:
    """Every footer the two-stage tail scan turns up."""
    for win_start in _tail_windows(data):
        found = _scan(data, win_start)
        if found:
            return found
    return []


# --- Strip / apply ---------------------------------------------------------


def strip(data: bytes, footer: Footer) -> bytes:
    """Remove ``footer`` from ``data``, returning the bare image."""
    if footer.kind is FooterKind.AVBV2:
        # The boundary comes from the footer's own record rather than a fixed
        # length: everything past the original image size is AVB metadata.
        size = footer.original_image_size
        if size is None:
            raise CorruptImageError("AVBv2 footer has no recorded image size")
        # A zero or oversized value would silently discard the whole image, so
        # refuse rather than hand back an empty buffer.
        if not 0 < size <= len(data):
            raise CorruptImageError(f"AVBv2 recorded image size {size} out of range")
        return data[:size]
    if footer.length <= 0 or footer.length > len(data):
        raise CorruptImageError(f"AVB/corrupt footer of {footer.length} bytes")
    return data[: len(data) - footer.length]


def apply(image: bytes, footer: Footer) -> bytes:
    """Re-append ``footer`` to ``image``.

    Bump and SEAndroid are verbatim byte appends.  Both AVB versions delegate
    to :mod:`aik_py.avb`, which owns the actual signing; it is imported
    defensively so this unit stands alone until that module lands.
    """
    match footer.kind:
        case FooterKind.BUMP:
            return image + BUMP_MAGIC
        case FooterKind.SEANDROID:
            return image + SEANDROID_MAGIC
        case FooterKind.AVBV1:
            return _apply_avb_v1(image, footer)
        case FooterKind.AVBV2:
            return _apply_avb_v2(image, footer)


def _avb_module():
    try:
        from aik_py import avb
    except ImportError as exc:
        raise SignatureError(
            "AVB re-signing not available: aik_py.avb could not be imported. "
            "Refusing to silently drop the AVB signature."
        ) from exc
    return avb


def _apply_avb_v1(image: bytes, footer: Footer) -> bytes:
    # The partition name is embedded *inside* the signature, so guessing it
    # would mint a block that names the wrong partition -- worse than failing
    # outright.  Only a footer that came out of detect() can be re-applied.
    if not footer.partition:
        raise SignatureError(
            "cannot re-sign AVBv1 without a partition name; re-run detect() to "
            "recover the footer"
        )
    avb = _avb_module()
    sign = getattr(avb, "sign_v1", None)
    if sign is None:
        raise SignatureError(
            "AVB re-signing not available: aik_py.avb has no sign_v1()."
        )
    # boot_signer takes the partition with its leading slash ("/boot").
    return sign(image, footer.partition)


def _apply_avb_v2(image: bytes, footer: Footer) -> bytes:
    avb = _avb_module()
    sign = getattr(avb, "sign_v2", None)
    if sign is None:
        raise SignatureError(
            "AVB re-signing not available: aik_py.avb has no sign_v2()."
        )
    return sign(image)
