"""Ramdisk compression formats.

The Bash toolkit detects a ramdisk with ``file -m bin/magic *-*ramdisk`` and
decompresses with ``$comp -dc``. Here the container is identified by sniffing
its magic bytes directly, and every format is handled natively except ``lzop``.

``lz4`` versus ``lz4-legacy`` is decided purely by the frame format of the
blob, never by its filename: both write a ``.lz4`` extension, and the only
thing distinguishing them afterwards is the compression type recorded by the
unpack step.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import struct
import subprocess
from enum import StrEnum

from . import bin

try:
    from .errors import (
        CompressionError,
        CorruptImageError,
        MissingToolError,
        UnsupportedCompressionError,
    )
except ImportError:  # pragma: no cover - errors module lands in a sibling unit

    class CompressionError(Exception):
        """Base class for compression failures."""

    class UnsupportedCompressionError(CompressionError):
        """The blob's compression format is not recognised or not supported."""

    class CorruptImageError(CompressionError):
        """The blob claims a known format but its payload is malformed."""

    # Reuse the resolver's class so that `except MissingToolError` around a
    # `bin.require()` call actually catches what it raises.
    MissingToolError = bin.MissingToolError


class Compression(StrEnum):
    GZIP = "gzip"
    BZIP2 = "bzip2"
    XZ = "xz"
    LZMA = "lzma"
    LZOP = "lzop"
    LZ4 = "lz4"
    LZ4_LEGACY = "lz4-l"
    CPIO = "cpio"
    EMPTY = "empty"

    @property
    def extension(self) -> str:
        """The filename suffix the Bash toolkit appends to ``ramdisk.cpio``."""
        return _EXTENSIONS[self]


_EXTENSIONS: dict[Compression, str] = {
    Compression.GZIP: ".gz",
    Compression.BZIP2: ".bz2",
    Compression.XZ: ".xz",
    Compression.LZMA: ".lzma",
    Compression.LZOP: ".lzo",
    Compression.LZ4: ".lz4",
    Compression.LZ4_LEGACY: ".lz4",
    Compression.CPIO: "",
    Compression.EMPTY: ".empty",
}

_LZOP_MAGIC = b"\x89LZO\x00\r\n\x1a\n"
_LZ4_MAGIC = 0x184D2204
_LZ4_LEGACY_MAGIC = 0x184C2102

# All four cpio magics bin/magic recognises: newc (SVR4, with and without
# CRC), odc/pre-SVR4, and the byte-swapped variant. Any of them means the
# ramdisk is uncompressed.
_CPIO_MAGICS = (b"070701", b"070702", b"070707", b"0143561")


def extension(comp: Compression) -> str:
    """Return the file extension used for ``comp`` (``.gz``, ``""``, ...)."""
    return _EXTENSIONS[Compression(comp)]


def _u32_le(data: bytes) -> int:
    return int.from_bytes(data[:4], "little")


def detect(data: bytes) -> Compression:
    """Identify the container format of ``data`` by its leading magic bytes.

    Raises:
        UnsupportedCompressionError: if the bytes match no known format, which
            is the case the Bash toolkit reports as ``Unrecognized format.``
    """
    if not data:
        return Compression.EMPTY

    if data.startswith(b"\x1f\x8b"):
        return Compression.GZIP
    if data.startswith(b"BZh"):
        return Compression.BZIP2
    if data.startswith(b"\xfd7zXZ\x00"):
        return Compression.XZ
    if data.startswith(_LZOP_MAGIC):
        return Compression.LZOP
    if data.startswith(_CPIO_MAGICS):
        return Compression.CPIO

    magic = _u32_le(data)
    if magic == _LZ4_MAGIC:
        return Compression.LZ4
    if magic == _LZ4_LEGACY_MAGIC:
        return Compression.LZ4_LEGACY

    # lzma-alone has no magic number of its own; the closest equivalent is the
    # properties byte 0x5d, which bin/magic tests as `lelong&0xffffff = 0x5d`.
    # It is deliberately checked last, after every format with a real magic.
    if len(data) >= 3 and magic & 0xFFFFFF == 0x5D:
        return Compression.LZMA

    raise UnsupportedCompressionError(
        f"Unrecognized ramdisk format: no known magic in {data[:8].hex()!r}."
    )


# Legacy lz4 frames use a fixed 8 MiB block and store no sizes of their own, so
# blocks are decoded into the maximum block size and trimmed to what the block
# decoder actually produced.
_LZ4_LEGACY_BLOCK = 1 << 23


def _lz4_legacy_compress(data: bytes, level: int) -> bytes:
    out = bytearray(struct.pack("<I", _LZ4_LEGACY_MAGIC))
    if not data:
        return bytes(out)
    for start in range(0, len(data), _LZ4_LEGACY_BLOCK):
        block = data[start : start + _LZ4_LEGACY_BLOCK]
        compressed = lz4_block().compress(
            block, store_size=False, mode="high_compression", compression=level
        )
        out += struct.pack("<I", len(compressed)) + compressed
    return bytes(out)


def _lz4_legacy_decompress(data: bytes) -> bytes:
    if data[:4] != struct.pack("<I", _LZ4_LEGACY_MAGIC):
        raise CorruptImageError("lz4-legacy blob is missing its frame magic.")
    out = bytearray()
    pos = 4
    while pos < len(data):
        # A legacy frame is terminated by EOF rather than a marker, so the end
        # of the buffer is a clean stop. Anything shorter than a full size
        # field is a truncated frame.
        if pos + 4 > len(data):
            raise CorruptImageError(
                f"lz4-legacy blob is truncated: {len(data) - pos} trailing bytes "
                f"cannot form a block size."
            )
        (block_size,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if block_size == 0:
            break
        if pos + block_size > len(data):
            raise CorruptImageError(
                f"lz4-legacy blob is truncated: block at {pos - 4} claims "
                f"{block_size} bytes but only {len(data) - pos} remain."
            )
        out += lz4_block().decompress(
            data[pos : pos + block_size], uncompressed_size=_LZ4_LEGACY_BLOCK
        )
        pos += block_size
    return bytes(out)


def lz4_block():
    """Import ``lz4.block`` lazily so importing this module stays cheap."""
    import lz4.block

    return lz4.block


def lz4_frame():
    """Import ``lz4.frame`` lazily so importing this module stays cheap."""
    import lz4.frame

    return lz4.frame


def decompress(data: bytes, comp: Compression) -> bytes:
    """Decompress ``data`` according to ``comp``.

    Raises:
        CompressionError: for a missing external tool (lzop), or if the blob's
            payload is malformed.
    """
    comp = Compression(comp)

    if comp is Compression.EMPTY:
        return b""
    if comp is Compression.CPIO:
        return data
    if comp is Compression.LZOP:
        return _lzop_decompress(data)
    if comp is Compression.LZ4_LEGACY:
        return _lz4_legacy_decompress(data)

    try:
        if comp is Compression.GZIP:
            return gzip.decompress(data)
        if comp is Compression.BZIP2:
            return bz2.decompress(data)
        if comp is Compression.XZ:
            return lzma.decompress(data, format=lzma.FORMAT_XZ)
        if comp is Compression.LZMA:
            return lzma.decompress(data, format=lzma.FORMAT_ALONE)
        return lz4_frame().decompress(data)
    except CompressionError:
        raise
    except Exception as exc:
        # gzip, bz2, lzma and lz4 each raise their own stdlib type on bad input;
        # normalise so callers only ever have to catch CompressionError.
        raise CorruptImageError(f"Malformed {comp} stream: {exc}") from exc


def _preset(level: int | None, default: int) -> int:
    """Normalise a level onto the 0-9 scale, accepting the shell's -1..-9 form."""
    return default if level is None else abs(level)


def compress(data: bytes, comp: Compression, level: int | None = None) -> bytes:
    """Compress ``data`` into the ``comp`` container.

    ``level`` is the compressor's own 0-9 scale, also accepted in the shell's
    signed -1..-9 form. ``None`` selects the default the Bash toolkit applies:
    level 1 for xz and level 9 for lz4/lz4-legacy, and the compressor's native
    default for every other format.
    """
    comp = Compression(comp)

    if comp is Compression.EMPTY:
        # An empty ramdisk is a zero-length file, matching repackimg.sh's
        # `touch ramdisk-new.cpio.empty`; this keeps compress the inverse of
        # decompress, which always yields b"" for EMPTY.
        return b""
    if comp is Compression.CPIO:
        return data
    if comp is Compression.GZIP:
        return gzip.compress(data, compresslevel=_preset(level, 9))
    if comp is Compression.BZIP2:
        return bz2.compress(data, compresslevel=_preset(level, 9))
    if comp is Compression.XZ:
        # Some bootloaders only accept xz streams carrying a CRC32 rather than
        # the format default of CRC64, so pin the check type rather than the
        # library's preference.
        return lzma.compress(
            data,
            format=lzma.FORMAT_XZ,
            check=lzma.CHECK_CRC32,
            preset=_preset(level, 1),
        )
    if comp is Compression.LZMA:
        return lzma.compress(data, format=lzma.FORMAT_ALONE, preset=_preset(level, 9))
    if comp is Compression.LZ4:
        return lz4_frame().compress(data, compression_level=_preset(level, 9))
    if comp is Compression.LZ4_LEGACY:
        return _lz4_legacy_compress(data, _preset(level, 9))
    return _lzop_compress(data, level)


def _lzop_decompress(data: bytes) -> bytes:
    return _run_lzop(["-d", "-c"], data)


def _lzop_compress(data: bytes, level: int | None) -> bytes:
    args = ["-c"]
    if level is not None:
        # lzop spells levels as -1..-9; note that -L is its licence flag, not
        # a level modifier.
        args.append(f"-{level}")
    return _run_lzop(args, data)


def _run_lzop(args: list[str], data: bytes) -> bytes:
    try:
        exe = bin.require("lzop")
    except MissingToolError as exc:
        raise MissingToolError(
            "lzop is required for .lzo ramdisks. Install it with "
            "`apt install lzop`, or point $AIK_LZOP at the binary."
        ) from exc

    # lzop has no pure-Python equivalent and none is published on PyPI, so the
    # vendored binary is the only way to read or write the .lzo container.
    result = subprocess.run(
        [exe, *args, "-f"], input=data, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise CompressionError(
            f"lzop failed with status {result.returncode}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout
