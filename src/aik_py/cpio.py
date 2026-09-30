"""Pure-Python reader/writer for SVR4 ``newc`` cpio archives.

Android boot/recovery ramdisks are gzip- or lz4-compressed ``newc`` cpio
archives. AIK's Bash implementation shells out to the system ``cpio`` and warns
about specific GNU cpio releases (2.12/2.13/2.15) producing subtly different or
unusable repacks. Implementing the format in-process removes both the
subprocess dependency and that version hazard.

Format, as produced by ``cpio -H newc``:

* Each entry is a 110-byte ASCII header, then the NUL-terminated filename,
  then the file payload.
* Every header field is uppercase hex. The layout is *not* 13 uniform 8-char
  fields: ``c_magic`` is only 6 chars (``070701``), followed by 12 eight-char
  fields (``c_ino`` .. ``c_namesize``), followed by the eight-char ``c_check``.
  This 6 + 12*8 + 8 = 110 asymmetry is the classic source of newc bugs.
* Field byte offsets within the 110-byte header:
  ``c_magic``[0:6], ``c_ino``[6:14], ``c_mode``[14:22], ``c_uid``[22:30],
  ``c_gid``[30:38], ``c_nlink``[38:46], ``c_mtime``[46:54],
  ``c_filesize``[54:62], ``c_devmajor``[62:70], ``c_devminor``[70:78],
  ``c_rdevmajor``[78:86], ``c_rdevminor``[86:94], ``c_namesize``[94:102],
  ``c_check``[102:110].
* The filename block is padded with NULs so ``110 + c_namesize`` is a multiple
  of 4; the payload is then padded so its own start plus length stays 4-aligned.
* The archive terminates with a ``TRAILER!!!`` entry. GNU cpio additionally
  pads the whole file out to 512 bytes; that padding is not part of the format
  and this reader tolerates (but the writer does not emit) it.

``c_mtime`` is embedded in every entry, so byte-for-byte comparisons of
archives built from a live filesystem are inherently unstable. Tests that
compare bytes must pin ``mtime`` to a fixed value.
"""

from __future__ import annotations

import posixpath
import stat
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

try:  # errors.py lands in the unit-1 PR; fall back until then.
    from aik_py.errors import CorruptImageError
except ModuleNotFoundError as exc:  # pragma: no cover - gone once unit 1 lands
    # Only tolerate the module being absent. A broken errors.py must surface
    # loudly, otherwise CpioError would stop being a subclass of the real
    # CorruptImageError and callers' ``except`` clauses would silently miss it.
    if exc.name != "aik_py.errors":
        raise

    class CorruptImageError(Exception):  # type: ignore[no-redef]
        """Raised when an image or its container cannot be parsed."""


MAGIC = b"070701"
HEADER_SIZE = 110
TRAILER_NAME = "TRAILER!!!"

# Byte offsets of each field inside the 110-byte header. ``c_magic`` is 6 chars;
# the rest are 8. See the module docstring for the derivation.
_OFF_INO = 6
_OFF_MODE = 14
_OFF_UID = 22
_OFF_GID = 30
_OFF_NLINK = 38
_OFF_MTIME = 46
_OFF_FILESIZE = 54
_OFF_DEVMAJOR = 62
_OFF_DEVMINOR = 70
_OFF_RDEVMAJOR = 78
_OFF_RDEVMINOR = 86
_OFF_NAMESIZE = 94
_OFF_CHECK = 102

# newc fields are conventionally uppercase hex, but GNU cpio parses either case
# and some writers emit lowercase, so accept both rather than rejecting
# archives that real tooling produces.
_HEX_DIGITS = frozenset(b"0123456789abcdefABCDEF")

# A 32-bit unsigned field tops out at 8 hex chars; anything larger than this is
# a corrupt/hostile header rather than a real file. 4 GiB is a generous ceiling
# for a ramdisk member and bounds every arithmetic below to something sane.
_MAX_FIELD = 0xFFFFFFFF


class CpioError(CorruptImageError):
    """Raised when a cpio archive is malformed or unsafe to extract."""


def _pad4(n: int) -> int:
    """Bytes of NUL padding needed after ``n`` bytes to reach a 4-byte boundary."""
    return -n % 4


@dataclass(slots=True)
class CpioEntry:
    """One archive member.

    ``data`` holds the file payload; for directories it is empty and for
    symlinks it is the link target. ``mode`` carries the ``S_IF*`` type bits
    alongside the permission bits.

    ``uid``/``gid`` default to 0 because the bootloader consumes a ramdisk
    before any Android userspace exists, so every file must be root-owned.
    This mirrors AIK's ``cpio -R 0:0``, which forces 0:0 when the repack is
    not running as root.
    """

    name: str
    data: bytes = b""
    mode: int = stat.S_IFREG | 0o644
    uid: int = 0
    gid: int = 0
    mtime: int = 0
    ino: int = 0
    nlink: int = 1
    devmajor: int = 0
    devminor: int = 0
    rdevmajor: int = 0
    rdevminor: int = 0
    check: int = 0

    def __post_init__(self) -> None:
        # Reject an out-of-range mode here so is_dir/is_file can never raise a
        # bare OverflowError, and so the writer never has to re-check it.
        if not 0 <= self.mode <= _MAX_FIELD:
            raise CpioError(f"cpio c_mode out of range: {self.mode}")

    @property
    def is_dir(self) -> bool:
        return stat.S_ISDIR(self.mode)

    @property
    def is_symlink(self) -> bool:
        return stat.S_ISLNK(self.mode)

    @property
    def is_file(self) -> bool:
        return stat.S_ISREG(self.mode)


def _encode_field(value: int, width: int, name: str) -> bytes:
    """Render ``value`` as uppercase hex, or raise if it does not fit."""
    if value < 0 or value > _MAX_FIELD:
        raise CpioError(f"cpio {name} out of range: {value}")
    return f"{value:0{width}X}".encode("ascii")


def _parse_field(raw: bytes, name: str) -> int:
    """Decode a hex header field, raising :class:`CpioError` if invalid."""
    for byte in raw:
        if byte not in _HEX_DIGITS:
            raise CpioError(f"cpio header field {name!r} is not valid hex: {raw!r}")
    return int(raw, 16)


def _sanitize(name: str) -> str:
    """Make an archive member name safe to write beneath a target directory.

    AIK extracts with ``cpio -i -d --no-absolute-filenames``, which strips
    leading ``/`` so a hostile archive cannot write outside the destination.
    We reproduce that, and additionally reject ``..`` components outright
    rather than silently rewriting them, since a traversal attempt means the
    archive is untrustworthy. A name that reduces to the archive root (``.``,
    ``/``, ``./``) is preserved as ``.`` — GNU cpio emits that entry first in
    every archive and it is not an escape.
    """
    if not name:
        # A zero-length member name is legal on the wire (c_namesize == 1, just
        # the NUL) and must survive a round trip unchanged.
        return name
    if "\0" in name:
        # Only the final byte is stripped as the name terminator, so an interior
        # NUL would survive into the name. Callers build paths with
        # os.path.join(), which raises a bare ValueError on such a name; fail
        # here instead so CpioError stays the only exception callers must catch.
        raise CpioError(f"cpio member name contains an interior NUL: {name!r}")
    cleaned = name.lstrip("/")
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise CpioError(f"cpio member escapes the extraction root: {name!r}")
    if not parts:
        return "."
    return posixpath.join(*parts)


def read_archive(data: bytes, *, sanitize: bool = True) -> list[CpioEntry]:
    """Parse a ``newc`` archive into a list of :class:`CpioEntry`.

    Stops at the ``TRAILER!!!`` entry; any bytes after it (GNU cpio pads the
    file to 512 bytes) are ignored. When ``sanitize`` is true, member names are
    passed through :func:`_sanitize` first.

    Hardlinks are resolved: newc stores every additional name for an inode as
    its own header with the same ``c_ino``/``c_nlink`` but ``c_filesize == 0``.
    Each name is returned carrying the shared payload, so a caller can
    materialise entries independently without truncating the second link.
    """
    parsed: list[tuple[CpioEntry, int, bool]] = []
    view = memoryview(data)
    total = len(view)
    offset = 0

    while True:
        if total - offset < HEADER_SIZE:
            raise CpioError(
                f"cpio archive truncated: {total - offset} trailing bytes at "
                f"offset {offset} cannot hold a {HEADER_SIZE}-byte header"
            )

        header = bytes(view[offset : offset + HEADER_SIZE])
        if header[0:6] != MAGIC:
            raise CpioError(f"bad cpio magic at offset {offset}: {header[0:6]!r}")

        namesize = _parse_field(header[_OFF_NAMESIZE : _OFF_NAMESIZE + 8], "c_namesize")
        filesize = _parse_field(header[_OFF_FILESIZE : _OFF_FILESIZE + 8], "c_filesize")
        # The format is unsigned; a "negative" size is really a huge one that
        # would run past the buffer, so bounds-check against what remains.
        if namesize > total - offset - HEADER_SIZE or namesize < 1:
            raise CpioError(
                f"cpio c_namesize {namesize} at offset {offset} runs past the archive"
            )

        name_start = offset + HEADER_SIZE
        name_raw = bytes(view[name_start : name_start + namesize])
        if name_raw[-1] != 0:
            raise CpioError(
                f"cpio member name at offset {offset} is not NUL-terminated"
            )
        name = name_raw[:-1].decode("utf-8", errors="surrogateescape")

        # newc pads the combined header+name block to a 4-byte boundary, so the
        # payload does not start on a 4-byte boundary of its own unless namesize
        # happens to cooperate.
        data_start = offset + HEADER_SIZE + namesize + _pad4(HEADER_SIZE + namesize)

        if data_start > total or filesize > total - data_start:
            raise CpioError(
                f"cpio entry {name!r} declares {filesize} payload bytes but only "
                f"{total - data_start} remain in the archive"
            )

        mode = _parse_field(header[_OFF_MODE : _OFF_MODE + 8], "c_mode")
        # Identify the trailer by an all-zero metadata header, not just by name,
        # so a member that merely shares the reserved name is not silently
        # treated as end-of-archive.
        is_trailer = name == TRAILER_NAME and mode == 0 and filesize == 0
        if not is_trailer:
            member = _sanitize(name) if sanitize else name
            entry = CpioEntry(
                name=member,
                data=bytes(view[data_start : data_start + filesize]),
                mode=mode,
                uid=_parse_field(header[_OFF_UID : _OFF_UID + 8], "c_uid"),
                gid=_parse_field(header[_OFF_GID : _OFF_GID + 8], "c_gid"),
                mtime=_parse_field(header[_OFF_MTIME : _OFF_MTIME + 8], "c_mtime"),
                ino=_parse_field(header[_OFF_INO : _OFF_INO + 8], "c_ino"),
                nlink=_parse_field(header[_OFF_NLINK : _OFF_NLINK + 8], "c_nlink"),
                devmajor=_parse_field(
                    header[_OFF_DEVMAJOR : _OFF_DEVMAJOR + 8], "c_devmajor"
                ),
                devminor=_parse_field(
                    header[_OFF_DEVMINOR : _OFF_DEVMINOR + 8], "c_devminor"
                ),
                rdevmajor=_parse_field(
                    header[_OFF_RDEVMAJOR : _OFF_RDEVMAJOR + 8], "c_rdevmajor"
                ),
                rdevminor=_parse_field(
                    header[_OFF_RDEVMINOR : _OFF_RDEVMINOR + 8], "c_rdevminor"
                ),
                check=_parse_field(header[_OFF_CHECK : _OFF_CHECK + 8], "c_check"),
            )
            parsed.append((entry, filesize, entry.is_dir))

        offset = data_start + filesize + _pad4(filesize)
        if is_trailer:
            break

    _resolve_hardlinks(parsed)
    return [entry for entry, _, _ in parsed]


def _resolve_hardlinks(
    parsed: list[tuple[CpioEntry, int, bool]],
) -> None:
    """Give every name in a hardlink group the group's shared payload, in place.

    newc records each additional link name as a header with the same ``c_ino``
    and ``c_nlink > 1`` but ``c_filesize == 0``. Without this pass a caller
    materialising entries one by one writes those names as empty files.
    """
    # First pass: index the inodes that actually carry a payload. Directories
    # legitimately have no payload, so they never act as a link-group source.
    payload_by_ino: dict[int, bytes] = {}
    for entry, filesize, is_dir in parsed:
        if not is_dir and filesize:
            payload_by_ino.setdefault(entry.ino, entry.data)

    if not payload_by_ino:
        return

    for entry, _, _ in parsed:
        if entry.nlink > 1 and not entry.data and entry.ino in payload_by_ino:
            entry.data = payload_by_ino[entry.ino]


def write_archive(entries: Sequence[CpioEntry] | Iterable[CpioEntry]) -> bytes:
    """Serialise ``entries`` as a canonical ``newc`` archive.

    ``write_archive(read_archive(x)) == x`` for any archive this module writes.
    Reading an archive from another writer does not guarantee byte-stability:
    ``read_archive`` normalises member names (see :func:`_sanitize`), resolves
    hardlinks, and ignores the 512-byte tail padding GNU cpio appends, so a
    round trip through a foreign archive returns semantically equal entries
    rather than identical bytes. No tail padding is emitted here.
    """
    chunks: list[bytes] = []
    for entry in entries:
        chunks.append(_encode_entry(entry))
    chunks.append(_encode_trailer())
    return b"".join(chunks)


def _encode_entry(entry: CpioEntry) -> bytes:
    if entry.name == TRAILER_NAME:
        # A member with the trailer's name would terminate parsing on read, so
        # the entry and everything after it would vanish from the repack.
        raise CpioError(f"cpio member name {TRAILER_NAME!r} is reserved")
    name_raw = entry.name.encode("utf-8", errors="surrogateescape") + b"\0"
    # Directories and other non-regular entries carry no payload; a nonzero
    # c_filesize on a directory is a corrupt archive, not something to mirror.
    payload = b"" if entry.is_dir else entry.data
    filesize = len(payload)
    if filesize > _MAX_FIELD:
        raise CpioError(
            f"cpio member {entry.name!r} payload exceeds the 4 GiB field limit"
        )

    header = b"".join(
        (
            MAGIC,
            _encode_field(entry.ino, 8, "c_ino"),
            _encode_field(entry.mode, 8, "c_mode"),
            _encode_field(entry.uid, 8, "c_uid"),
            _encode_field(entry.gid, 8, "c_gid"),
            _encode_field(entry.nlink, 8, "c_nlink"),
            _encode_field(entry.mtime, 8, "c_mtime"),
            _encode_field(filesize, 8, "c_filesize"),
            _encode_field(entry.devmajor, 8, "c_devmajor"),
            _encode_field(entry.devminor, 8, "c_devminor"),
            _encode_field(entry.rdevmajor, 8, "c_rdevmajor"),
            _encode_field(entry.rdevminor, 8, "c_rdevminor"),
            _encode_field(len(name_raw), 8, "c_namesize"),
            _encode_field(entry.check, 8, "c_check"),
        )
    )
    name_pad = b"\0" * _pad4(HEADER_SIZE + len(name_raw))
    return header + name_raw + name_pad + payload + b"\0" * _pad4(filesize)


def _encode_trailer() -> bytes:
    # Every field is zero except c_namesize, which must carry the length of
    # "TRAILER!!!\0" (11) or conforming readers reject the archive as truncated.
    name_raw = TRAILER_NAME.encode("ascii") + b"\0"
    header = b"".join(
        (
            MAGIC,
            b"0" * (_OFF_NAMESIZE - _OFF_INO),  # c_ino .. c_rdevminor
            _encode_field(len(name_raw), 8, "c_namesize"),
            b"0" * (HEADER_SIZE - _OFF_CHECK),  # c_check
        )
    )
    return header + name_raw + b"\0" * _pad4(HEADER_SIZE + len(name_raw))
