"""Head signature wrappers that some OEMs place in front of a boot image.

Detection lives in :mod:`aik_py.detect`; this module is the strip/apply half
that runs once a signature type is known.

Which of these are exact inverses
---------------------------------
:class:`SignatureType.NOOK` and :class:`SignatureType.NOOKTAB` are pure byte
slicing and re-concatenation, so ``strip`` then ``apply`` reproduces the
original image bit for bit.  Every other type is *not* an exact inverse:
:class:`SignatureType.BLOB` and :class:`SignatureType.DHTB` regenerate their
header with a bundled signing tool (so the header bytes differ even though the
payload round-trips), and :class:`SignatureType.CHROMEOS` depends on key
material this library does not own.  :attr:`StripResult.exact_inverse` records
which case you are in.
"""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

#: ``uname -m`` values that do not match the bundle's directory names.
_ARCH_BY_MACHINE = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
}

__all__ = [
    "BLOB_HEADER",
    "BLOB_NAME_LEN",
    "DHTB_HEADER_SIZE",
    "NOOKTAB_BOOTLOADER_SIZE",
    "NOOK_BOOTLOADER_SIZE",
    "SONY_DUMP_ABORT_EXIT",
    "MissingToolError",
    "SignatureError",
    "SignatureType",
    "StripResult",
    "apply",
    "strip",
]

# --------------------------------------------------------------------------
# Errors.  Prefer the shared hierarchy when another unit has landed it.
# --------------------------------------------------------------------------

try:  # pragma: no cover - depends on concurrent unit
    from aik_py.errors import MissingToolError, SignatureError
except ImportError:  # pragma: no cover - standalone fallback

    class SignatureError(Exception):
        """Raised when a signature wrapper cannot be stripped or re-applied."""

    class MissingToolError(SignatureError):
        """Raised when a bundled helper binary required by a wrapper is absent."""


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: ``unpackimg.sh`` line 112 scrapes the partition name from ``blobunpack``
#: stdout with ``tail -n+5 | cut -d" " -f2 | dd bs=1 count=3``, so the name is
#: truncated to exactly three characters (``boo``, ``rec``, ...).
BLOB_NAME_LEN = 3

#: ``repackimg.sh`` line 342 writes this literal before appending the packed
#: blob.  The prefix is 20 bytes and the awk literal adds eight ``\00``, so the
#: header is **28** bytes, not the 27 the shell snippet's arithmetic suggests.
BLOB_HEADER = b"-SIGNED-BY-SIGNBLOB-" + bytes(8)
assert len(BLOB_HEADER) == 28

#: ``unpackimg.sh`` line 116 reads ``dd bs=4096 skip=512 iflag=skip_bytes``.
#: ``iflag=skip_bytes`` makes ``skip`` count *bytes*, so exactly 512 bytes come
#: off the front.  Reading it as 512 blocks (2 MiB) is the classic trap: the
#: ``bs=4096`` and the ``skip=512`` look like they compose into 2 MiB, but the
#: ``iflag=skip_bytes`` flag silently redefines the unit.  Do not reproduce the
#: confusion -- strip 512 bytes.
DHTB_HEADER_SIZE = 512

#: ``unpackimg.sh`` lines 118-119 stash the first megabyte as ``master_boot.key``.
NOOK_BOOTLOADER_SIZE = 1 << 20  # 1 MiB

#: ``unpackimg.sh`` lines 122-123 use the same trick with a quarter-megabyte.
NOOKTAB_BOOTLOADER_SIZE = 256 * 1024  # 256 KiB


class SignatureType(StrEnum):
    """Signature/packaging wrappers understood by the unpack/repack pipeline."""

    BLOB = "BLOB"
    CHROMEOS = "CHROMEOS"
    DHTB = "DHTB"
    NOOK = "NOOK"
    NOOKTAB = "NOOKTAB"
    SINV1 = "SINv1"
    SINV2 = "SINv2"
    SINV3 = "SINv3"

    @property
    def is_sin(self) -> bool:
        """SONY SIN containers are packaging, not signing.

        ``unpackimg.sh`` line 128 writes the sigtype and then immediately
        ``rm -f``s it, so nothing is re-applied on repack.  The container only
        needs unwrapping; there is no signature to restore.
        """
        return self in _SIN_TYPES

    @property
    def bootloader_size(self) -> int | None:
        """Leading stashed bootloader size, or ``None`` if not a NOOK variant."""
        return _BOOTLOADER_SIZES.get(self)


_SIN_TYPES = frozenset({SignatureType.SINV1, SignatureType.SINV2, SignatureType.SINV3})
_BOOTLOADER_SIZES = {
    SignatureType.NOOK: NOOK_BOOTLOADER_SIZE,
    SignatureType.NOOKTAB: NOOKTAB_BOOTLOADER_SIZE,
}


# --------------------------------------------------------------------------
# Bundled tool resolution
# --------------------------------------------------------------------------


#: Env override for the toolkit root, for installs where ``bin/`` does not sit
#: next to the package (the wheel ships only ``aik_py/``, not the 18 binaries).
_BIN_ROOT_ENV = "AIK_PY_BIN_ROOT"


def _bin_root() -> Path:
    """Locate the bundled ``bin/`` tree.

    Prefers the shared resolver, then ``$AIK_PY_BIN_ROOT``, then the repo
    layout ``<root>/src/aik_py/patches/signatures.py``.  The last hop only
    resolves under an editable install; for a real wheel the env var is the
    supported route, which is why the error message names it.
    """
    try:  # pragma: no cover - depends on concurrent unit
        from aik_py.bin import tool_path  # type: ignore[import-not-found]

        return Path(tool_path("blobpack")).parent.parent
    except (ImportError, AttributeError, TypeError, NotImplementedError):
        pass
    override = os.environ.get(_BIN_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[3] / "bin"


def _tool(name: str) -> Path:
    """Resolve a bundled helper binary for the running platform."""
    arch = _ARCH_BY_MACHINE.get(platform.machine(), platform.machine())
    plat = "macos" if platform.system() == "Darwin" else "linux"
    root = _bin_root()
    candidate = root / plat / arch / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    raise MissingToolError(
        f"bundled tool {name!r} not found (or not executable) at {candidate}. "
        f"The signature types backed by prebuilt binaries cannot run in-process; "
        f"set {_BIN_ROOT_ENV} to the directory containing the AIK bin/ tree."
    )


#: Exit status the bundled ``sony_dump`` returns.  It writes every member it
#: found and then dies in its teardown, so the SIGABRT is expected and the
#: members on disk are the real result.  Verified against the bundled binary.
SONY_DUMP_ABORT_EXIT = -6


def _run(argv: Sequence[str], *, cwd: Path, allow_abort: bool = False) -> str:
    """Run a bundled tool, surfacing failures instead of swallowing them.

    The Bash pipelines redirect tool output to ``/dev/null`` and then check
    ``$?`` in ways that lose the real error.  A non-zero exit is a hard failure
    here -- a silently wrong blob or a half-written image is far worse -- with
    one exception: ``allow_abort`` tolerates the teardown crash of a tool that
    has already produced its output (see :data:`SONY_DUMP_ABORT_EXIT`).
    """
    try:
        proc = subprocess.run(
            list(argv),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if not (allow_abort and exc.returncode == SONY_DUMP_ABORT_EXIT):
            raise SignatureError(
                f"{Path(argv[0]).name} failed (exit {exc.returncode}): "
                f"{(exc.stderr or exc.stdout or '').strip()[:400]}"
            ) from exc
        return exc.stdout or ""
    except OSError as exc:
        raise SignatureError(f"could not execute {argv[0]}: {exc}") from exc
    return proc.stdout


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StripResult:
    """Output of :func:`strip`.

    ``image`` is the bare boot image.  ``sidecar`` carries whatever must be
    preserved so :func:`apply` can rebuild the wrapped form: the Nook
    bootloader, or the three-character BLOB partition name.
    """

    sigtype: SignatureType
    image: bytes
    sidecar: bytes | None = None
    #: Every partition ``blobunpack`` extracted, keyed by name.  ASUS blobs are
    #: usually single-partition, but keeping the rest prevents silent loss when
    #: one is not.
    partitions: dict[str, bytes] = field(default_factory=dict)
    #: All member payloads, in sorted-filename order.
    members: tuple[bytes, ...] = field(default=())

    @property
    def blob_name(self) -> str | None:
        """BLOB partition name recovered from ``blobunpack`` output."""
        if self.sidecar is None or not self.is_blob:
            return None
        return self.sidecar.decode("ascii", errors="replace")

    @property
    def is_blob(self) -> bool:
        return self.sigtype == SignatureType.BLOB

    @property
    def is_sin(self) -> bool:
        return self.sigtype.is_sin

    @property
    def exact_inverse(self) -> bool:
        """Whether ``apply(strip(x).image, strip(x))`` reproduces ``x`` exactly.

        True only for the Nook variants, whose wrapper is plain byte
        concatenation.  DHTB and BLOB regenerate a header via a signing tool, so
        the wrapped bytes match in structure but not in value.
        """
        return self.sigtype in (SignatureType.NOOK, SignatureType.NOOKTAB)


# --------------------------------------------------------------------------
# strip
# --------------------------------------------------------------------------


def _strip_nook(data: bytes, size: int, name: str) -> StripResult:
    if len(data) < size:
        raise SignatureError(
            f"{name} wrapper needs at least {size} bytes of bootloader, got {len(data)}"
        )
    return StripResult(
        sigtype=SignatureType.NOOK if name == "NOOK" else SignatureType.NOOKTAB,
        image=data[size:],
        sidecar=data[:size],
    )


def _strip_dhtb(data: bytes) -> StripResult:
    if len(data) <= DHTB_HEADER_SIZE:
        raise SignatureError(
            f"DHTB image must exceed the {DHTB_HEADER_SIZE}-byte header, got {len(data)}"
        )
    return StripResult(sigtype=SignatureType.DHTB, image=data[DHTB_HEADER_SIZE:])


def _strip_blob(data: bytes) -> StripResult:
    blobunpack = _tool("blobunpack")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        target = work / "file"
        target.write_bytes(data)
        stdout = _run([str(blobunpack), "file"], cwd=work)
        members = sorted(work.glob("file.*"))
        if not members:
            raise SignatureError("blobunpack produced no partition members")
        name = _parse_blob_name(stdout)
        parts = _parse_blob_partitions(stdout, members)
        # ASUS blobs normally carry a single partition, but blobunpack happily
        # emits several (verified: a 2-partition blob yields file.boo and
        # file.rec).  Picking members[-1] would silently discard the others, so
        # expose every partition and only default `image` when there is one.
        primary = parts.get(name)
        return StripResult(
            sigtype=SignatureType.BLOB,
            image=(primary if primary is not None else members[0]).read_bytes(),
            sidecar=name.encode("ascii")[:BLOB_NAME_LEN],
            members=tuple(m.read_bytes() for m in members),
            partitions={k: v.read_bytes() for k, v in parts.items()},
        )


def _parse_blob_name(stdout: str) -> str:
    """Extract the partition name the way the Bash pipeline does.

    ``unpackimg.sh`` pipes stdout through ``tail -n+5 | cut -d" " -f2`` and then
    truncates to three bytes, so match the ``Name: <part>`` line itself instead
    of trusting a fixed line offset -- blobunpack's banner length varies with
    the container version.  A name that cannot be found is an error: guessing
    would emit a blob keyed to a partition name the bootloader will reject.
    """
    for line in stdout.splitlines():
        if line.startswith("Name:"):
            return line.split(" ", 1)[1].strip()[:BLOB_NAME_LEN]
    raise SignatureError(
        f"could not read partition name from blobunpack output: {stdout!r}"
    )


def _parse_blob_partitions(stdout: str, members: Sequence[Path]) -> dict[str, Path]:
    """Map partition name -> unpacked member file, per blobunpack's stdout."""
    paths = {p.name[len("file.") :]: p for p in members}
    found: dict[str, Path] = {}
    for line in stdout.splitlines():
        if not line.startswith("Name:"):
            continue
        name = line.split(" ", 1)[1].strip()[:BLOB_NAME_LEN]
        if name in paths:
            found[name] = paths[name]
    return found


def _strip_chromeos(data: bytes) -> StripResult:
    futility = _tool("futility")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src, dst = work / "in.img", work / "out.vmlinuz"
        src.write_bytes(data)
        _run(
            [
                str(futility),
                "vbutil_kernel",
                "--get-vmlinuz",
                src.name,
                "--vmlinuz-out",
                dst.name,
            ],
            cwd=work,
        )
        if not dst.is_file():
            raise SignatureError("futility produced no vmlinuz output")
        return StripResult(sigtype=SignatureType.CHROMEOS, image=dst.read_bytes())


#: sony_dump names its members ``<image>-<part>`` (e.g. ``boot.img-zImage``),
#: *not* ``<image>.<part>``.  The Bash glob ``"$file."*`` only matches because
#: ``$file`` is ``boot.img``, so the dot in ``.img`` supplies the separator.
#: Globbing ``<image>-*`` is what actually matches the tool's real output.
def _strip_sin(data: bytes, sigtype: SignatureType) -> StripResult:
    sony_dump = _tool("sony_dump")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        # Mirror the Bash, which hands the tool the image under its own
        # basename and then collects that basename plus a separator.
        name = "image.img"
        (work / name).write_bytes(data)
        _run([str(sony_dump), ".", name], cwd=work, allow_abort=True)
        members = sorted(work.glob(f"{name}-*"))
        if not members:
            raise SignatureError("sony_dump produced no members")
        # sony_dump decomposes a SIN container into its parts (kernel, ramdisk,
        # cmdline, offsets, ...).  There is no single member that reassembles
        # the image, so hand back every part and let the caller repack; the
        # Bash's `mv -f "$file."* "$file"` was a glob that silently failed with
        # "Not a directory" whenever more than one member existed.
        return StripResult(
            sigtype=sigtype,
            image=members[0].read_bytes(),
            members=tuple(m.read_bytes() for m in members),
        )


_STRIPPERS: dict[SignatureType, Callable[[bytes, SignatureType], StripResult]] = {
    SignatureType.NOOK: lambda d, _: _strip_nook(d, NOOK_BOOTLOADER_SIZE, "NOOK"),
    SignatureType.NOOKTAB: lambda d, _: _strip_nook(
        d, NOOKTAB_BOOTLOADER_SIZE, "NOOKTAB"
    ),
    SignatureType.DHTB: lambda d, _: _strip_dhtb(d),
    SignatureType.BLOB: lambda d, _: _strip_blob(d),
    SignatureType.CHROMEOS: lambda d, _: _strip_chromeos(d),
    SignatureType.SINV1: _strip_sin,
    SignatureType.SINV2: _strip_sin,
    SignatureType.SINV3: _strip_sin,
}


def strip(data: bytes, sigtype: SignatureType | str) -> StripResult:
    """Remove ``sigtype``'s wrapper from ``data``.

    ``sigtype`` may be a :class:`SignatureType` or the raw string libmagic
    reported (``"NOOK"``, ``"SINv3"``, ...).
    """
    try:
        resolved = SignatureType(sigtype)
    except ValueError as exc:
        raise SignatureError(f"unknown signature type {sigtype!r}") from exc
    if len(data) == 0:
        raise SignatureError("cannot strip a signature wrapper from empty data")
    return _STRIPPERS[resolved](data, resolved)


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------


def _apply_blob(data: bytes, result: StripResult) -> bytes:
    blobpack = _tool("blobpack")
    name = result.blob_name
    if not name:
        raise SignatureError(
            "BLOB repack needs the partition name from the strip result"
        )
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "image").write_bytes(data)
        _run([str(blobpack), "blob.tmp", name, "image"], cwd=work)
        packed = work / "blob.tmp"
        if not packed.is_file():
            raise SignatureError("blobpack produced no output")
        return BLOB_HEADER + packed.read_bytes()


def _apply_chromeos(data: bytes, result: StripResult) -> bytes:
    futility = _tool("futility")
    root = _bin_root()
    keyblock, signkey = (
        root / "chromeos" / "kernel.keyblock",
        root / "chromeos" / "kernel_data_key.vbprivk",
    )
    empty = root / "chromeos" / "empty"
    for path in (keyblock, signkey, empty):
        if not path.is_file():
            raise MissingToolError(f"ChromeOS signing material missing: {path}")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "vmlinuz").write_bytes(data)
        out = work / "kernel.blob"
        _run(
            [
                str(futility),
                "vbutil_kernel",
                "--pack",
                out.name,
                "--keyblock",
                str(keyblock),
                "--signprivate",
                str(signkey),
                "--version",
                "1",
                "--vmlinuz",
                "vmlinuz",
                "--bootloader",
                str(empty),
                "--config",
                str(empty),
                # repackimg.sh hardcodes --arch arm; the bundled key is arm-only.
                "--arch",
                "arm",
                "--flags",
                "0x1",
            ],
            cwd=work,
        )
        if not out.is_file():
            raise SignatureError("futility produced no packed kernel")
        return out.read_bytes()


def apply(data: bytes, result: StripResult) -> bytes:
    """Re-wrap ``data`` using the metadata recorded in ``result``."""
    sigtype = result.sigtype
    if sigtype in (SignatureType.NOOK, SignatureType.NOOKTAB):
        if not result.sidecar:
            raise SignatureError(
                f"{sigtype} repack needs the stashed bootloader from strip()"
            )
        return result.sidecar + data
    if sigtype == SignatureType.DHTB:
        # dhtbsign regenerates the DHTB header itself, so no bytes are prepended.
        # repackimg.sh also deletes the recorded tailtype here, so a DHTB footer
        # is not appended on top of the signed header.
        return _apply_dhtb(data)
    if sigtype == SignatureType.BLOB:
        return _apply_blob(data, result)
    if sigtype == SignatureType.CHROMEOS:
        return _apply_chromeos(data, result)
    if sigtype.is_sin:
        raise SignatureError(
            f"{sigtype} is a packaging container, not a signature; "
            "unpackimg.sh discards the sigtype after unpacking, so nothing is re-applied"
        )
    raise SignatureError(f"cannot apply signature type {sigtype!r}")


def _apply_dhtb(data: bytes) -> bytes:
    dhtbsign = _tool("dhtbsign")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "in.img").write_bytes(data)
        out = work / "out.img"
        _run([str(dhtbsign), "-i", "in.img", "-o", "out.img"], cwd=work)
        if not out.is_file():
            raise SignatureError("dhtbsign produced no output")
        return out.read_bytes()
