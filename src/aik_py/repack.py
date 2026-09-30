"""Repack path: mirror of ``repackimg.sh``.

Rebuilds ``ramdisk/`` into a cpio newc archive, re-applies the OEM wrappers the
unpack path stripped, builds the boot image in the recorded format, signs it,
patches it, appends the footer and optionally pads back to the original size.

Stage order is load-bearing and mirrors the Bash exactly; the one deliberate
deviation is that ``--origsize`` refuses to truncate signed data instead of
silently chopping a footer or signature off the image (see :func:`pad_to_size`).
"""

from __future__ import annotations

import bz2
import gzip
import importlib
import lzma
import os
import shutil
import stat
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

__all__ = [
    "MTK_MARKS",
    "CpioEntry",
    "RepackError",
    "RepackOptions",
    "RepackPlan",
    "compress_ramdisk",
    "compression_extension",
    "empty_ramdisk",
    "pack_ramdisk",
    "pad_to_size",
    "parse_newc",
    "read_newc",
    "repack",
    "repack_ramdisk",
    "resolve_level",
    "write_newc",
    "write_newc_entry",
]

NEWC_MAGIC = b"070701"
NEWC_TRAILER = b"TRAILER!!!\0"
NEWC_HEADER_SIZE = 110
NEWC_TRAILING_PAD = 512

#: Magic stamped on a Bump image so bootloaders can recognise the footer.
BUMP_FOOTER = b"\x41\xa9\xe4\x67\x74\x4d\x1d\x1b\xa4\x29\xf2\xec\xea\x65\x52\x79"
#: SEAndroid enforcing marker, appended verbatim.
SEANDROID_FOOTER = b"SEANDROIDENFORCE"

#: Sentinel compression types that mean "do not build a cpio at all".
COMP_EMPTY = "empty"
COMP_CPIO = "cpio"

#: The Bash records these verbatim; we only need to recognise MTK variants.
MTK_MARKS = ("boot", "recovery", "rootfs", "vendor")

_DEFAULT_LEVELS = {
    "xz": 1,
    "lz4": 9,
    "lz4-l": 9,
}
_DEFAULT_LEVEL_MARK = 0xFFFF_FFFF
_MAX_LEVEL = 9


class RepackError(RuntimeError):
    """Raised for every condition the Bash would have reported as ``Error!``."""


# --------------------------------------------------------------------------
# cpio newc writer / reader
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CpioEntry:
    """One member of a cpio newc archive.

    ``ino``/``nlink``/``dev_*`` are carried explicitly because ``cpio -o``
    copies them straight off the filesystem and a bootloader does not care
    about their values, but ``cpio -i`` and byte-exact round trips do.
    """

    name: str
    mode: int
    data: bytes = b""
    ino: int = 0
    nlink: int = 1
    mtime: int = 0
    uid: int = 0
    gid: int = 0
    devmaj: int = 0
    devmin: int = 0
    rdevmaj: int = 0
    rdevmin: int = 0

    @property
    def filesize(self) -> int:
        return len(self.data)


def _hex8(value: int) -> bytes:
    return f"{value & 0xFFFF_FFFF:08X}".encode("ascii")


def write_newc_entry(entry: CpioEntry) -> bytes:
    """Serialise a single member, including the 4-byte header pad.

    Field order is fixed by the format: 110 bytes of 13 hex-8 words after the
    6-byte ``070701`` magic, then the NUL-terminated name, padded to 4.
    """
    name = entry.name.encode("utf-8") + b"\0"
    fields = (
        entry.ino,
        entry.mode,
        entry.uid,
        entry.gid,
        entry.nlink,
        entry.mtime,
        entry.filesize,
        entry.devmaj,
        entry.devmin,
        entry.rdevmaj,
        entry.rdevmin,
        len(name),
        0,
    )
    header = NEWC_MAGIC + b"".join(_hex8(f) for f in fields) + name
    header += b"\0" * ((-len(header)) % 4)
    payload = entry.data + b"\0" * ((-entry.filesize) % 4)
    return header + payload


def _trailer() -> bytes:
    return write_newc_entry(
        CpioEntry(name="TRAILER!!!", mode=0, ino=0, nlink=1, devmin=0)
    )


def write_newc(entries: Sequence[CpioEntry]) -> bytes:
    """Build a complete newc archive, zero-padded to 512 bytes.

    GNU cpio always pads the tail to the 512-byte block size, so anything else
    produces archives that differ from the Bash output byte for byte.
    """
    blob = b"".join(write_newc_entry(e) for e in entries) + _trailer()
    return blob + b"\0" * (-len(blob) % NEWC_TRAILING_PAD)


def _u32(buf: bytes, off: int) -> int:
    return int(buf[off : off + 8], 16)


def read_newc(blob: bytes) -> list[CpioEntry]:
    """Parse a newc archive. Used by the tests and by the round-trip check."""
    entries: list[CpioEntry] = []
    off = 0
    end = len(blob)
    while off + NEWC_HEADER_SIZE <= end:
        if blob[off : off + 6] != NEWC_MAGIC:
            break
        v = [_u32(blob, off + 6 + i * 8) for i in range(13)]
        namesize = v[11]
        name_at = off + NEWC_HEADER_SIZE
        name = blob[name_at : name_at + namesize].rstrip(b"\0").decode("utf-8")
        data_at = name_at + namesize
        data_at += (-(data_at - off)) % 4
        data = blob[data_at : data_at + v[6]]
        if name == "TRAILER!!!":
            break
        entries.append(
            CpioEntry(
                name=name,
                mode=v[1],
                data=data,
                ino=v[0],
                nlink=v[4],
                mtime=v[5],
                uid=v[2],
                gid=v[3],
                devmaj=v[7],
                devmin=v[8],
                rdevmaj=v[9],
                rdevmin=v[10],
            )
        )
        off = data_at + v[6] + (-v[6]) % 4
    return entries


parse_newc = read_newc


# --------------------------------------------------------------------------
# ramdisk walking
# --------------------------------------------------------------------------


def _iter_ramdisk(root: Path) -> Iterator[Path]:
    """Yield ``root`` and every descendant, sorted, never following symlinks.

    The Bash pipes bare ``find .`` into cpio, so its output order is whatever
    the filesystem returns. Sorting the complete relative-path list makes the
    archive byte-reproducible across machines, which is the whole point of a
    round trip.

    The sort must be global over the whole relative path, not per-directory:
    sorting siblings while descending a stack would emit ``b``'s children
    before ``a``'s, and a per-directory walk groups by depth. Sorting the
    finished list also puts every parent ahead of its children, since
    ``a/m`` sorts after ``a``.
    """
    descendants: list[Path] = []
    for parent, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(parent)
        descendants += [base / name for name in dirnames + filenames]
    yield root
    yield from sorted(descendants, key=lambda p: p.relative_to(root).parts)


def _entry_for(path: Path, root: Path) -> CpioEntry:
    st = path.lstat()
    rel = os.path.relpath(path, root)
    mode = stat.S_IMODE(st.st_mode)
    # GNU cpio copies st_dev verbatim into the newc dev fields; matching it
    # keeps our archives byte-comparable with the ones repackimg.sh produces.
    devmaj, devmin = os.major(st.st_dev), os.minor(st.st_dev)
    common = {
        "ino": st.st_ino,
        "nlink": st.st_nlink,
        "mtime": int(st.st_mtime),
        "devmaj": devmaj,
        "devmin": devmin,
    }
    if stat.S_ISLNK(st.st_mode):
        return CpioEntry(
            name=rel,
            mode=stat.S_IFLNK | 0o777,
            data=os.readlink(path).encode("utf-8"),
            **common,
        )
    if stat.S_ISDIR(st.st_mode):
        return CpioEntry(
            name=rel,
            mode=stat.S_IFDIR | mode,
            **common,
        )
    if not stat.S_ISREG(st.st_mode):
        raise RepackError(f"unsupported file type in ramdisk: {path}")
    return CpioEntry(
        name=rel,
        mode=stat.S_IFREG | mode,
        data=path.read_bytes(),
        **common,
    )


def build_archive(root: Path) -> bytes:
    """Walk ``root`` and return the raw cpio newc bytes.

    Every member gets uid=0/gid=0 written directly, which is what the Bash
    achieves with ``cpio -R 0:0`` on a non-root tree. Because we never shell
    out, the Bash's ``sudo`` branch is unnecessary: there is no external tool
    to escalate on behalf of, and a root-owned ``ramdisk/`` is read fine by the
    same unprivileged process that owns the pack. ``unpackimg.sh`` still needs
    sudo to *restore* ownership, but that is the other unit's problem.
    """
    if not root.is_dir():
        raise RepackError(f"ramdisk directory not found: {root}")
    entries = [_entry_for(p, root) for p in _iter_ramdisk(root)]
    return write_newc(entries)


# --------------------------------------------------------------------------
# compression
# --------------------------------------------------------------------------

_EXTENSION_FOR_COMP = {
    "gzip": "gz",
    "lzop": "lzo",
    "bzip2": "bz2",
    "lz4": "lz4",
    "lz4-l": "lz4",
    "cpio": "",
    "xz": "xz",
    "lzma": "lzma",
    "empty": "empty",
}


def compression_extension(comp: str) -> str:
    """Return the ``.ext`` (with leading dot) the Bash appends to the archive."""
    if comp not in _EXTENSION_FOR_COMP:
        raise RepackError(f"Unsupported format: compression '{comp}'")
    ext = _EXTENSION_FOR_COMP[comp]
    return f".{ext}" if ext else ""


def resolve_level(comp: str, level: int | None) -> int:
    """Resolve the effective compression level.

    xz defaults to 1 and lz4 to 9 to match the Bash; everything else keeps the
    compressor's own default, expressed as "no level".

    The Bash silently discards a non-numeric ``--level`` and carries on at the
    default. That is a bug: a typo in a build recipe produces a subtly
    different image with no diagnostic, so we raise instead.
    """
    if level is not None:
        if isinstance(level, bool) or not isinstance(level, int):
            raise RepackError(f"compression level must be an integer, got {level!r}")
        if not 0 <= level <= _MAX_LEVEL:
            raise RepackError(
                f"compression level {level} out of range, expected 0-{_MAX_LEVEL}"
            )
        return level
    if comp in ("xz", "lzma"):
        return _DEFAULT_LEVELS["xz"]
    if comp in ("lz4", "lz4-l"):
        return _DEFAULT_LEVELS["lz4"]
    return _DEFAULT_LEVEL_MARK


def _lz4_compress(data: bytes, level: int, legacy: bool) -> bytes:
    """lz4 has no stdlib codec; shell out to whichever lz4 is installed.

    Prefers AIK's bundled binary because the unpack path reads the archives it
    produced, then the system lz4.

    ``-c -`` is required, not ``-``: bare ``lz4 -N -`` treats ``-`` as a name
    of a *file list* to compress rather than as stdin.
    """
    argv = [f"-{level}"]
    if legacy:
        argv.append("-l")
    argv += ["-c", "-"]
    candidates = [_bundled_binary("lz4"), "lz4"]
    errors: list[str] = []
    for exe in candidates:
        cmd = [exe or "lz4", *argv]
        try:
            proc = subprocess.run(cmd, input=data, capture_output=True, check=False)
        except OSError as exc:
            errors.append(f"{' '.join(cmd)}: {exc}")
            continue
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        errors.append(
            f"{' '.join(cmd)}: rc={proc.returncode} "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    raise RepackError(
        f"lz4 compression requires the lz4 binary (tried {'; '.join(errors)})"
    )


def _bundled_path(*parts: str) -> Path | None:
    """Locate a file under ``bin/`` next to the package, if it is present."""
    root = Path(__file__).resolve().parent.parent.parent
    plat = "macos" if os.uname().sysname == "Darwin" else "linux"
    candidate = root.joinpath("bin", plat, os.uname().machine, *parts)
    return candidate if candidate.exists() else None


def _bundled_binary(name: str) -> str | None:
    """The bundled executable of that name, if it exists and is runnable."""
    candidate = _bundled_path(name)
    if candidate is None or not os.access(candidate, os.X_OK):
        return None
    return str(candidate)


def compress_ramdisk(data: bytes, comp: str, level: int | None = None) -> bytes:
    """Compress cpio ``data`` with the recorded compression type."""
    effective = resolve_level(comp, level)
    match comp:
        case "gzip":
            # mtime=0 and no FNAME, matching the Bash's `gzip -N`-style output.
            return gzip.compress(data, compresslevel=_clamp(effective, 9), mtime=0)
        case "xz":
            # The Bash uses `xz -Ccrc32`, so match the check byte exactly.
            return lzma.compress(
                data, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC32, preset=effective
            )
        case "lzma":
            return lzma.compress(data, format=lzma.FORMAT_ALONE, preset=effective)
        case "bzip2":
            return bz2.compress(data, _clamp(effective, 9))
        case "lzop":
            return _lzop_compress(data, effective)
        case "lz4":
            return _lz4_compress(data, effective, legacy=False)
        case "lz4-l":
            return _lz4_compress(data, effective, legacy=True)
        case "cpio" | "empty":
            return data
        case _:
            raise RepackError(f"Unsupported format: compression '{comp}'")


def _clamp(effective: int, default: int) -> int:
    """Turn the "compressor default" sentinel back into a real level."""
    return default if effective == _DEFAULT_LEVEL_MARK else effective


def _lzop_compress(data: bytes, level: int) -> bytes:
    """lzo has no stdlib codec; shell out to the bundled or system lzop."""
    argv = ["lzop", f"-{_clamp(level, 1)}", "-c", "-"]
    bundled = _bundled_binary("lzop")
    errors: list[str] = []
    for exe in (bundled, "lzop"):
        cmd = [exe or "lzop", *argv[1:]]
        try:
            proc = subprocess.run(cmd, input=data, capture_output=True, check=False)
        except OSError as exc:
            errors.append(f"{' '.join(cmd)}: {exc}")
            continue
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout
        errors.append(
            f"{' '.join(cmd)}: rc={proc.returncode} "
            f"{proc.stderr.decode(errors='replace').strip()}"
        )
    raise RepackError(
        f"lzop compression requires the lzop binary (tried {'; '.join(errors)})"
    )


def empty_ramdisk() -> bytes:
    """A zero-length stand-in the Bash writes as ``ramdisk-new.cpio.empty``."""
    return b""


def pack_ramdisk(root: Path, comp: str, level: int | None = None) -> bytes:
    """Walk ``root``, build a cpio newc archive and compress it."""
    if not root.is_dir():
        raise RepackError(f"ramdisk directory not found: {root}")
    if not any(root.iterdir()):
        raise RepackError(f"ramdisk directory is empty: {root}")
    return compress_ramdisk(build_archive(root), comp, level)


# --------------------------------------------------------------------------
# options and the manifest contract
# --------------------------------------------------------------------------


@runtime_checkable
class ManifestLike(Protocol):
    """The slice of ``aik_py.manifest`` this module actually consumes.

    The unpack unit owns the real manifest; depending on a Protocol keeps this
    unit importable and testable before that module lands. Any object with
    these attributes works, including a plain dataclass or ``types.SimpleNamespace``.
    """

    img_type: str
    ramdisk_compression: str
    original_size: int | None


@dataclass(slots=True)
class RepackOptions:
    """The Bash's ``repackimg.sh`` flags, as keyword arguments."""

    original: bool = False
    origsize: bool = False
    level: int | None = None
    avbkey: str | None = None
    forceelf: bool = False
    aboot: Path | None = None

    def __post_init__(self) -> None:
        # Delegate so the eager check and the per-codec check can never drift.
        if self.level is not None:
            resolve_level("gzip", self.level)


@dataclass(slots=True)
class RepackPlan:
    """Intermediate state threaded through the repack stages.

    Exposed so callers (and the CLI) can see what each stage decided, e.g.
    which output name the signature fork picked.
    """

    ramdisk: Path
    kernel: Path | None = None
    output: Path | None = None
    unsigned: Path | None = None
    comp_extension: str = ""
    used_original_ramdisk: bool = False
    is_empty_ramdisk: bool = False
    img_type: str | None = None

    @property
    def signed(self) -> bool:
        return self.unsigned is not None


def _flag(plan_work: Path, suffix: str) -> str | None:
    """Read a ``<image>-<suffix>`` marker from a split_img-style directory."""
    matches = sorted(plan_work.glob(f"*-{suffix}"))
    if not matches:
        return None
    return matches[0].read_text().strip()


# --------------------------------------------------------------------------
# pipeline stages
# --------------------------------------------------------------------------


def clear_scratch(work: Path, pattern: str = "*-new.*") -> list[Path]:
    """Delete stale ``*-new.*`` scratch files, as the Bash does before packing.

    Returns the removed paths so the caller can warn about overwriting, which
    the Bash does by listing the glob before deleting it.
    """
    removed: list[Path] = []
    for path in sorted(work.glob(pattern)):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def _ramdisk_path(work: Path, comp: str) -> Path:
    return work / f"ramdisk-new.cpio{compression_extension(comp)}"


def repack_ramdisk(
    work: Path,
    plan: RepackPlan,
    comp: str,
    original_archive: Path | None,
    level: int | None,
) -> Path:
    """Stage 3: produce the ramdisk archive the image build will embed."""
    if original_archive is not None:
        logger.info("Repacking with original ramdisk...")
        plan.comp_extension = compression_extension(comp)
        plan.used_original_ramdisk = True
        return original_archive
    if comp == COMP_EMPTY or plan.is_empty_ramdisk:
        logger.warning("Using empty ramdisk for repack!")
        path = work / "ramdisk-new.cpio.empty"
        path.write_bytes(empty_ramdisk())
        plan.comp_extension = ".empty"
        plan.is_empty_ramdisk = True
        return path
    source = work / "ramdisk"
    logger.info("Packing ramdisk...")
    logger.info(f"Using compression: {comp}")
    path = _ramdisk_path(work, comp)
    path.write_bytes(pack_ramdisk(source, comp, level))
    plan.comp_extension = compression_extension(comp)
    return path


def apply_mtk(work: Path, plan: RepackPlan, mtktype: str | None) -> RepackPlan:
    """Stage 4: re-wrap kernel and ramdisk in their MTK headers.

    AIK calls the prebuilt ``mkmtkhdr`` binary; we defer to
    ``aik_py.patches`` when it exists and otherwise report the gap rather than
    silently producing an image with the header missing.
    """
    if not mtktype:
        return plan
    if mtktype not in MTK_MARKS:
        raise RepackError(f"unsupported MTK ramdisk type: {mtktype}")
    logger.info("Generating MTK headers...")
    logger.info(f"Using ramdisk type: {mtktype}")
    if plan.kernel is None:
        raise RepackError("MTK repack requires a kernel file")
    add_mtk_header = _optional("aik_py.patches", "add_mtk_header", "MTK repack")
    kernel_mtk = work / "kernel-new.mtk"
    ramdisk_mtk = work / f"{mtktype}-new.mtk"
    add_mtk_header(plan.kernel, plan.ramdisk, mtktype, kernel_mtk, ramdisk_mtk)
    plan.kernel = kernel_mtk
    plan.ramdisk = ramdisk_mtk
    return plan


def build_image(plan: RepackPlan, out: Path, manifest: Any) -> Path:
    """Stage 5: build the image in its recorded format.

    ``plan.img_type`` is the post-downgrade format, which may differ from
    ``manifest.img_type`` when an ELF image without an RPM was rebuilt as AOSP.
    """
    build = _optional("aik_py.formats", "build_image", "image building")
    img_type = plan.img_type or getattr(manifest, "img_type", "AOSP")
    logger.info("Building image...")
    logger.info(f"Using format: {img_type}")
    build(
        manifest=manifest,
        out=out,
        kernel=plan.kernel,
        ramdisk=plan.ramdisk,
        img_type=img_type,
    )
    return out


def sign_image(
    plan: RepackPlan, out: Path, options: RepackOptions, manifest: Any
) -> Path:
    """Stage 6: sign ``unsigned`` into ``out``.

    The Bash forks the output name on whether a signature was recorded: with a
    signature the build targets ``unsigned-new.img`` and signing produces
    ``image-new.img``; without one the build writes ``image-new.img`` directly.
    DHTB additionally records a footer suppression, because a DHTB image
    carries its own DHTB footer instead of a Bump/SEAndroid tail.
    """
    sigtype = getattr(manifest, "signature_type", None)
    if not sigtype:
        plan.output = out
        return out
    if plan.unsigned is None:
        raise RepackError("signed repack requires an unsigned image to be built first")
    logger.info("Signing new image...")
    logger.info(f"Using signature: {sigtype}")
    sign = _optional("aik_py.avb", "sign", "image signing")
    sign(
        unsigned=plan.unsigned,
        out=out,
        sigtype=sigtype,
        avbtype=getattr(manifest, "avb_type", None),
        blobtype=getattr(manifest, "blob_type", None),
        avbkey=options.avbkey or _default_avbkey(),
    )
    plan.output = out
    return out


def _default_avbkey() -> str | None:
    """The Bash falls back to its bundled verity key when --avbkey is absent."""
    bundled = _bundled_path("avb", "verity")
    return str(bundled) if bundled is not None else None


def _optional(module: str, attribute: str, purpose: str) -> Any:
    """Fetch ``module.attribute`` from a sibling unit, or explain its absence.

    The format, patch and AVB units are being written concurrently, so this
    module must stay importable and testable without them. When one is missing
    we raise a message naming the exact symbol so the gap is obvious rather
    than surfacing as an opaque ``ImportError`` from deep in a stage.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError:
        mod = None
    fn = getattr(mod, attribute, None)
    if fn is None:
        raise RepackError(
            f"{purpose} requires {module}.{attribute}, which is not available"
        )
    return fn


def apply_loki(
    plan: RepackPlan, image: Path, lokitype: str | None, aboot: Path | None
) -> Path:
    """Stage 7a: re-apply the Loki patch, which needs a device ``aboot.img``."""
    if not lokitype:
        return image
    # Validate the user's input before the dependency: a missing aboot.img is
    # the actionable error, and it must surface even where patches is absent.
    if aboot is None or not Path(aboot).is_file():
        raise RepackError(
            "Device aboot.img required to find the Loki patch offset; "
            "pass RepackOptions(aboot=...)"
        )
    patch = _optional("aik_py.patches", "apply_loki", "Loki repack")
    logger.info("Loki patching new image...")
    logger.info(f"Using type: {lokitype}")
    unlokied = image.with_name("unlokied-new.img")
    image.rename(unlokied)
    patch(unlokied, image, aboot=Path(aboot), lokitype=lokitype)
    return image


def apply_amonet(plan: RepackPlan, image: Path, microloader: Path | None) -> Path:
    """Stage 7b: splice the 1 KiB Amonet microloader over the image's first 2 KiB.

    The Bash slides the first KiB down by one and writes the microloader in at
    offset 0, so the loader lands ahead of the payload.
    """
    if microloader is None:
        return image
    patch = _optional("aik_py.patches", "apply_amonet", "Amonet repack")
    logger.info("Amonet patching new image...")
    unamonet = image.with_name("unamonet-new.img")
    shutil.copyfile(image, unamonet)
    patch(unamonet, image, microloader=Path(microloader))
    return image


def append_footer(image: Path, tailtype: str | None) -> Path:
    """Stage 8: append the Bump or SEAndroid footer."""
    if not tailtype:
        return image
    match tailtype:
        case "Bump":
            logger.info("Appending footer...")
            logger.info("Using type: Bump")
            with image.open("ab") as fh:
                fh.write(BUMP_FOOTER)
        case "SEAndroid":
            logger.info("Appending footer...")
            logger.info("Using type: SEAndroid")
            with image.open("ab") as fh:
                fh.write(SEANDROID_FOOTER)
        case _:
            raise RepackError(f"Unsupported format: footer '{tailtype}'")
    return image


def pad_to_size(image: Path, target: int, *, strict: bool = True) -> int:
    """Stage 9: pad the image out to ``target`` bytes.

    **Deviation from the Bash.** ``repackimg.sh`` copies the image to
    ``unpadded-new.img`` and then runs ``truncate -s $filesize``, which
    *shrinks* the file when the rebuild grew past the original size. Because
    this stage runs last, after signing and footer append, that truncation can
    silently amputate a signature block or footer and yield an image that looks
    fine but will not boot. We refuse instead, leaving the correct image intact
    and saying why; callers who want the Bash's behaviour pass ``strict=False``.

    Returns the resulting size.
    """
    current = image.stat().st_size
    if current == target:
        return current
    if current > target:
        message = (
            f"refusing to pad {image.name} to {target} bytes: rebuilt image is "
            f"{current} bytes, so padding would truncate {current - target} bytes "
            "of signed/padded data"
        )
        if strict:
            raise RepackError(message)
        logger.warning(f"{message}; truncating anyway to match repackimg.sh")
        with image.open("r+b") as fh:
            fh.truncate(target)
        return target
    logger.info("Padding to original size...")
    with image.open("ab") as fh:
        fh.write(b"\0" * (target - current))
    return target


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def repack(
    work: Path,
    manifest: Any,
    options: RepackOptions | None = None,
    *,
    output: Path | None = None,
) -> RepackPlan:
    """Run the full repack pipeline, mirroring ``repackimg.sh`` stage for stage.

    ``work`` is the AIK working directory holding ``ramdisk/`` and
    ``split_img/``-equivalent state; ``manifest`` is whatever
    :class:`ManifestLike` describes.
    """
    opts = options or RepackOptions()
    work = Path(work)

    split = work / "split_img"
    if not split.is_dir():
        split = work

    kernel = _split_file(split, "kernel")
    img_type = _effective_img_type(manifest, split, opts)

    comp = getattr(manifest, "ramdisk_compression", COMP_CPIO)

    # Stage 1: nothing to do means nothing to do.
    if not (work / "ramdisk").is_dir() and not _flag(split, "imgtype"):
        raise RepackError("No files found to be packed/built.")
    if (
        (work / "ramdisk").is_dir()
        and not any((work / "ramdisk").iterdir())
        and comp != COMP_EMPTY
        and not opts.original
    ):
        raise RepackError("No files found to be packed/built.")

    # Stage 2: clear stale scratch, warning first the way the Bash does.
    stale = clear_scratch(work)
    if stale:
        logger.warning(f"Warning: Overwriting existing files! ({len(stale)} removed)")

    out = Path(output) if output else work / "image-new.img"
    unsigned = (
        work / "unsigned-new.img" if getattr(manifest, "signature_type", None) else None
    )

    plan = RepackPlan(ramdisk=work / "ramdisk", kernel=kernel, img_type=img_type)
    plan.unsigned = unsigned
    plan.output = out if unsigned is None else unsigned

    original_archive = _split_ramdisk(split) if opts.original else None
    plan.ramdisk = repack_ramdisk(work, plan, comp, original_archive, opts.level)

    plan = apply_mtk(work, plan, getattr(manifest, "mtk_type", None))

    build_image(plan, plan.output, manifest)

    if unsigned is not None:
        sign_image(plan, out, opts, manifest)

    image = out
    image = apply_loki(plan, image, getattr(manifest, "loki_type", None), opts.aboot)
    image = apply_amonet(plan, image, _split_file(split, "microloader.bin"))

    if getattr(manifest, "signature_type", None) != "DHTB":
        append_footer(image, getattr(manifest, "footer_type", None))

    if opts.origsize:
        target = getattr(manifest, "original_size", None)
        if target:
            pad_to_size(image, int(target))

    plan.output = image
    logger.info("Done!")
    return plan


def _effective_img_type(manifest: Any, split: Path, opts: RepackOptions) -> str:
    """Resolve the format to build, applying the Bash's ELF downgrade.

    ``repackimg.sh`` treats an ELF image as AOSP when no RPM was detected and
    the user did not pass ``--forceelf``, because elftool cannot repack an ELF
    image without its resource pack. Honouring ``--forceelf`` here keeps that
    opt-in explicit rather than silently emitting a mislabelled image.
    """
    img_type = getattr(manifest, "img_type", "AOSP")
    dttype = _flag(split, "dttype")
    if dttype == "ELF":
        opts.forceelf = True
    if img_type == "ELF" and not (opts.forceelf or _split_file(split, "header")):
        logger.warning(
            "Warning: ELF format without RPM detected; "
            "will be repacked using AOSP format!"
        )
        return "AOSP"
    return img_type


def _split_file(split: Path, suffix: str) -> Path | None:
    matches = sorted(split.glob(f"*-{suffix}"))
    if matches:
        return matches[0]
    candidate = split / suffix
    return candidate if candidate.is_file() else None


def _split_ramdisk(split: Path) -> Path | None:
    matches = sorted(split.glob("*-*ramdisk.cpio*"))
    return matches[0] if matches else None
