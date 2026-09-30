"""MediaTek (MTK) 512-byte junk header, prepended to kernels and ramdisks.

MediaTek and some Qualcomm "Quark" devices prefix a 512-byte header onto the
kernel and/or the ramdisk inside a boot image. AIK strips it on unpack and
regenerates it on repack; the header carries the payload size and a type tag,
so a stripped component cannot be round-tripped without reconstructing it.

The layout below was recovered empirically by diffing the output of the
bundled ``mkmtkhdr`` binary; :func:`build` reproduces it byte for byte.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "HEADER_SIZE",
    "MAGIC",
    "MIN_PAYLOAD_SIZE",
    "MtkError",
    "MtkHeaderError",
    "MtkHeaders",
    "MtkToolError",
    "MtkType",
    "apply",
    "apply_subprocess",
    "build",
    "detect_type",
    "has_header",
    "payload_size",
    "record_type",
    "resolve_mkmtkhdr",
    "strip",
]

#: Fixed prefix length of an MTK header; the payload starts immediately after.
HEADER_SIZE = 512

#: Magic searched at offset 0 of a split kernel or ramdisk (``bin/androidbootimg.magic``).
MAGIC = b"\x88\x16\x88\x58"

#: Smallest payload ``mkmtkhdr`` will header; it rejects anything shorter, so
#: :func:`build` does too in order to stay byte-compatible.
MIN_PAYLOAD_SIZE = 4

_TYPE_FIELD_SIZE = 32
_TYPE_NAMES = {"KERNEL": "kernel", "ROOTFS": "rootfs", "RECOVERY": "recovery"}

try:  # pragma: no cover - exercised only once the shared module lands
    from aik_py.errors import AikError as MtkError
except ImportError:  # pragma: no cover - standalone fallback

    class MtkError(Exception):
        """Base error for MTK header handling."""


class MtkHeaderError(MtkError):
    """Raised when a buffer cannot be interpreted as an MTK header."""


class MtkToolError(MtkError):
    """Raised when the bundled ``mkmtkhdr`` binary fails."""


class MtkType(StrEnum):
    """Component kind tagged inside the header.

    The values are the exact tokens AIK records in ``*-mtktype`` and
    concatenates into the ``--<type>`` flag passed to ``mkmtkhdr``.
    """

    KERNEL = "kernel"
    ROOTFS = "rootfs"
    RECOVERY = "recovery"

    @property
    def tag(self) -> str:
        """Uppercase NUL-padded type string as stored in the header."""
        return self.name


def has_header(data: bytes) -> bool:
    """Return whether ``data`` is a complete MTK header.

    Requires the full 512 bytes, not just the magic: a truncated extract must
    not be classified as headered, because the matching :func:`strip` would
    then refuse it and the component would be lost mid-unpack.
    """
    return len(data) >= HEADER_SIZE and data.startswith(MAGIC)


def detect_type(data: bytes) -> MtkType | None:
    """Return the tagged component type, or ``None`` if this is not an MTK header.

    A recognised type is not implied by the magic: a header whose tag is
    unreadable or unrecognised is still stripped by AIK, so the header is
    present while the type is unknown.
    """
    if not has_header(data):
        return None
    try:
        name = data[8 : 8 + _TYPE_FIELD_SIZE].split(b"\x00", 1)[0].decode("ascii")
    except UnicodeDecodeError:
        return None
    value = _TYPE_NAMES.get(name)
    return MtkType(value) if value is not None else None


def payload_size(data: bytes) -> int | None:
    """Return the size the header claims for its payload, if this is a header."""
    if not has_header(data):
        return None
    return struct.unpack_from("<I", data, 4)[0]


def build(payload: bytes, mtk_type: MtkType) -> bytes:
    """Prefix ``payload`` with a freshly generated 512-byte MTK header.

    Reproduces ``mkmtkhdr`` byte for byte: magic, little-endian payload length,
    a 32-byte NUL-padded type tag, then 0xFF fill out to the fixed 512 bytes.
    """
    if len(payload) < MIN_PAYLOAD_SIZE:
        raise MtkHeaderError(
            f"payload is {len(payload)} bytes; mkmtkhdr requires at least {MIN_PAYLOAD_SIZE}"
        )
    if has_header(payload):
        raise MtkHeaderError("payload already carries an MTK header; strip it first")
    header = (
        MAGIC
        + struct.pack("<I", len(payload))
        + mtk_type.tag.encode("ascii").ljust(_TYPE_FIELD_SIZE, b"\x00")
        + b"\xff" * (HEADER_SIZE - 8 - _TYPE_FIELD_SIZE)
    )
    return header + payload


def strip(data: bytes) -> bytes:
    """Remove the 512-byte header, returning the payload unchanged.

    Refuses a buffer that is not a header rather than discarding 512 bytes of
    a genuine kernel or ramdisk, which the ``dd bs=512 skip=1`` original does
    without complaint.
    """
    if not has_header(data):
        raise MtkHeaderError(
            f"buffer does not start with the MTK magic {MAGIC.hex()}"
            if len(data) >= HEADER_SIZE
            else f"buffer is {len(data)} bytes, too short to hold a {HEADER_SIZE}-byte MTK header"
        )
    return data[HEADER_SIZE:]


@dataclass(frozen=True, slots=True)
class MtkHeaders:
    """A kernel and ramdisk pair re-headered together."""

    kernel: bytes
    ramdisk: bytes


def apply(kernel: bytes, ramdisk: bytes, mtk_type: MtkType) -> MtkHeaders:
    """Regenerate the MTK headers for a kernel/ramdisk pair.

    The two headers are always written as a pair, matching ``mkmtkhdr``, which
    only accepts ``--kernel`` alongside a ramdisk. Rockchip ``KRNL`` images
    have no kernel component and therefore never reach this path.
    """
    return MtkHeaders(
        kernel=build(kernel, MtkType.KERNEL),
        ramdisk=build(ramdisk, mtk_type),
    )


def record_type(
    kernel: bytes,
    ramdisk: bytes | None,
) -> tuple[MtkType | None, tuple[str, ...]]:
    """Decide the ramdisk type AIK records while unpacking.

    Returns the type to persist and the warnings AIK prints. ``None`` means no
    MTK header was present on either component, so nothing is recorded and the
    repack side stays on its non-MTK path entirely.

    The fallback to ``ROOTFS`` mirrors AIK: when the kernel carries a header but
    the ramdisk type is unknowable, a plain rootfs is the safe reading of a
    non-recovery image.
    """
    kernel_mtk = has_header(kernel)
    ramdisk_mtk = ramdisk is not None and has_header(ramdisk)
    if not kernel_mtk and not ramdisk_mtk:
        return None, ()

    warnings: list[str] = []
    if not kernel_mtk:
        warnings.append("Warning: No MTK header found in kernel!")

    ramdisk_type = detect_type(ramdisk) if ramdisk_mtk else None
    if ramdisk_type is not None:
        return ramdisk_type, tuple(warnings)

    # Deliberate divergence: the Bash records an empty type here when the
    # ramdisk carries an unrecognised tag, and repackimg.sh then builds the
    # flag as "--", which mkmtkhdr rejects. Assuming rootfs keeps that image
    # repackable instead of failing the whole job over an ambiguous tag.
    warnings.append('Warning: No MTK header found in ramdisk, assuming "rootfs" type!')
    return MtkType.ROOTFS, tuple(warnings)


def resolve_mkmtkhdr(tool: str = "mkmtkhdr") -> Path | None:
    """Locate the bundled ``mkmtkhdr`` binary, or ``None`` if it is absent.

    AIK ships the tool per platform under ``bin/<os>/<arch>/``; this mirrors
    that lookup so the cross-check path works from a source checkout.
    """
    try:
        from aik_py.bin import resolve_tool
    except ImportError:
        pass
    else:
        try:
            return resolve_tool(tool)
        except (LookupError, OSError, ValueError):
            return None

    if sys.platform.startswith("win"):
        platform_dir, name = "windows/x86_64", f"{tool}.exe"
    elif sys.platform == "darwin":
        platform_dir, name = "macos/x86_64", tool
    else:
        platform_dir, name = "linux/x86_64", tool

    repo_root = Path(__file__).resolve().parents[3]
    for base in (repo_root / "bin", Path.cwd() / "bin"):
        candidate = base / platform_dir / name
        if candidate.is_file():
            return candidate
    return None


def apply_subprocess(
    kernel: bytes,
    ramdisk: bytes,
    mtk_type: MtkType,
    *,
    tool: str = "mkmtkhdr",
) -> MtkHeaders:
    """Regenerate headers by shelling out to the bundled ``mkmtkhdr``.

    Retained as a cross-check for :func:`build`; :func:`apply` is preferred
    because it needs no subprocess and no temporary files.
    """
    if mtk_type not in (MtkType.ROOTFS, MtkType.RECOVERY):
        raise MtkHeaderError(f"{mtk_type!s} is not a ramdisk type accepted by {tool}")
    path = resolve_mkmtkhdr(tool)
    if path is None:
        raise MtkToolError(f"could not locate the bundled {tool} binary")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        kernel_path = work / "kernel.bin"
        ramdisk_path = work / "rootfs.bin"
        kernel_path.write_bytes(kernel)
        ramdisk_path.write_bytes(ramdisk)
        try:
            proc = subprocess.run(
                [
                    str(path),
                    "--kernel",
                    str(kernel_path),
                    f"--{mtk_type}",
                    str(ramdisk_path),
                ],
                cwd=work,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            # A stripped exec bit or a truncated download resolves but will not
            # run; that is still a tool failure, not a caller error.
            raise MtkToolError(f"could not execute {path}: {exc}") from exc
        if proc.returncode != 0:
            detail = proc.stderr.decode(errors="replace").strip() or "no error output"
            raise MtkToolError(f"{tool} failed ({proc.returncode}): {detail}")
        out_kernel = kernel_path.with_name(f"{kernel_path.name}-mtk")
        out_ramdisk = ramdisk_path.with_name(f"{ramdisk_path.name}-mtk")
        if not (out_kernel.is_file() and out_ramdisk.is_file()):
            raise MtkToolError(f"{tool} exited 0 but produced no output files")
        return MtkHeaders(
            kernel=out_kernel.read_bytes(),
            ramdisk=out_ramdisk.read_bytes(),
        )
