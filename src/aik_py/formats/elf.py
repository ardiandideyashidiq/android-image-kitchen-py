"""Sony ELF images (``elftool`` / ``unpackelf``).

A Sony ``boot.img`` is an ELF container holding a kernel, a ramdisk and -- on
some devices -- a resource partition (RPM) that carries the device tree.
"""

from __future__ import annotations

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path

from ._tools import Unpacked, run_tool

__all__ = [
    "ELF_MAGIC",
    "QCDT_TYPES",
    "SonyElfState",
    "append_dtb_to_kernel",
    "gzip_bytes",
    "is_elf",
    "pack",
    "pack_arg",
    "should_fallback_to_aosp",
    "unpack",
]

ELF_MAGIC = b"\x7fELF"

#: dt types that keep the device tree as its own ELF part. Anything else gets
#: the lossy append treatment below.
QCDT_TYPES = frozenset({"QCDT", "ELF"})

#: dt types that mean "there is a real RPM program header to pack into".
_ELF_DT_TYPE = "ELF"

_TEMP_DIR = "elftool_out"


def is_elf(data: bytes) -> bool:
    return data[:4] == ELF_MAGIC


def gzip_bytes(data: bytes) -> bytes:
    """Gzip at level 9 without recording a filename or timestamp.

    ``gzip --no-name -9`` in the original: a stored name would make the output
    non-reproducible and would defeat the byte-for-byte comparison users do
    when checking a repack.
    """
    out = bytearray(gzip.compress(data, compresslevel=9, mtime=0))
    # Python stamps the OS field with 0xff (unknown); GNU gzip writes 0x03
    # (Unix). Matching it keeps our output byte-identical to the tool the Bash
    # called, so a repack still diffs cleanly against the original image.
    out[9] = 0x03
    return bytes(out)


def append_dtb_to_kernel(kernel: bytes, dtb: bytes) -> bytes:
    """Gzip the kernel and append the dtb, the QCDT workaround.

    When the device tree is a bare DTB rather than a QCDT/ELF blob, the RPM
    program header the packer needs does not exist, so the dtb is folded into
    the kernel instead. This is lossy: the dtb is destroyed as a separate
    part, and unpacking the result will not yield it back.
    """
    return gzip_bytes(kernel) + dtb


def unpack(
    path: Path,
    workdir: Path,
    *,
    dt_type: str | None = None,
) -> SonyElfState:
    """Split a Sony ELF image into header, kernel, ramdisk, rpm, dt and fields.

    ``dt_type`` is the magic-detected device-tree type (``QCDT``, ``ELF`` or a
    bare-dtb name); the caller owns that detection rather than shelling out to
    ``file(1)`` here. A bare dtb triggers the lossy append below. The default of
    ``None`` deliberately means "not QCDT/ELF" -- the original applies the
    append to every dt that is not QCDT or ELF, with no separate unknown case,
    so the conservative reading is the lossy one rather than silently keeping
    the parts separate.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, Path] = {}

    temp = workdir / _TEMP_DIR
    run_tool("elftool", "unpack", "-i", path.resolve(), "-o", temp.resolve())
    header_src = temp / "header"
    if header_src.is_file():
        header = workdir / "header"
        header_src.replace(header)
        parts["header"] = header
    shutil.rmtree(temp, ignore_errors=True)

    run_tool("unpackelf", "-i", str(path.resolve()), cwd=workdir)
    # unpackelf writes one file per ELF section it recognises. The ramdisk is
    # one of them, so it has to be collected or the repack silently drops it.
    for name in ("base", "pagesize", "cmdline", "kernel", "ramdisk", "dt"):
        src = workdir / f"{path.name}-{name}"
        if src.is_file():
            parts[name] = src

    fields = {
        key: (workdir / f"{path.name}-{key}").read_text().strip()
        for key in ("base", "pagesize", "cmdline")
        if (workdir / f"{path.name}-{key}").is_file()
    }

    if "dt" in parts and dt_type not in QCDT_TYPES:
        # Lossy by design: the RPM program header elftool needs to pack the
        # device tree back does not exist for a bare dtb, so the dtb is folded
        # into the kernel and the separate part is destroyed. Unpacking the
        # result will not yield it back.
        kernel = parts.get("kernel", workdir / "kernel")
        if kernel.is_file():
            kernel.write_bytes(
                append_dtb_to_kernel(kernel.read_bytes(), parts["dt"].read_bytes())
            )
            parts["kernel"] = kernel
        for leftover in workdir.glob(f"{path.name}-dt*"):
            leftover.unlink()
        parts.pop("dt", None)
    elif "dt" in parts:
        # A QCDT/ELF dt is the RPM payload: repackimg.sh feeds *-dt back to
        # elftool as the ",rpm" section, so name the part accordingly.
        parts["rpm"] = parts.pop("dt")

    return SonyElfState(
        directory=workdir, parts=parts, fields=fields, image_name=path.name
    )


@dataclass(frozen=True, slots=True)
class SonyElfState(Unpacked):
    """Unpacked Sony ELF state, tagged with the image name.

    ``unpackelf`` names its outputs after the image it was given, so the
    work directory has to remember that name to find the parts again.
    """

    image_name: str = ""


def pack_arg(path: Path, section: str | None = None, address: str | None = None) -> str:
    """Build one ``elftool pack`` component argument.

    The grammar is ``<path>``, optionally ``<path>@<addr>`` to pin a load
    address, optionally with a trailing ``,<section>`` to give the part a
    type. ``@cmdline`` after the path is the one exception: it is a literal
    keyword, not a load address, and is passed through as the suffix.
    """
    arg = f"{path}@{address}" if address else str(path)
    return f"{arg},{section}" if section else arg


def should_fallback_to_aosp(
    *,
    has_header: bool,
    dt_type: str | None,
    force_elf: bool = False,
) -> bool:
    """Decide whether an ELF image must be repacked as AOSP instead.

    The original requires *both* a recovered header and a genuine RPM for the
    ELF path; anything else warns and silently rebuilds an AOSP image, which
    is why an unpacked-then-repacked image can come back a different format.
    """
    if not has_header:
        return True
    if force_elf:
        return False
    return dt_type != _ELF_DT_TYPE


def pack(state: Unpacked, out: Path) -> Path:
    """Rebuild a Sony ELF image.

    Only the parts the original passes are forwarded: the header, the kernel,
    the ramdisk, the rpm and the cmdline.
    """
    args: list[str] = ["pack", "-o", str(out.resolve())]
    if "header" in state.parts:
        args.append(f"header={state.part('header')}")
    if "kernel" in state.parts:
        args.append(pack_arg(state.part("kernel")))
    if "ramdisk" in state.parts:
        args.append(pack_arg(state.part("ramdisk"), section="ramdisk"))
    if "rpm" in state.parts:
        args.append(pack_arg(state.part("rpm"), section="rpm"))
    if "cmdline" in state.parts:
        args.append(f"{state.part('cmdline')}@cmdline")

    out.parent.mkdir(parents=True, exist_ok=True)
    run_tool("elftool", *args)
    return out
