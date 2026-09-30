"""Intel OSIP images (``mboot``).

``mboot`` reads and writes a flat set of files in its working directory, so
both directions are pure rename games around the tool's fixed filenames.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ._tools import Unpacked, run_tool

__all__ = [
    "MBOOT_HEADER_BYTES",
    "MBOOT_PARTS",
    "RAMDISK_MBOOT_NAME",
    "REPACK_PARTS",
    "REQUIRED_PARTS",
    "STAGING_DIR",
    "mboot_name",
    "pack",
    "unpack",
]

#: ``mboot``'s own filenames, keyed by our part name. ``cmdline.txt`` loses the
#: ``.txt`` and ``hdr`` becomes ``header``; the rest keep their spelling.
MBOOT_PARTS: dict[str, str] = {
    "bootstub": "bootstub",
    "cmdline": "cmdline.txt",
    "header": "hdr",
    "kernel": "kernel",
    "parameter": "parameter",
    "ramdisk": "ramdisk.cpio.gz",
    "sig": "sig",
}

#: The parts the original repack loop stages, with the rename applied
#: *backwards*. ``sig`` is included because the original copies it even though
#: mboot has no reader for it -- it is written on unpack and silently dropped on
#: pack. Carrying it keeps us bit-compatible with the Bash.
REPACK_PARTS: tuple[str, ...] = (
    "bootstub",
    "cmdline",
    "header",
    "kernel",
    "parameter",
    "sig",
)

#: mboot hard-fails on a missing input, so all six of these have to be staged
#: for a pack to succeed -- verified against the bundled binary, which aborts
#: with "cannot open input file" rather than defaulting them.
REQUIRED_PARTS: tuple[str, ...] = (
    "header",
    "kernel",
    "cmdline",
    "ramdisk",
    "parameter",
    "bootstub",
)

#: mboot's header handling is only reliable for the images it was built from.
#: It synthesises the ``hdr`` slot on pack instead of preserving it, so a
#: synthesised image desynchronises: a 1-byte hdr overflows into the cmdline
#: ("H" + "conso" -> "Hconso"), any longer one shifts the component offsets and
#: the unpack aborts with "kernel size likely wrong". We pass the header
#: through unchanged and let the tool do what it does; reproducing this quirk
#: in a test would only pin a tool bug. See tests/test_exotic_formats.py.
MBOOT_HEADER_BYTES = 1

STAGING_DIR = ".temp"

#: mboot only ever looks for this name. The ramdisk is force-renamed to it
#: whatever it actually is, because the pack path assumes gzip and never
#: sniffs the magic.
RAMDISK_MBOOT_NAME = "ramdisk.cpio.gz"


def mboot_name(key: str) -> str:
    return MBOOT_PARTS[key]


def unpack(path: Path, workdir: Path) -> Unpacked:
    """Split an OSIP image.

    ``mboot`` drops its parts into the process CWD, so it runs with ``cwd``
    set to the work directory and the parts are renamed in place afterwards.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    run_tool("mboot", "-u", "-f", path.resolve(), cwd=workdir)

    parts: dict[str, Path] = {}
    for key, src_name in MBOOT_PARTS.items():
        src = workdir / src_name
        if not src.is_file():
            continue
        dst = workdir / key
        if src != dst:
            src.replace(dst)
        parts[key] = dst

    return Unpacked(directory=workdir, parts=parts, fields={})


def pack(state: Unpacked, out: Path) -> Path:
    """Rebuild an OSIP image from a staging directory of mboot-named files."""
    staging = state.directory / STAGING_DIR
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for key in REPACK_PARTS:
        src = state.parts.get(key)
        if src is not None and src.is_file():
            shutil.copyfile(src, staging / mboot_name(key))

    ramdisk = state.parts.get("ramdisk")
    if ramdisk is not None and ramdisk.is_file():
        shutil.copyfile(ramdisk, staging / RAMDISK_MBOOT_NAME)

    # mboot aborts on the first missing input with a bare filename, which reads
    # like a tool bug. Check up front so the caller learns which part is absent.
    missing = sorted(
        key for key in REQUIRED_PARTS if not (staging / mboot_name(key)).is_file()
    )
    if missing:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(
            f"mboot needs these parts to build an OSIP image, and they are "
            f"missing from the unpacked state: {missing}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_tool("mboot", "-d", staging, "-f", out.resolve(), cwd=staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out
