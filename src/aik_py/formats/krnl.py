"""Rockchip KRNL images.

A KRNL file is an 8-byte prefix followed by the ramdisk, CRC-protected by
``rkcrc``. There is no kernel and no header to recover.
"""

from __future__ import annotations

from pathlib import Path

from ._tools import Unpacked, run_tool

__all__ = ["PREFIX_LEN", "is_krnl", "pack", "split_ramdisk", "unpack"]

#: The original runs ``dd bs=4096 skip=8 iflag=skip_bytes``. ``skip_bytes``
#: switches ``skip`` from blocks to bytes, so the ramdisk starts at byte 8 --
#: not at byte 8*4096, which is the reading the flag name invites.
PREFIX_LEN = 8

#: rkcrc writes this tag at offset 0, and bin/androidbootimg.magic matches it
#: there to identify a "KRNL bootimg". Checked by ``is_krnl`` so a format
#: sniffer cannot route an ordinary AOSP/ELF/OSIP image into this handler.
MAGIC = b"KRNL"


def is_krnl(data: bytes) -> bool:
    """True when the file carries the Rockchip KRNL tag.

    Both conditions matter: the tag has to be there *and* there has to be a
    payload behind the 8-byte prefix. Without the tag check this would claim
    every image AIK handles, and repacking one would silently rewrite it into a
    KRNL container.
    """
    return len(data) > PREFIX_LEN and data.startswith(MAGIC)


def split_ramdisk(data: bytes) -> bytes:
    return data[PREFIX_LEN:]


def unpack(path: Path, workdir: Path) -> Unpacked:
    workdir.mkdir(parents=True, exist_ok=True)
    ramdisk = workdir / "ramdisk"
    ramdisk.write_bytes(split_ramdisk(path.read_bytes()))
    return Unpacked(directory=workdir, parts={"ramdisk": ramdisk}, fields={})


def pack(state: Unpacked, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run_tool("rkcrc", "-k", state.part("ramdisk"), out)
    return out
