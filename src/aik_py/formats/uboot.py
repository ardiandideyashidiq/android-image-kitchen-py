"""DENX U-Boot images (``mkimage`` / ``dumpimage``).

A U-Boot legacy image is a 64-byte ``image_header_t`` followed by payload
data. Several OEM kernels ship a header plus one or more components; ``Multi``
is the two-component case, ``RAMDisk`` a single one.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, replace
from pathlib import Path

from ._tools import Unpacked, run_tool

__all__ = [
    "UbootInfo",
    "be_u32",
    "header_overhead",
    "image_size_offset",
    "pack",
    "parse_listing",
    "trim_size",
    "unpack",
]

#: U-Boot stores ``image_size`` as a big-endian u32 at this offset inside the
#: 64-byte header. Reading it little-endian yields garbage, which is why the
#: original reaches for ``hexdump`` rather than a plain unpack.
image_size_offset = 12

#: header (64 bytes) plus the four size fields the header layout reserves.
header_overhead = 64

_HEADER_MAGIC = 0x27051956

#: U-Boot image types, taken verbatim from the bundled ``mkimage -T list``.
#: The parser anchors on this vocabulary so a listing is split on the type
#: rather than on a whitespace count. The tool lowercases these, but a listing
#: title-cases them ("Multi-File Image", "RAMDisk Image", "Kernel Image"), so
#: the pattern is matched case-insensitively.
_TYPE_NAMES = frozenset(
    {
        "aisimage",
        "atmelimage",
        "copro",
        "filesystem",
        "firmware",
        "firmware_ivt",
        "flat_dt",
        "fpga",
        "gpimage",
        "imx8image",
        "imx8mimage",
        "imximage",
        "invalid",
        "kernel",
        "kernel_noload",
        "kwbimage",
        "lpc32xximage",
        "mtk_image",
        "multi",
        "mxsimage",
        "omapimage",
        "pblimage",
        "pmmc",
        "ramdisk",
        "rkimage",
        "rksd",
        "rkspi",
        "script",
        "socfpgaimage",
        "socfpgaimage_v1",
        "standalone",
        "stm32image",
        "sunxi_egon",
        "tee",
        "ublimage",
        "vybridimage",
        "x86_setup",
        "zynqimage",
        "zynqmpbif",
        "zynqmpimage",
    }
)

_NAME_RE = re.compile(r"^Image Name:\s*(?P<name>.*?)\s*$", re.MULTILINE)
# "Image Type:   ARM Linux Multi-File Image (uncompressed)" and the two-word
# "Image Type:   Intel x86 Linux Kernel Image (bzip2 compressed)".
#
# Counting whitespace-separated fields from the left, as the Bash's
# `cut -d' ' -f1/f2/f3` does, breaks on the two-word arch: it reads
# "Intel x86 Linux" as arch="Intel", os="x86", type="Linux", and a repack
# then feeds those back to mkimage and mis-splices the header. Anchor the type
# on U-Boot's own vocabulary instead -- `mkimage -T list` is the source of
# truth -- and take everything before it as the "ARCH [ARCH] OS" run.
_TYPE_ALTERNATIVES = "|".join(sorted(_TYPE_NAMES, key=len, reverse=True))
_TYPE_RE = re.compile(
    rf"^Image Type:\s+(?P<archos>.*?)\s+(?P<type>{_TYPE_ALTERNATIVES})"
    r"(?:-[A-Za-z0-9_]+)?(?:\s+Image)?\s+\((?P<comp>[^)]*)\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_ADDR_RE = re.compile(r"^Load Address:\s*(?P<addr>\S+)\s*$", re.MULTILINE)
_EP_RE = re.compile(r"^Entry Point:\s*(?P<ep>\S+)\s*$", re.MULTILINE)

#: mkimage accepts exactly these ``-C`` values, and its ``-C list`` output
#: pairs each with the phrase the listing prints in parentheses. Passing the
#: full phrase back ("bzip2 compressed") is rejected as an invalid compression.
COMPRESSION_ALIASES = {"uncompressed": "none"}
COMPRESSIONS = frozenset({"bzip2", "gzip", "lz4", "lzma", "lzo", "none", "zstd"})

#: ``mkimage -A list`` / ``-O list`` print "<token>  <description>", and the
#: listing reuses that description verbatim: an image built with ``-A x86``
#: lists as "Intel x86 Linux Kernel Image". Handing the description straight
#: back exits 1 with "Invalid architecture", so the description is mapped to
#: its token. Captured from the bundled mkimage.
_ARCH_TOKENS = {
    "Alpha": "alpha",
    "ARC": "arc",
    "ARM": "arm",
    "AArch64": "arm64",
    "AVR32": "avr32",
    "Blackfin": "blackfin",
    "IA64": "ia64",
    "Invalid ARCH": "invalid",
    "M68K": "m68k",
    "MicroBlaze": "microblaze",
    "MIPS": "mips",
    "MIPS 64 Bit": "mips64",
    "NDS32": "nds32",
    "NIOS II": "nios2",
    "OpenRISC 1000": "or1k",
    "PowerPC": "powerpc",
    "RISC-V": "riscv",
    "IBM S390": "s390",
    "Sandbox": "sandbox",
    "SuperH": "sh",
    "SPARC": "sparc",
    "SPARC 64 Bit": "sparc64",
    "Intel x86": "x86",
    "AMD x86_64": "x86_64",
    "Xtensa": "xtensa",
}
_OS_TOKENS = {
    "4_4BSD": "4_4bsd",
    "ARM Trusted Firmware": "arm-trusted-firmware",
    "Dell": "dell",
    "EFI Firmware": "efi",
    "Esix": "esix",
    "FreeBSD": "freebsd",
    "INTEGRITY": "integrity",
    "Invalid OS": "invalid",
    "Irix": "irix",
    "Linux": "linux",
    "LynxOS": "lynxos",
    "NCR": "ncr",
    "NetBSD": "netbsd",
    "OpenBSD": "openbsd",
    "OpenRTOS": "openrtos",
    "RISC-V OpenSBI": "opensbi",
    "Enea OSE": "ose",
    "Plan 9": "plan9",
    "pSOS": "psos",
    "QNX": "qnx",
    "RTEMS": "rtems",
    "SCO": "sco",
    "Solaris": "solaris",
    "SVR4": "svr4",
    "Trusted Execution Environment": "tee",
    "U-Boot": "u-boot",
    "VxWorks": "vxworks",
}


@dataclass(frozen=True, slots=True)
class UbootInfo:
    """Header fields recovered from a ``dumpimage -l`` listing."""

    name: str
    arch: str
    os: str
    type: str
    comp: str
    addr: str
    ep: str

    def as_fields(self) -> dict[str, str]:
        return {
            "name": self.name,
            "arch": self.arch,
            "os": self.os,
            "type": self.type,
            "comp": self.comp,
            "addr": self.addr,
            "ep": self.ep,
        }


def be_u32(buf: bytes, offset: int) -> int:
    return struct.unpack_from(">I", buf, offset)[0]


def trim_size(data: bytes) -> int:
    """Return the payload-bearing length of a U-Boot image.

    ``image_size`` counts only the payload, so the file on disk is
    ``image_size + 64``. OEM images often carry trailing padding or appended
    data that has to be cut before ``dumpimage`` will read the components.
    """
    return be_u32(data, image_size_offset) + header_overhead


def parse_listing(text: str) -> UbootInfo:
    """Parse the human-readable listing produced by ``dumpimage -l``.

    Returns the tokens ``mkimage -A/-O/-T/-C`` expect, not the phrases the
    listing prints: the original's ``grep``/``cut`` pipeline yields those
    tokens (``uncompressed`` -> ``none``, "ARM" -> ``arm``, "Intel x86" ->
    ``x86``), and feeding a phrase back makes mkimage exit 1.
    """
    name_match = _NAME_RE.search(text)
    type_match = _TYPE_RE.search(text)
    addr_match = _ADDR_RE.search(text)
    ep_match = _EP_RE.search(text)
    if not (name_match and type_match and addr_match and ep_match):
        raise ValueError(f"could not parse dumpimage listing:\n{text}")

    try:
        arch, os_name = split_arch_os(type_match.group("archos"))
    except ValueError as exc:
        raise ValueError(
            f"could not parse arch/os from dumpimage listing:\n{text}"
        ) from exc

    comp = normalize_comp(type_match.group("comp") or "")
    return UbootInfo(
        name=name_match.group("name"),
        arch=arch,
        os=os_name,
        type=type_match.group("type").strip(),
        comp=comp,
        addr=addr_match.group("addr"),
        ep=ep_match.group("ep"),
    )


def split_arch_os(archos: str) -> tuple[str, str]:
    """Split the "ARCH OS" run into mkimage's ``-A`` and ``-O`` tokens.

    Listings spell both out ("ARM Linux", "Intel x86 Linux", "ARM EFI
    Firmware"), and the run has no fixed word count, so the boundary cannot be
    found by counting fields. Try the whole run against the tool's arch table
    first, then every arch/os split, then fall back to "last word is the OS",
    which is right for the common two-word cases.
    """
    run = archos.strip()
    if run in _ARCH_TOKENS:
        return _ARCH_TOKENS[run], "linux"

    words = run.split()
    for i in range(1, len(words)):
        arch_desc, os_desc = " ".join(words[:i]), " ".join(words[i:])
        if arch_desc in _ARCH_TOKENS and os_desc in _OS_TOKENS:
            return _ARCH_TOKENS[arch_desc], _OS_TOKENS[os_desc]

    if len(words) >= 2:
        return _ARCH_TOKENS.get(
            " ".join(words[:-1]), " ".join(words[:-1])
        ), _OS_TOKENS.get(words[-1], words[-1])
    raise ValueError(f"could not split arch/os from {archos!r}")


def normalize_comp(raw: str) -> str:
    """Reduce the parenthesised phrase to a value ``mkimage -C`` accepts.

    A listing spells the compression out in prose -- ``(bzip2 compressed)``,
    ``(gzip compressed)``, ``(uncompressed)``. The original cuts that at the
    first space and dash (``cut -d' ' -f1 | cut -d- -f1``) to get the bare
    token, then maps ``uncompressed`` to ``none``. Handing the whole phrase
    back makes mkimage exit 1 with "Invalid compression type".
    """
    token = raw.strip().split(" ", 1)[0].split("-", 1)[0].strip()
    return COMPRESSION_ALIASES.get(token, token)


def is_uboot(data: bytes) -> bool:
    return len(data) >= header_overhead and be_u32(data, 0) == _HEADER_MAGIC


def unpack(path: Path, workdir: Path) -> Unpacked:
    """Split a U-Boot image into its components and header fields."""
    workdir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, Path] = {}
    fields: dict[str, str] = {}

    raw = path.read_bytes()
    source = path
    if is_uboot(raw) and (want := trim_size(raw)) != len(raw):
        trimmed = workdir / "trimmed.img"
        trimmed.write_bytes(raw[:want])
        source = trimmed

    listing = run_tool("dumpimage", "-l", source).stdout
    info = parse_listing(listing)
    fields.update(info.as_fields())

    kernel = workdir / "kernel"
    run_tool("dumpimage", "-p", "0", "-o", kernel, source)
    parts["kernel"] = kernel

    if info.type == "Multi":
        ramdisk = workdir / "ramdisk"
        run_tool("dumpimage", "-p", "1", "-o", ramdisk, source)
        parts["ramdisk"] = ramdisk
    elif info.type == "RAMDisk":
        # The single component *is* the ramdisk; the Bash renames rather than
        # extracting again, so the payload is never split a second time.
        ramdisk = workdir / "ramdisk"
        kernel.rename(ramdisk)
        del parts["kernel"]
        parts["ramdisk"] = ramdisk
    else:
        parts["ramdisk"] = workdir / "ramdisk"
        parts["ramdisk"].touch()

    return Unpacked(directory=workdir, parts=parts, fields=fields)


def pack(state: Unpacked, out: Path) -> Path:
    """Rebuild a U-Boot image from an unpacked state.

    ``mkimage`` accepts several components as one ``-d`` argument joined by
    ``:``, so ``Multi`` images concatenate kernel and ramdisk on that axis.
    """
    info = UbootInfo(
        **{
            k: state.field(k)
            for k in ("name", "arch", "os", "type", "comp", "addr", "ep")
        }
    )
    kernel = state.parts.get("kernel")
    ramdisk = state.parts.get("ramdisk")

    if info.type == "Multi":
        if kernel is None or ramdisk is None:
            raise ValueError("Multi images need both a kernel and a ramdisk")
        data_arg = f"{kernel}:{ramdisk}"
    elif info.type == "RAMDisk":
        if ramdisk is None:
            raise ValueError("RAMDisk images need a ramdisk")
        data_arg = str(ramdisk)
    elif kernel is not None:
        data_arg = str(kernel)
    else:
        raise ValueError(f"unsupported U-Boot image type: {info.type!r}")

    out.parent.mkdir(parents=True, exist_ok=True)
    run_tool(
        "mkimage",
        "-A",
        info.arch,
        "-O",
        info.os,
        "-T",
        info.type,
        "-C",
        info.comp,
        "-a",
        info.addr,
        "-e",
        info.ep,
        "-n",
        info.name,
        "-d",
        data_arg,
        out,
    )
    return out


def with_updates(state: Unpacked, **updates: str) -> Unpacked:
    return replace(state, fields={**state.fields, **updates})
