"""Tests for the exotic OEM format wrappers.

Everything that can be checked without a genuine device image is exercised
here: the U-Boot header maths and listing parser, the KRNL split, the OSIP
rename maps, the QCDT append and the ELF->AOSP fallback rule. The cases that
need a real bundled binary are skipped rather than faked.
"""

from __future__ import annotations

import gzip
import random
import struct
import subprocess
from pathlib import Path

import pytest

from aik_py.formats import elf, krnl, osip, uboot
from aik_py.formats._tools import ToolError, Unpacked, bin_dir, run_tool

ELF_MAGIC = elf.ELF_MAGIC

REAL_LISTING = """\
Image Name:   TestUboot
Created:      Wed Sep 30 10:05:04 2026
Image Type:   ARM Linux Multi-File Image (uncompressed)
Data Size:    6156 Bytes = 6.01 KiB = 0.01 MiB
Load Address: 10008000
Entry Point:  10008000
Contents:
   Image 0: 4096 Bytes = 4.00 KiB = 0.00 MiB
   Image 1: 2048 Bytes = 2.00 KiB = 0.00 MiB
"""


def _have(tool: str) -> bool:
    return (bin_dir() / tool).is_file()


needs_dumpimage = pytest.mark.skipif(
    not _have("dumpimage"), reason="bundled dumpimage unavailable"
)
needs_mkimage = pytest.mark.skipif(
    not _have("mkimage"), reason="bundled mkimage unavailable"
)
needs_mboot = pytest.mark.skipif(not _have("mboot"), reason="bundled mboot unavailable")
needs_rkcrc = pytest.mark.skipif(not _have("rkcrc"), reason="bundled rkcrc unavailable")
needs_elftool = pytest.mark.skipif(
    not (_have("elftool") and _have("unpackelf")),
    reason="bundled elftool/unpackelf unavailable",
)


def make_uboot_image(payload: bytes, *, pad: int = 0) -> bytes:
    """Build a 64-byte U-Boot header around ``payload``."""
    header = bytearray(64)
    struct.pack_into(">I", header, 0, 0x27051956)
    struct.pack_into(">I", header, 4, len(payload))  # data size, unused by us
    struct.pack_into(">I", header, 12, len(payload))  # image_size, big-endian
    return bytes(header) + payload + b"\x00" * pad


# --------------------------------------------------------------------------
# U-Boot: big-endian size trim
# --------------------------------------------------------------------------


def test_be_u32_reads_big_endian():
    assert uboot.be_u32(b"\x00\x00\x18\x0c", 0) == 0x180C
    assert uboot.be_u32(b"\xff\xff\xff\xff\x00\x00\x18\x0c", 4) == 0x180C


def test_be_u32_is_not_little_endian():
    buf = b"\x00\x00\x18\x0c"
    assert uboot.be_u32(buf, 0) != struct.unpack("<I", buf)[0]


def test_trim_size_is_image_size_plus_header_overhead():
    payload = b"\xaa" * 3000
    data = make_uboot_image(payload)
    assert uboot.trim_size(data) == 3000 + uboot.header_overhead == 3064


def test_trim_size_matches_exact_file_size_so_no_truncation():
    data = make_uboot_image(b"\xbb" * 4096)
    assert uboot.trim_size(data) == len(data)


def test_trim_size_detects_trailing_padding():
    padded = make_uboot_image(b"\xcc" * 1000, pad=512)
    assert uboot.trim_size(padded) < len(padded)
    truncated = padded[: uboot.trim_size(padded)]
    assert len(truncated) == 1000 + 64
    assert uboot.trim_size(truncated) == len(truncated)


def test_trim_size_uses_offset_12_not_some_other_field():
    data = bytearray(make_uboot_image(b"\xdd" * 100))
    # The u32 at offset 4 is a different field; the trim must ignore it.
    struct.pack_into(">I", data, 4, 0xDEADBEEF)
    assert uboot.trim_size(bytes(data)) == 100 + 64


# --------------------------------------------------------------------------
# U-Boot: listing parse
# --------------------------------------------------------------------------


def test_parse_listing_extracts_every_field():
    """Fields come back as the tokens mkimage accepts, not the listing's prose.

    The listing prints "ARM Linux"; mkimage -A/-O want "arm"/"linux", so the
    parser maps through the tool's own -A/-O tables.
    """
    info = uboot.parse_listing(REAL_LISTING)
    assert info.name == "TestUboot"
    assert info.arch == "arm"
    assert info.os == "linux"
    assert info.type == "Multi"
    assert info.comp == "none"
    assert info.addr == "10008000"
    assert info.ep == "10008000"


def test_parse_listing_translates_uncompressed_to_none():
    assert uboot.parse_listing(REAL_LISTING).comp == "none"
    assert "uncompressed" in REAL_LISTING


def test_parse_listing_keeps_real_compression_names():
    for comp in ("gzip", "bzip2", "lzma", "xz", "lzo"):
        listing = REAL_LISTING.replace("(uncompressed)", f"({comp})")
        assert uboot.parse_listing(listing).comp == comp


# Captured from the bundled mkimage/dumpimage pair: the listing spells the
# compression out in prose, and mkimage -C accepts only the bare token.
COMPRESSED_LISTING = """\
Image Name:   P
Created:      Wed Sep 30 10:35:45 2026
Image Type:   Intel x86 Linux Kernel Image (bzip2 compressed)
Data Size:    8192 Bytes = 8.00 KiB = 0.02 MiB
Load Address: 00008000
Entry Point:  00008000
"""


def test_parse_listing_cuts_the_prose_compression_down_to_a_token():
    """A listing says "(bzip2 compressed)"; mkimage -C only accepts "bzip2"."""
    assert uboot.parse_listing(COMPRESSED_LISTING).comp == "bzip2"
    assert uboot.parse_listing(COMPRESSED_LISTING).comp in uboot.COMPRESSIONS


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("uncompressed", "none"),
        ("gzip compressed", "gzip"),
        ("bzip2 compressed", "bzip2"),
        ("lzma compressed", "lzma"),
        ("lzo compressed", "lzo"),
        ("lz4 compressed", "lz4"),
        ("zstd compressed", "zstd"),
        ("none", "none"),
        ("gzip", "gzip"),
    ],
)
def test_normalize_comp(raw, expected):
    assert uboot.normalize_comp(raw) == expected


def test_normalize_comp_always_yields_an_acceptable_token():
    for phrase in ("uncompressed", "gzip compressed", "bzip2 compressed"):
        assert uboot.normalize_comp(phrase) in uboot.COMPRESSIONS


@needs_mkimage
@needs_dumpimage
def test_compressed_image_roundtrips_against_the_bundled_tools(tmp_path):
    """The regression this guards: a compressed image used to be unrepackable."""
    payload = tmp_path / "k.bin"
    payload.write_bytes(b"\x5a" * 8192)
    img = tmp_path / "k.img"
    run_tool(
        "mkimage",
        "-A",
        "x86",
        "-O",
        "Linux",
        "-T",
        "Kernel",
        "-C",
        "bzip2",
        "-a",
        "0x8000",
        "-e",
        "0x8000",
        "-n",
        "P",
        "-d",
        payload,
        img,
    )
    info = uboot.parse_listing(run_tool("dumpimage", "-l", img).stdout)
    assert info.comp == "bzip2"
    assert info.type == "Kernel"

    state = uboot.unpack(img, tmp_path / "work")
    rebuilt = uboot.pack(state, tmp_path / "rebuilt.img")
    again = uboot.parse_listing(run_tool("dumpimage", "-l", rebuilt).stdout)
    assert again == info, (again, info)


def test_parse_listing_handles_single_file_type():
    listing = REAL_LISTING.replace("Multi-File Image", "RAMDisk Image")
    info = uboot.parse_listing(listing)
    assert info.type == "RAMDisk"
    assert info.arch == "arm"


def test_parse_listing_handles_bare_multi_type():
    listing = REAL_LISTING.replace("Multi-File Image", "Multi")
    assert uboot.parse_listing(listing).type == "Multi"


def test_parse_listing_maps_a_two_word_arch_to_its_token():
    """ "Intel x86 Linux" is one arch plus the OS, not arch="Intel" os="x86".

    Counting whitespace fields from the left, as the Bash does, gets this
    wrong; mkimage -A then rejects the image with "Invalid architecture".
    """
    listing = REAL_LISTING.replace("ARM Linux", "Intel x86 Linux")
    info = uboot.parse_listing(listing)
    assert info.arch == "x86"
    assert info.os == "linux"


def test_parse_listing_maps_every_known_arch_and_os_token():
    for desc, token in uboot._ARCH_TOKENS.items():
        listing = REAL_LISTING.replace("ARM Linux", f"{desc} Linux")
        info = uboot.parse_listing(listing)
        assert info.arch == token, desc
        assert info.os == "linux", desc
    for desc, token in uboot._OS_TOKENS.items():
        # Substitute the whole "ARM Linux" arch/os run so a multi-word OS
        # description ("EFI Firmware", "Plan 9") cannot be mistaken for the
        # type token.
        listing = REAL_LISTING.replace("ARM Linux", f"ARM {desc}")
        assert uboot.parse_listing(listing).os == token, desc


def test_parse_listing_preserves_name_with_spaces():
    listing = REAL_LISTING.replace("TestUboot", "My Test Image")
    assert uboot.parse_listing(listing).name == "My Test Image"


def test_parse_listing_rejects_garbage():
    with pytest.raises(ValueError, match="could not parse"):
        uboot.parse_listing("not a dumpimage listing at all\n")


def test_as_fields_round_trips_into_state_fields():
    fields = uboot.parse_listing(REAL_LISTING).as_fields()
    assert set(fields) == {"name", "arch", "os", "type", "comp", "addr", "ep"}


# --------------------------------------------------------------------------
# U-Boot: pack data argument
# --------------------------------------------------------------------------


def test_pack_multi_joins_kernel_and_ramdisk_with_colon(tmp_path):
    kernel, ramdisk = tmp_path / "k", tmp_path / "r"
    kernel.write_bytes(b"k")
    ramdisk.write_bytes(b"r")
    fields = uboot.parse_listing(REAL_LISTING).as_fields()
    state = Unpacked(
        directory=tmp_path, parts={"kernel": kernel, "ramdisk": ramdisk}, fields=fields
    )
    out = tmp_path / "out.img"

    recorded: list[list[str]] = []
    original = uboot.run_tool
    uboot.run_tool = lambda name, *a, **k: (
        recorded.append([name, *(str(x) for x in a)])
        or subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    )
    try:
        uboot.pack(state, out)
    finally:
        uboot.run_tool = original

    assert recorded, "mkimage was never invoked"
    argv = recorded[0]
    assert argv[0] == "mkimage"
    assert f"{kernel}:{ramdisk}" in argv
    assert argv[argv.index("-T") + 1] == "Multi"
    assert argv[argv.index("-C") + 1] == "none"
    assert argv[argv.index("-A") + 1] == "arm"
    assert argv[argv.index("-O") + 1] == "linux"
    assert argv[argv.index("-a") + 1] == "10008000"
    assert argv[argv.index("-e") + 1] == "10008000"
    assert argv[argv.index("-n") + 1] == "TestUboot"


def test_pack_ramdisk_uses_ramdisk_alone(tmp_path):
    listing = REAL_LISTING.replace("Multi-File Image", "RAMDisk Image")
    fields = uboot.parse_listing(listing).as_fields()
    ramdisk = tmp_path / "r"
    ramdisk.write_bytes(b"r")
    state = Unpacked(directory=tmp_path, parts={"ramdisk": ramdisk}, fields=fields)

    recorded: list[list[str]] = []
    original = uboot.run_tool
    uboot.run_tool = lambda name, *a, **k: (
        recorded.append([name, *(str(x) for x in a)])
        or subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    )
    try:
        uboot.pack(state, tmp_path / "out.img")
    finally:
        uboot.run_tool = original

    assert str(ramdisk) in recorded[0]
    assert not any(":" in a for a in recorded[0])


# --------------------------------------------------------------------------
# KRNL
# --------------------------------------------------------------------------


def test_krnl_ramdisk_is_everything_after_byte_8():
    data = bytes(range(64))
    assert krnl.split_ramdisk(data) == data[8:]
    assert krnl.split_ramdisk(data) == bytes(range(8, 64))


def test_krnl_split_is_not_block_aligned():
    # bs=4096 with iflag=skip_bytes means 8 BYTES, not 8 blocks.
    data = b"\x00" * 8 + b"payload"
    assert krnl.split_ramdisk(data) == b"payload"
    assert krnl.split_ramdisk(data) != data[8 * 4096 :]
    assert krnl.split_ramdisk(data) == b"payload", "prefix is 8 bytes, not 8*4096"


def test_krnl_split_of_empty_tail():
    assert krnl.split_ramdisk(b"\x00" * 8) == b""


def test_krnl_unpack_writes_ramdisk_part(tmp_path):
    img = tmp_path / "boot.img"
    img.write_bytes(b"KRNL\x00\x00\x00\x00" + b"ramdisk-bytes")
    state = krnl.unpack(img, tmp_path / "work")
    assert state.part_bytes("ramdisk") == b"ramdisk-bytes"
    assert not state.parts.get("kernel")


def test_is_krnl_requires_the_rockchip_tag():
    """Without the tag check this would claim every image AIK handles."""
    assert krnl.is_krnl(b"KRNL\x00\x00\x00\x00payload") is True
    assert krnl.is_krnl(b"ANDROID!" + b"\x00" * 100) is False
    assert krnl.is_krnl(b"\x7fELF" + b"\x00" * 100) is False
    assert krnl.is_krnl(b"KRNL") is False, "no payload behind the prefix"
    assert krnl.is_krnl(b"") is False


def test_krnl_tag_is_what_rkcrc_writes(tmp_path):
    """Pin the magic against the bundled rkcrc output."""
    assert krnl.MAGIC == b"KRNL"


# --------------------------------------------------------------------------
# OSIP rename maps
# --------------------------------------------------------------------------


def test_osip_unpack_rename_map():
    assert osip.MBOOT_PARTS == {
        "bootstub": "bootstub",
        "cmdline": "cmdline.txt",
        "header": "hdr",
        "kernel": "kernel",
        "parameter": "parameter",
        "ramdisk": "ramdisk.cpio.gz",
        "sig": "sig",
    }


def test_osip_unpack_strips_txt_and_renames_hdr():
    assert osip.mboot_name("cmdline") == "cmdline.txt"
    assert osip.mboot_name("header") == "hdr"
    assert osip.mboot_name("ramdisk") == "ramdisk.cpio.gz"


def test_osip_repack_map_is_the_reverse_of_unpack():
    for key, mboot_name in osip.MBOOT_PARTS.items():
        assert osip.mboot_name(key) == mboot_name
    assert set(osip.REPACK_PARTS) <= set(osip.MBOOT_PARTS)


def test_osip_repack_stages_every_part_the_bash_does():
    # The Bash repack loop copies `sig` into the staging dir even though mboot
    # has no reader for it, so we stage it too rather than diverging. The
    # ramdisk is deliberately absent: it is staged separately under a forced
    # name, so it never goes through the plain rename map.
    assert "sig" in osip.MBOOT_PARTS
    assert set(osip.REPACK_PARTS) == set(osip.MBOOT_PARTS) - {"ramdisk"}
    assert "sig" in {osip.mboot_name(k) for k in osip.REPACK_PARTS}
    assert set(osip.REPACK_PARTS) | {"ramdisk"} == set(osip.MBOOT_PARTS)


def test_osip_required_parts_match_what_mboot_demands():
    # Verified against the bundled binary: it aborts with "cannot open input
    # file" for each of hdr/kernel/cmdline.txt/parameter/bootstub/ramdisk.
    assert set(osip.REQUIRED_PARTS) == {
        "header",
        "kernel",
        "cmdline",
        "ramdisk",
        "parameter",
        "bootstub",
    }
    assert set(osip.REQUIRED_PARTS) <= set(osip.MBOOT_PARTS)
    assert "sig" not in osip.REQUIRED_PARTS


def test_osip_ramdisk_staged_as_gz_regardless_of_compression(tmp_path):
    ramdisk = tmp_path / "ramdisk.cpio.xz"
    original_bytes = b"\xfd7zXZ\x00not-really-gzip-at-all"
    ramdisk.write_bytes(original_bytes)
    parts = {
        "header": tmp_path / "header",
        "kernel": tmp_path / "kernel",
        "bootstub": tmp_path / "bootstub",
        "cmdline": tmp_path / "cmdline",
        "parameter": tmp_path / "parameter",
        "ramdisk": ramdisk,
    }
    for name, path in parts.items():
        if name != "ramdisk":
            path.write_bytes(b"stub")
    state = Unpacked(directory=tmp_path, parts=parts, fields={})

    captured: dict[str, dict[str, bytes]] = {}
    original = osip.run_tool

    def spy(name, *args, **kwargs):
        staging = Path(kwargs["cwd"])
        captured["files"] = {p.name: p.read_bytes() for p in staging.iterdir()}
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    osip.run_tool = spy
    try:
        osip.pack(state, tmp_path / "out.img")
    finally:
        osip.run_tool = original

    assert "ramdisk.cpio.gz" in captured["files"], (
        "mboot only ever looks for ramdisk.cpio.gz, so the ramdisk is copied "
        "in under that name whatever compression it really uses"
    )
    assert not any(n.startswith("ramdisk.cpio.xz") for n in captured["files"])
    assert captured["files"]["ramdisk.cpio.gz"] == original_bytes, (
        "the bytes are copied verbatim; mboot only relies on the name"
    )
    assert (tmp_path / osip.STAGING_DIR).exists() is False, (
        "staging dir is cleaned up after pack"
    )


def test_osip_staging_uses_mboot_names_and_omits_sig(tmp_path):
    parts = {
        "bootstub": b"B",
        "cmdline": b"console=ttyS0\n",
        "header": b"H",
        "kernel": b"K",
        "parameter": b"P",
        "sig": b"S",
        "ramdisk": gzip.compress(b"R"),
    }
    for name, data in parts.items():
        (tmp_path / name).write_bytes(data)
    state = Unpacked(
        directory=tmp_path, parts={k: tmp_path / k for k in parts}, fields={}
    )

    captured: dict[str, dict[str, bytes]] = {}
    original = osip.run_tool

    def spy(name, *args, **kwargs):
        staging = Path(kwargs["cwd"])
        captured["files"] = {p.name: p.read_bytes() for p in staging.iterdir()}
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    osip.run_tool = spy
    try:
        osip.pack(state, tmp_path / "out.img")
    finally:
        osip.run_tool = original

    assert set(captured["files"]) == {
        "bootstub",
        "cmdline.txt",
        "hdr",
        "kernel",
        "parameter",
        "ramdisk.cpio.gz",
        "sig",
    }
    assert captured["files"]["cmdline.txt"] == b"console=ttyS0\n"
    assert captured["files"]["hdr"] == b"H"
    # mboot never reads sig back; the original still stages it, so we do too
    # rather than silently diverging from the Bash.
    assert captured["files"]["sig"] == b"S"


# --------------------------------------------------------------------------
# QCDT
# --------------------------------------------------------------------------


def test_qcdt_types_are_the_types_that_keep_a_separate_dt():
    assert elf.QCDT_TYPES == frozenset({"QCDT", "ELF"})


def test_append_dtb_gzips_kernel_with_no_embedded_filename():
    kernel = b"kernel-bytes" * 10
    dtb = b"\xd0\x0d\xfe\xed-dtb-bytes"
    out = elf.append_dtb_to_kernel(kernel, dtb)

    assert out.endswith(dtb), "dtb must be appended verbatim"
    compressed = out[: len(out) - len(dtb)]
    assert gzip.decompress(compressed) == kernel


def test_gzip_output_carries_no_filename_and_no_mtime():
    out = elf.gzip_bytes(b"payload" * 100)
    # gzip header: magic(2) cm(1) flg(1) mtime(4) xfl(1) os(2)
    assert out[:2] == b"\x1f\x8b"
    flags = out[3]
    assert not (flags & 0x08), "FNAME flag set: filename embedded in gzip header"
    assert out[4:8] == b"\x00\x00\x00\x00", "mtime should be zeroed"


def test_gzip_matches_cli_gzip_no_name(tmp_path):
    """Our gzip must be byte-identical to the `gzip --no-name -9` the Bash calls.

    Matching exactly (not merely decompressing to the same data) is what lets a
    user diff a repacked image against the original and see no change in the
    kernel; Python's 0xff OS byte would otherwise show up as a one-byte delta.
    """
    import subprocess

    kernel = b"kernel payload " * 200
    ours = elf.gzip_bytes(kernel)

    src = tmp_path / "kernel.bin"
    src.write_bytes(kernel)
    cli = subprocess.run(
        ["gzip", "--no-name", "-9", "-c", str(src)], capture_output=True, check=True
    ).stdout

    assert ours == cli, f"header differs: ours={ours[:10].hex()} gzip={cli[:10].hex()}"
    assert ours[9] == 0x03, "OS byte must be Unix, matching GNU gzip"
    assert gzip.decompress(ours) == kernel


_KERNEL = b"kernel-payload" * 8
_DTB = b"\xd0\x0d\xfe\xed" + b"dtb-payload" * 4


def _fake_unpack(tmp_path, monkeypatch, *, dt_type, with_dt: bool = True):
    """Drive ``elf.unpack`` with a stubbed tool layer.

    The real ``elftool``/``unpackelf`` need a genuine Sony ELF image, so the
    stub fabricates the same on-disk layout the Bash expects to find: a
    ``elftool_out/header`` and ``<image>-{base,pagesize,cmdline,kernel,dt}``.
    """
    workdir = tmp_path / "work"
    workdir.mkdir()
    img = tmp_path / "boot.img"
    img.write_bytes(ELF_MAGIC + b"\x00" * 60)

    def fake_run_tool(name, *args, **kwargs):
        if name == "elftool":
            out_dir = Path(next(str(a) for a in args if str(a).endswith("elftool_out")))
            out_dir.mkdir(parents=True)
            (out_dir / "header").write_bytes(b"\x7fELF header blob")
        elif name == "unpackelf":
            (workdir / "boot.img-base").write_text("0x80000000\n")
            (workdir / "boot.img-pagesize").write_text("2048\n")
            (workdir / "boot.img-cmdline").write_text("console=ttyS0\n")
            (workdir / "boot.img-kernel").write_bytes(_KERNEL)
            if with_dt:
                (workdir / "boot.img-dt").write_bytes(_DTB)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(elf, "run_tool", fake_run_tool)
    return elf.unpack(img, workdir, dt_type=dt_type)


@pytest.mark.parametrize("dt_type", ["QCDT", "ELF"])
def test_qcdt_types_keep_the_dt_as_the_rpm_part(tmp_path, monkeypatch, dt_type):
    """A QCDT/ELF dt is the RPM payload, so it is kept and named 'rpm'.

    repackimg.sh feeds *-dt back to elftool as the ",rpm" section, so a QCDT dt
    must survive unpack as a part that pack will pick up under that name.
    """
    state = _fake_unpack(tmp_path, monkeypatch, dt_type=dt_type)
    assert "rpm" in state.parts
    assert "dt" not in state.parts
    assert state.part_bytes("rpm") == _DTB
    assert not state.part_bytes("kernel").endswith(_DTB)


@pytest.mark.parametrize("dt_type", ["DTB", "QCDT header", "data"])
def test_non_qcdt_dt_types_append_dtb_to_kernel(tmp_path, monkeypatch, dt_type):
    """A bare dtb gets folded into the kernel and the dt part is deleted."""
    state = _fake_unpack(tmp_path, monkeypatch, dt_type=dt_type)
    assert "dt" not in state.parts
    assert "rpm" not in state.parts

    kernel = state.part_bytes("kernel")
    assert kernel.endswith(_DTB), "dtb must be appended to the kernel"
    assert gzip.decompress(kernel[: len(kernel) - len(_DTB)]) == _KERNEL

    for leftover in tmp_path.glob("boot.img-dt*"):
        raise AssertionError(f"dt file should have been removed: {leftover}")


def test_unknown_dt_type_appends_rather_than_silently_keeping_parts(
    tmp_path, monkeypatch
):
    """A caller that omits dt_type gets the lossy path, matching the Bash.

    The original appends for every dt that is not QCDT or ELF and has no
    separate "unknown" case, so the default must not quietly produce a
    different image shape.
    """
    state = _fake_unpack(tmp_path, monkeypatch, dt_type=None)
    assert "dt" not in state.parts
    assert state.part_bytes("kernel").endswith(_DTB)


# --------------------------------------------------------------------------
# ELF -> AOSP fallback
# --------------------------------------------------------------------------


def test_fallback_when_no_header():
    assert elf.should_fallback_to_aosp(has_header=False, dt_type="ELF") is True
    assert (
        elf.should_fallback_to_aosp(has_header=False, dt_type="QCDT", force_elf=True)
        is True
    )


def test_fallback_when_non_elf_dt_without_force():
    assert elf.should_fallback_to_aosp(has_header=True, dt_type="QCDT") is True
    assert elf.should_fallback_to_aosp(has_header=True, dt_type=None) is True
    assert elf.should_fallback_to_aosp(has_header=True, dt_type="DTB") is True


def test_elf_dt_with_header_stays_elf():
    assert elf.should_fallback_to_aosp(has_header=True, dt_type="ELF") is False


def test_forceelf_with_header_stays_elf_even_for_non_elf_dt():
    assert (
        elf.should_fallback_to_aosp(has_header=True, dt_type="QCDT", force_elf=True)
        is False
    )
    assert (
        elf.should_fallback_to_aosp(has_header=True, dt_type=None, force_elf=True)
        is False
    )


# --------------------------------------------------------------------------
# elftool pack argument grammar
# --------------------------------------------------------------------------


def test_pack_arg_bare_path():
    assert elf.pack_arg(Path("/tmp/kernel")) == "/tmp/kernel"


def test_pack_arg_with_load_address():
    assert (
        elf.pack_arg(Path("/tmp/kernel"), address="0x10008000")
        == "/tmp/kernel@0x10008000"
    )


def test_pack_arg_with_section():
    assert (
        elf.pack_arg(Path("/tmp/ramdisk"), section="ramdisk") == "/tmp/ramdisk,ramdisk"
    )
    assert elf.pack_arg(Path("/tmp/rpm"), section="rpm") == "/tmp/rpm,rpm"


def test_pack_arg_with_address_and_section():
    assert (
        elf.pack_arg(Path("/tmp/ramdisk"), section="ramdisk", address="0x200")
        == "/tmp/ramdisk@0x200,ramdisk"
    )


# --------------------------------------------------------------------------
# Magic helpers
# --------------------------------------------------------------------------


def test_is_elf():
    assert elf.is_elf(b"\x7fELF" + b"\x00" * 60) is True
    assert elf.is_elf(b"ANDROID!") is False


def test_is_uboot():
    assert uboot.is_uboot(make_uboot_image(b"x" * 16)) is True
    assert uboot.is_uboot(b"ANDROID!" + b"\x00" * 100) is False


# --------------------------------------------------------------------------
# End-to-end against the bundled binaries
# --------------------------------------------------------------------------


@needs_mkimage
@needs_dumpimage
def test_uboot_roundtrip_against_bundled_tools(tmp_path):
    part0, part1 = tmp_path / "part0.bin", tmp_path / "part1.bin"
    part0.write_bytes(b"\x11" * 4096)
    part1.write_bytes(b"\x22" * 2048)
    img = tmp_path / "uboot.img"

    run_tool(
        "mkimage",
        "-A",
        "ARM",
        "-O",
        "Linux",
        "-T",
        "Multi",
        "-C",
        "none",
        "-a",
        "10008000",
        "-e",
        "10008000",
        "-n",
        "TestUboot",
        "-d",
        f"{part0}:{part1}",
        img,
    )
    assert img.is_file()

    listing = run_tool("dumpimage", "-l", img).stdout
    info = uboot.parse_listing(listing)
    assert info.name == "TestUboot"
    assert info.arch == "arm"
    assert info.os == "linux"
    assert info.type == "Multi"
    assert info.comp == "none"

    raw = img.read_bytes()
    assert uboot.is_uboot(raw)
    assert uboot.trim_size(raw) == len(raw), "mkimage output needs no trimming"

    state = uboot.unpack(img, tmp_path / "work")
    assert state.part_bytes("kernel") == part0.read_bytes()
    assert state.part_bytes("ramdisk") == part1.read_bytes()

    rebuilt = uboot.pack(state, tmp_path / "rebuilt.img")
    assert rebuilt.is_file()
    again = uboot.parse_listing(run_tool("dumpimage", "-l", rebuilt).stdout)
    assert again == info


@needs_mkimage
@needs_dumpimage
def test_uboot_unpack_trims_padded_image(tmp_path):
    part0, part1 = tmp_path / "part0.bin", tmp_path / "part1.bin"
    part0.write_bytes(b"\x33" * 512)
    part1.write_bytes(b"\x44" * 256)
    img = tmp_path / "padded.img"
    run_tool(
        "mkimage",
        "-A",
        "ARM",
        "-O",
        "Linux",
        "-T",
        "Multi",
        "-C",
        "none",
        "-a",
        "0x8000",
        "-e",
        "0x8000",
        "-n",
        "Padded",
        "-d",
        f"{part0}:{part1}",
        img,
    )
    padded = img.read_bytes() + b"\x00" * 256
    padded_path = tmp_path / "padded2.img"
    padded_path.write_bytes(padded)

    assert uboot.trim_size(padded) == len(padded) - 256
    state = uboot.unpack(padded_path, tmp_path / "work2")
    assert state.part_bytes("kernel") == part0.read_bytes()
    assert state.part_bytes("ramdisk") == part1.read_bytes()


@needs_dumpimage
def test_dumpimage_exits_zero_on_a_non_image_but_parsing_rejects_it(tmp_path):
    """dumpimage -l is lenient: it exits 0 on junk and prints a bogus header.

    This is exactly why the parse has to validate rather than trusting a
    zero exit status the way the Bash pipeline does.
    """
    junk = tmp_path / "junk.img"
    junk.write_bytes(b"not a u-boot image" * 8)
    proc = run_tool("dumpimage", "-l", junk)
    assert proc.returncode == 0
    assert "Image Type:" not in proc.stdout
    with pytest.raises(ValueError, match="could not parse"):
        uboot.parse_listing(proc.stdout)


@needs_mboot
def test_mboot_pack_stages_and_cleans_up(tmp_path):
    parts = {
        "header": b"H",
        "kernel": b"K",
        "bootstub": b"B",
        "cmdline": b"console=ttyS0\n",
        "parameter": b"mtdparts=mmcblk0\n",
    }
    work = tmp_path / "work"
    work.mkdir()
    for name, data in parts.items():
        (work / name).write_bytes(data)
    ramdisk = work / "ramdisk"
    ramdisk.write_bytes(gzip.compress(b"cpio-data"))
    state = Unpacked(
        directory=work,
        parts={k: work / k for k in (*parts, "ramdisk")},
        fields={},
    )

    out = tmp_path / "osip.img"
    osip.pack(state, out)
    assert out.is_file()
    assert not (work / osip.STAGING_DIR).exists(), "staging dir must be cleaned up"


def test_mboot_unpack_applies_the_rename_map(tmp_path, monkeypatch):
    """Pin the unpack rename map by stubbing the tool's file emission.

    The bundled mboot cannot unpack a synthesised image (see the sibling test),
    so this stubs the one thing we do not own -- mboot writing its fixed
    filenames into the work dir -- and asserts the rename behaviour that is
    ours: mboot's ``cmdline.txt`` becomes ``cmdline``, ``hdr`` becomes
    ``header``, and stale files with the old names are cleared.
    """

    def fake_run_tool(name, *args, **kwargs):
        workdir = Path(kwargs["cwd"])
        for mboot_name, data in {
            "bootstub": b"B",
            "cmdline.txt": b"console=ttyS0\n",
            "hdr": b"H",
            "kernel": b"K",
            "parameter": b"P",
            "ramdisk.cpio.gz": gzip.compress(b"R"),
            "sig": b"S",
        }.items():
            (workdir / mboot_name).write_bytes(data)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(osip, "run_tool", fake_run_tool)

    dest = tmp_path / "destwork"
    dest.mkdir()
    (dest / "cmdline.txt").write_bytes(b"stale")
    (dest / "hdr").write_bytes(b"stale")
    (dest / "ramdisk.cpio.gz").write_bytes(b"stale")

    img = tmp_path / "osip.img"
    img.write_bytes(b"fake osip image")
    unpacked = osip.unpack(img, dest)

    assert not (dest / "cmdline.txt").exists(), "cmdline.txt must be renamed away"
    assert not (dest / "hdr").exists(), "hdr must be renamed away"
    assert not (dest / "ramdisk.cpio.gz").exists(), (
        "ramdisk.cpio.gz must be renamed away"
    )
    assert {p.name for p in dest.iterdir()} == {
        "bootstub",
        "cmdline",
        "header",
        "kernel",
        "parameter",
        "ramdisk",
        "sig",
    }
    assert unpacked.part_bytes("cmdline") == b"console=ttyS0\n"
    assert unpacked.part_bytes("header") == b"H"
    assert unpacked.part_bytes("sig") == b"S"
    assert gzip.decompress(unpacked.part_bytes("ramdisk")) == b"R"


def test_mboot_unpack_tolerates_missing_parts(tmp_path, monkeypatch):
    """mboot only writes the parts an image actually has; ours must not require all."""

    def fake_run_tool(name, *args, **kwargs):
        workdir = Path(kwargs["cwd"])
        (workdir / "kernel").write_bytes(b"K")
        (workdir / "cmdline.txt").write_bytes(b"console=ttyS0\n")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(osip, "run_tool", fake_run_tool)
    img = tmp_path / "osip.img"
    img.write_bytes(b"fake")
    unpacked = osip.unpack(img, tmp_path / "dest")

    assert set(unpacked.parts) == {"kernel", "cmdline"}
    assert "header" not in unpacked.parts


@needs_mboot
def test_mboot_pack_builds_an_image_from_staged_parts(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    body = random.Random(20260930).randbytes(16384)
    parts = {
        "header": b"H",
        "kernel": body[:8192],
        "bootstub": body[8192:12288],
        "cmdline": b"console=ttyS0\n",
        "parameter": b"mtdparts=mmcblk0\n",
        "ramdisk": gzip.compress(body),
    }
    for name, data in parts.items():
        (work / name).write_bytes(data)
    state = Unpacked(directory=work, parts={k: work / k for k in parts}, fields={})

    out = tmp_path / "src.img"
    osip.pack(state, out)
    assert out.is_file() and out.stat().st_size > 0
    assert not (work / osip.STAGING_DIR).exists(), "staging dir must be cleaned up"


@needs_mboot
def test_mboot_header_handling_is_broken_for_synthetic_images(tmp_path):
    """Document mboot's header limitation instead of pretending it round-trips.

    mboot synthesises the hdr slot rather than preserving it, so on unpack the
    header byte bleeds into the cmdline ("H" + "conso" -> "Hconso"). This is a
    tool bug, not a wrapper bug: the Bash leans on the same tool behaving well
    on the real device images it was written for. Asserted here so a future
    mboot bump that fixes it shows up as a failing test rather than silently
    changing our behaviour.
    """
    work = tmp_path / "work"
    work.mkdir()
    body = random.Random(11).randbytes(4096)
    parts = {
        "header": b"H",
        "kernel": random.Random(12).randbytes(4096),
        "bootstub": random.Random(13).randbytes(4096),
        "cmdline": b"console=ttyS0\n",
        "parameter": b"mtdparts=mmcblk0\n",
        "ramdisk": gzip.compress(body),
    }
    for name, data in parts.items():
        (work / name).write_bytes(data)
    state = Unpacked(directory=work, parts={k: work / k for k in parts}, fields={})
    out = tmp_path / "out.img"
    osip.pack(state, out)
    assert out.is_file()

    dest = tmp_path / "dest"
    dest.mkdir()
    try:
        unpacked = osip.unpack(out, dest)
    except ToolError as exc:
        # A different header width desynchronises the layout outright and mboot
        # refuses to unpack; the Bash hits the same wall.
        assert "likely wrong" in str(exc), exc
    else:
        cmdline = unpacked.part_bytes("cmdline")
        assert cmdline != parts["cmdline"], (
            "mboot is expected to corrupt the cmdline with the header byte; "
            f"got {cmdline!r}"
        )
        assert cmdline.startswith(b"H"), cmdline


@pytest.mark.parametrize(
    "missing", ["header", "kernel", "bootstub", "cmdline", "parameter", "ramdisk"]
)
def test_mboot_missing_required_part_is_reported_clearly(tmp_path, missing):
    """A missing input is named up front instead of surfacing as a tool error.

    mboot itself aborts with a bare "cannot open input file 'cmdline.txt'",
    which reads like a tool bug rather than a missing part in the state.
    """
    work = tmp_path / "work"
    work.mkdir()
    parts = {
        "header": b"H",
        "kernel": b"K",
        "bootstub": b"B",
        "cmdline": b"console=ttyS0\n",
        "parameter": b"mtdparts=mmcblk0\n",
        "ramdisk": gzip.compress(b"R"),
    }
    del parts[missing]
    for name, data in parts.items():
        (work / name).write_bytes(data)
    state = Unpacked(directory=work, parts={k: work / k for k in parts}, fields={})

    with pytest.raises(ValueError, match=missing):
        osip.pack(state, tmp_path / "out.img")
    assert not (work / osip.STAGING_DIR).exists(), (
        "staging dir cleaned up even on failure"
    )


@needs_rkcrc
def test_rkcrc_roundtrip(tmp_path):
    ramdisk = tmp_path / "ramdisk"
    payload = b"rockchip-ramdisk" * 64
    ramdisk.write_bytes(payload)
    state = Unpacked(directory=tmp_path, parts={"ramdisk": ramdisk}, fields={})
    out = krnl.pack(state, tmp_path / "out.img")
    assert out.is_file()
    packed = out.read_bytes()
    assert packed.startswith(krnl.MAGIC), "rkcrc writes the KRNL tag is_krnl checks for"
    assert payload in packed, "rkcrc must embed the ramdisk"
    assert krnl.is_krnl(packed) is True


@needs_elftool
def test_elftool_pack_requires_kernel(tmp_path):
    with pytest.raises(ToolError):
        run_tool("elftool", "pack", "-o", str(tmp_path / "out.img"))


@needs_elftool
def test_unpackelf_rejects_non_elf(tmp_path):
    junk = tmp_path / "junk.img"
    junk.write_bytes(b"\x00" * 256)
    with pytest.raises(ToolError):
        run_tool("unpackelf", "-i", str(junk))
