"""Tests for the repack path (``aik_py.repack``).

The round-trip tests exercise the cpio writer against GNU cpio where it is
available, because "our archive is internally consistent" is a much weaker
claim than "GNU cpio extracts it".
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aik_py.repack import (
    BUMP_FOOTER,
    SEANDROID_FOOTER,
    CpioEntry,
    RepackError,
    RepackOptions,
    RepackPlan,
    append_footer,
    build_archive,
    clear_scratch,
    compress_ramdisk,
    compression_extension,
    empty_ramdisk,
    pack_ramdisk,
    pad_to_size,
    read_newc,
    repack_ramdisk,
    resolve_level,
    write_newc,
)

HAVE_GNU_CPIO = shutil.which("cpio") is not None
HAVE_LZ4 = shutil.which("lz4") is not None
_BUNDLED_LZ4 = Path(__file__).resolve().parents[1] / "bin/linux/x86_64/lz4"
requires_cpio = pytest.mark.skipif(not HAVE_GNU_CPIO, reason="GNU cpio not installed")
requires_lz4 = pytest.mark.skipif(
    not (HAVE_LZ4 or _BUNDLED_LZ4.is_file()),
    reason="no lz4 binary available",
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def ramdisk(tmp_path: Path) -> Path:
    """A small synthetic ramdisk with a known, fixed mtime.

    Deliberately has two non-empty sibling directories (``a`` and ``sub``) that
    both sort before/after a file. A single-directory tree would let a
    depth-grouped walk agree with a global sort by coincidence, which is
    exactly the ordering bug the byte-exactness test must be able to catch.
    """
    import os

    root = tmp_path / "ramdisk"
    (root / "a").mkdir(parents=True)
    (root / "sub").mkdir(parents=True)
    (root / "init").write_bytes(b"#!/bin/sh\nexit 0\n")
    (root / "init").chmod(0o755)
    (root / "a" / "one").write_bytes(b"1\n")
    (root / "sub" / "data").write_bytes(b"payload\n")
    for path in sorted(root.rglob("*")):
        os.utime(path, (0, 0), follow_symlinks=False)
    os.utime(root, (0, 0))
    return root


# --------------------------------------------------------------------------
# newc writer
# --------------------------------------------------------------------------


def test_archive_contains_exact_entries(ramdisk: Path) -> None:
    entries = read_newc(build_archive(ramdisk))
    by_name = {e.name: e for e in entries}
    assert set(by_name) == {".", "a", "a/one", "init", "sub", "sub/data"}
    assert by_name["init"].data == b"#!/bin/sh\nexit 0\n"
    assert by_name["init"].mode == stat.S_IFREG | 0o755
    assert by_name["sub"].mode == stat.S_IFDIR | 0o755
    assert by_name["."].mode == stat.S_IFDIR | 0o755
    assert by_name["sub/data"].data == b"payload\n"
    assert by_name["a/one"].data == b"1\n"


def test_entries_are_in_sorted_path_order(ramdisk: Path) -> None:
    """Global path sort, so a parent always precedes its children.

    A per-directory walk or a LIFO stack emits `a/one` after `sub`, which
    would make the archive differ from a sorted `find | cpio`.
    """
    names = [e.name for e in read_newc(build_archive(ramdisk))]
    assert names == [".", "a", "a/one", "init", "sub", "sub/data"]


def test_entries_are_owned_by_root(ramdisk: Path) -> None:
    """Mirrors the Bash's `cpio -R 0:0`: uid/gid are 0 regardless of the host."""
    for entry in read_newc(build_archive(ramdisk)):
        assert entry.uid == 0
        assert entry.gid == 0


def test_mtimes_are_preserved(ramdisk: Path) -> None:
    for entry in read_newc(build_archive(ramdisk)):
        assert entry.mtime == 0, f"{entry.name} lost its recorded mtime"


def test_archive_is_padded_to_512(ramdisk: Path) -> None:
    """GNU cpio rounds the archive up to a whole 512-byte block."""
    blob = build_archive(ramdisk)
    assert len(blob) % 512 == 0
    # The pad after the trailer record is what closes the last block, so it is
    # always shorter than one full block.
    trailer = blob.index(b"TRAILER!!!")
    pad = blob[trailer:]
    assert len(pad) - len(pad.rstrip(b"\x00")) < 512


def test_archive_ends_with_trailer(ramdisk: Path) -> None:
    blob = build_archive(ramdisk)
    assert blob.rindex(b"TRAILER!!!") < len(blob)
    # Everything after the trailer record is alignment padding.
    trailer = blob.index(b"TRAILER!!!")
    assert set(blob[trailer + len(b"TRAILER!!!\0") :]) <= {0x00, 0x30}


def test_symlinks_are_stored_as_links(tmp_path: Path) -> None:
    root = tmp_path / "ramdisk"
    root.mkdir()
    (root / "target").write_bytes(b"x")
    (root / "link").symlink_to("target")
    entries = {e.name: e for e in read_newc(build_archive(root))}
    assert stat.S_ISLNK(entries["link"].mode)
    assert entries["link"].data == b"target"


def test_hardlinks_share_an_inode(tmp_path: Path) -> None:
    root = tmp_path / "ramdisk"
    root.mkdir()
    (root / "a").write_bytes(b"same")
    (root / "b").hardlink_to(root / "a")
    entries = {e.name: e for e in read_newc(build_archive(root))}
    assert entries["a"].ino == entries["b"].ino


def test_nlink_matches_subdir_count(tmp_path: Path) -> None:
    """A directory's link count is 2 + its number of subdirectories."""
    root = tmp_path / "ramdisk"
    (root / "a" / "b").mkdir(parents=True)
    (root / "c").mkdir()
    entries = {e.name: e for e in read_newc(build_archive(root))}
    assert entries["."].nlink == 4  # 2 + subdirs a and c
    assert entries["a"].nlink == 3  # 2 + subdir b
    assert entries["a/b"].nlink == 2  # 2 + none


def test_write_newc_roundtrips() -> None:
    entries = [
        CpioEntry(name="dir", mode=stat.S_IFDIR | 0o755),
        CpioEntry(name="dir/f", mode=stat.S_IFREG | 0o644, data=b"abc"),
    ]
    parsed = read_newc(write_newc(entries))
    assert [e.name for e in parsed] == ["dir", "dir/f"]
    assert parsed[1].data == b"abc"
    assert parsed[1].mode == stat.S_IFREG | 0o644


# --------------------------------------------------------------------------
# GNU cpio interoperability -- the real gate
# --------------------------------------------------------------------------


@requires_cpio
def test_gnu_cpio_extracts_our_archive(ramdisk: Path, tmp_path: Path) -> None:
    """The real gate: GNU cpio must be able to read what we write."""
    blob = compress_ramdisk(build_archive(ramdisk), "gzip")
    archive = tmp_path / "ramdisk.cpio.gz"
    archive.write_bytes(blob)

    dest = tmp_path / "extract"
    dest.mkdir()
    # cpio reads an uncompressed stream, so decompress into its stdin.
    subprocess.run(["cpio", "-idm"], cwd=dest, input=gzip.decompress(blob), check=True)

    assert (dest / "init").read_bytes() == b"#!/bin/sh\nexit 0\n"
    assert (dest / "sub" / "data").read_bytes() == b"payload\n"
    assert stat.S_IMODE((dest / "init").stat().st_mode) == 0o755
    assert (dest / "init").stat().st_mtime == 0


@requires_cpio
def test_byte_identical_to_gnu_cpio(ramdisk: Path) -> None:
    """Our writer should match `find . | sort | cpio -o -H newc -R 0:0` exactly.

    We sort where the Bash's bare `find .` does not, so we feed GNU cpio the
    same sorted order to make the comparison meaningful.
    """
    listing = subprocess.run(
        ["find", ".", "-mindepth", "0"],
        cwd=ramdisk,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    entries = sorted(line for line in listing.splitlines() if line)

    ref = subprocess.run(
        ["cpio", "-o", "-H", "newc", "-R", "0:0"],
        cwd=ramdisk,
        input=("\n".join(entries) + "\n").encode(),
        capture_output=True,
        check=True,
    ).stdout

    assert build_archive(ramdisk) == ref


# --------------------------------------------------------------------------
# compression
# --------------------------------------------------------------------------


def test_xz_defaults_to_level_1() -> None:
    assert resolve_level("xz", None) == 1
    assert resolve_level("lzma", None) == 1


def test_lz4_defaults_to_level_9() -> None:
    assert resolve_level("lz4", None) == 9
    assert resolve_level("lz4-l", None) == 9


def test_other_codecs_use_their_default() -> None:
    # 0xFFFFFFFF is our "let the compressor choose" sentinel.
    assert resolve_level("gzip", None) == 0xFFFFFFFF
    assert resolve_level("bzip2", None) == 0xFFFFFFFF
    assert resolve_level("cpio", None) == 0xFFFFFFFF


def test_explicit_level_overrides_default() -> None:
    assert resolve_level("xz", 6) == 6
    assert resolve_level("lz4", 3) == 3


def test_xz_level_changes_output() -> None:
    payload = b"A" * 100_000
    assert compress_ramdisk(payload, "xz", 1) != compress_ramdisk(payload, "xz", 9)


def test_xz_uses_crc32_check() -> None:
    """The Bash invokes `xz -Ccrc32`; the check bytes must match."""
    blob = compress_ramdisk(b"hello" * 1000, "xz")
    assert lzma.decompress(blob) == b"hello" * 1000
    assert blob[:6] == b"\xfd7zXZ\x00"
    # XZ stream flags: the low 4 bits of byte 7 are the integrity check id,
    # where 0x01 is CRC32 and 0x04 (the .lzma default) would be CRC64.
    assert blob[7] & 0x0F == 0x01


def test_gzip_is_deterministic_and_nameless() -> None:
    payload = b"hello" * 100
    a = compress_ramdisk(payload, "gzip")
    b = compress_ramdisk(payload, "gzip")
    assert a == b
    assert not a[3] & 0x08, "FNAME flag set; the Bash's gzip stores no name"
    assert gzip.decompress(a) == payload


def test_bzip2_roundtrips() -> None:
    payload = b"hello" * 100
    assert bz2.decompress(compress_ramdisk(payload, "bzip2")) == payload


def test_cpio_compression_is_identity() -> None:
    payload = b"raw cpio bytes"
    assert compress_ramdisk(payload, "cpio") == payload


def test_unknown_compression_rejected() -> None:
    with pytest.raises(RepackError, match="Unsupported format"):
        compress_ramdisk(b"x", "brotli")


@pytest.mark.parametrize(
    ("comp", "ext"),
    [
        ("gzip", ".gz"),
        ("lzop", ".lzo"),
        ("bzip2", ".bz2"),
        ("xz", ".xz"),
        ("lzma", ".lzma"),
        ("lz4", ".lz4"),
        ("lz4-l", ".lz4"),
        ("cpio", ""),
        ("empty", ".empty"),
    ],
)
def test_compression_extensions_match_bash(comp: str, ext: str) -> None:
    assert compression_extension(comp) == ext


@requires_lz4
def test_lz4_roundtrips() -> None:
    payload = b"hello" * 100
    assert compress_ramdisk(payload, "lz4", 9)[:4] == b"\x04\x22\x4d\x18"


@requires_lz4
def test_lz4_legacy_frame_magic() -> None:
    payload = b"hello" * 100
    assert compress_ramdisk(payload, "lz4-l", 9)[:4] == b"\x02\x21\x4c\x18"


# --------------------------------------------------------------------------
# level validation -- the Bash silently ignores a bad --level
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1, 10, 99])
def test_out_of_range_level_raises(bad: int) -> None:
    with pytest.raises(RepackError, match="out of range"):
        resolve_level("xz", bad)
    with pytest.raises(RepackError, match="out of range"):
        RepackOptions(level=bad)


@pytest.mark.parametrize("bad", ["9", 3.5, None.__class__, True])
def test_non_integer_level_raises(bad: object) -> None:
    with pytest.raises(RepackError, match="out of range|must be an integer"):
        resolve_level("xz", bad)  # type: ignore[arg-type]


def test_options_reject_bad_level_eagerly() -> None:
    """Validation happens in __post_init__, not deep in the pipeline."""
    with pytest.raises(RepackError, match="out of range"):
        RepackOptions(level=42)


# --------------------------------------------------------------------------
# pack_ramdisk
# --------------------------------------------------------------------------


def test_pack_ramdisk_produces_decompressible_output(ramdisk: Path) -> None:
    payload = pack_ramdisk(ramdisk, "gzip")
    assert gzip.decompress(payload) == build_archive(ramdisk)


def test_pack_ramdisk_is_reproducible(ramdisk: Path) -> None:
    assert pack_ramdisk(ramdisk, "xz") == pack_ramdisk(ramdisk, "xz")


def test_pack_ramdisk_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(RepackError, match="ramdisk directory not found"):
        pack_ramdisk(tmp_path / "nope", "gzip")


def test_pack_ramdisk_rejects_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "ramdisk"
    empty.mkdir()
    with pytest.raises(RepackError, match="empty"):
        pack_ramdisk(empty, "gzip")


# --------------------------------------------------------------------------
# the repack_ramdisk stage
# --------------------------------------------------------------------------


def test_original_reuses_the_archive_without_repacking(
    ramdisk: Path, tmp_path: Path
) -> None:
    work = tmp_path / "work"
    (work / "ramdisk").mkdir(parents=True)
    shutil.copytree(ramdisk, work / "ramdisk", dirs_exist_ok=True)
    original = work / "orig-ramdisk.cpio.gz"
    sentinel = b"\x1f\x8b" + b"pre-existing archive bytes"
    original.write_bytes(sentinel)

    plan = RepackPlan(ramdisk=work / "ramdisk")
    result = repack_ramdisk(work, plan, "gzip", original, None)

    assert result == original
    assert result.read_bytes() == sentinel
    # No new archive was produced.
    assert not (work / "ramdisk-new.cpio.gz").exists()


def test_repack_ramdisk_writes_a_new_archive(ramdisk: Path, tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "ramdisk").mkdir(parents=True)
    shutil.copytree(ramdisk, work / "ramdisk", dirs_exist_ok=True)
    plan = RepackPlan(ramdisk=work / "ramdisk")
    out = repack_ramdisk(work, plan, "gzip", None, None)
    assert out == work / "ramdisk-new.cpio.gz"
    assert gzip.decompress(out.read_bytes()) == build_archive(work / "ramdisk")


def test_empty_ramdisk_writes_zero_length_file(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "ramdisk").mkdir(parents=True)
    plan = RepackPlan(ramdisk=work / "ramdisk", is_empty_ramdisk=True)
    out = repack_ramdisk(work, plan, "empty", None, None)

    assert out.name == "ramdisk-new.cpio.empty"
    assert out.read_bytes() == b""
    assert out.stat().st_size == 0
    assert plan.is_empty_ramdisk is True


def test_empty_ramdisk_helper_is_zero_length() -> None:
    assert empty_ramdisk() == b""
    assert len(empty_ramdisk()) == 0


# --------------------------------------------------------------------------
# scratch cleanup
# --------------------------------------------------------------------------


def test_clear_scratch_removes_stale_new_files(tmp_path: Path) -> None:
    (tmp_path / "ramdisk-new.cpio.gz").write_bytes(b"stale")
    (tmp_path / "kernel-new.mtk").write_bytes(b"stale")
    (tmp_path / "image-new.img").write_bytes(b"stale")
    keeper = tmp_path / "split_img"
    keeper.mkdir()
    (keeper / "image-origsize").write_text("12345")

    removed = clear_scratch(tmp_path)

    assert len(removed) == 3
    assert sorted(p.name for p in removed) == [
        "image-new.img",
        "kernel-new.mtk",
        "ramdisk-new.cpio.gz",
    ]
    assert not (tmp_path / "ramdisk-new.cpio.gz").exists()
    assert not (tmp_path / "image-new.img").exists()
    # Files that are not scratch survive.
    assert (keeper / "image-origsize").read_text() == "12345"


def test_clear_scratch_is_a_noop_on_a_clean_dir(tmp_path: Path) -> None:
    (tmp_path / "image-new.img").parent.mkdir(exist_ok=True)
    (tmp_path / "keep.txt").write_text("hi")
    assert clear_scratch(tmp_path) == []
    assert (tmp_path / "keep.txt").exists()


# --------------------------------------------------------------------------
# padding
# --------------------------------------------------------------------------


def test_pad_grows_the_image(tmp_path: Path) -> None:
    img = tmp_path / "image-new.img"
    img.write_bytes(b"a" * 100)
    assert pad_to_size(img, 1000) == 1000
    assert img.stat().st_size == 1000
    assert img.read_bytes() == b"a" * 100 + b"\0" * 900


def test_pad_is_a_noop_at_the_right_size(tmp_path: Path) -> None:
    img = tmp_path / "image-new.img"
    img.write_bytes(b"a" * 100)
    assert pad_to_size(img, 100) == 100
    assert img.read_bytes() == b"a" * 100


def test_pad_refuses_to_truncate_signed_data(tmp_path: Path) -> None:
    """The Bash's `truncate` here would silently amputate a signature."""
    img = tmp_path / "image-new.img"
    signed = b"image" + BUMP_FOOTER
    img.write_bytes(signed)

    with pytest.raises(RepackError, match="refusing to pad"):
        pad_to_size(img, 10)

    # The good image is left intact rather than chopped.
    assert img.read_bytes() == signed


def test_pad_can_still_truncate_when_not_strict(tmp_path: Path) -> None:
    img = tmp_path / "image-new.img"
    img.write_bytes(b"a" * 100)
    assert pad_to_size(img, 10, strict=False) == 10
    assert img.read_bytes() == b"a" * 10


# --------------------------------------------------------------------------
# footer
# --------------------------------------------------------------------------


def test_bump_footer_is_appended(tmp_path: Path) -> None:
    img = tmp_path / "image-new.img"
    img.write_bytes(b"image")
    append_footer(img, "Bump")
    assert img.read_bytes() == b"image" + BUMP_FOOTER
    assert len(BUMP_FOOTER) == 16


def test_seandroid_footer_is_appended(tmp_path: Path) -> None:
    img = tmp_path / "image-new.img"
    img.write_bytes(b"image")
    append_footer(img, "SEAndroid")
    assert img.read_bytes() == b"image" + SEANDROID_FOOTER


def test_no_footer_when_unspecified(tmp_path: Path) -> None:
    img = tmp_path / "image-new.img"
    img.write_bytes(b"image")
    append_footer(img, None)
    assert img.read_bytes() == b"image"


def test_unknown_footer_rejected(tmp_path: Path) -> None:
    img = tmp_path / "image-new.img"
    img.write_bytes(b"image")
    with pytest.raises(RepackError, match="Unsupported format"):
        append_footer(img, "AVBv2")


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------


def test_options_defaults() -> None:
    opts = RepackOptions()
    assert opts.original is False
    assert opts.origsize is False
    assert opts.level is None
    assert opts.avbkey is None
    assert opts.forceelf is False
    assert opts.aboot is None


def test_options_accepts_aboot_path(tmp_path: Path) -> None:
    aboot = tmp_path / "aboot.img"
    assert RepackOptions(aboot=aboot).aboot == aboot


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


@dataclass
class FakeManifest:
    """Stands in for aik_py.manifest, which another unit owns."""

    img_type: str = "AOSP"
    ramdisk_compression: str = "gzip"
    original_size: int | None = None
    signature_type: str | None = None
    avb_type: str | None = None
    blob_type: str | None = None
    mtk_type: str | None = None
    loki_type: str | None = None
    footer_type: str | None = None
    microloader: Path | None = None


@pytest.fixture
def work(tmp_path: Path, ramdisk: Path) -> Path:
    """An AIK-style working directory with ramdisk/ and split_img/ populated."""
    root = tmp_path / "work"
    (root / "ramdisk").mkdir(parents=True)
    for item in ramdisk.iterdir():
        target = root / "ramdisk" / item.name
        if item.is_dir():
            target.mkdir()
            for sub in item.iterdir():
                shutil.copy2(sub, target / sub.name)
        else:
            shutil.copy2(item, target)
    (root / "split_img").mkdir()
    (root / "split_img" / "boot-imgtype").write_text("AOSP\n")
    (root / "split_img" / "boot-kernel").write_bytes(b"kernel bytes")
    return root


@pytest.fixture
def fake_build(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Stub out the sibling `formats`/`avb` units and record what was asked."""
    calls: dict[str, object] = {}
    import aik_py.repack as mod

    def build(plan, out, manifest):
        calls["build_out"] = Path(out)
        calls["build_kernel"] = plan.kernel
        calls["build_ramdisk"] = plan.ramdisk
        Path(out).write_bytes(b"built image")

    def sign(plan, out, opts, manifest):
        calls["sign_out"] = Path(out)
        out.write_bytes(b"signed image")
        return out

    monkeypatch.setattr(mod, "build_image", build)
    monkeypatch.setattr(mod, "sign_image", sign)
    return calls


def test_repack_reports_missing_inputs(tmp_path: Path) -> None:
    from aik_py.repack import repack

    with pytest.raises(RepackError, match="No files found"):
        repack(tmp_path, FakeManifest())


def test_repack_reports_empty_ramdisk(tmp_path: Path) -> None:
    from aik_py.repack import repack

    (tmp_path / "ramdisk").mkdir()
    (tmp_path / "split_img").mkdir()
    (tmp_path / "split_img" / "boot-imgtype").write_text("AOSP\n")
    with pytest.raises(RepackError, match="No files found"):
        repack(tmp_path, FakeManifest())


def test_loki_requires_aboot(tmp_path: Path) -> None:
    from aik_py.repack import apply_loki

    img = tmp_path / "image-new.img"
    img.write_bytes(b"image")
    plan = RepackPlan(ramdisk=tmp_path / "ramdisk")

    with pytest.raises(RepackError, match="aboot.img required"):
        apply_loki(plan, img, "aboot", aboot=None)


def test_loki_missing_aboot_file_is_reported(tmp_path: Path) -> None:
    from aik_py.repack import apply_loki

    img = tmp_path / "image-new.img"
    img.write_bytes(b"image")
    plan = RepackPlan(ramdisk=tmp_path / "ramdisk")

    with pytest.raises(RepackError, match="aboot.img required"):
        apply_loki(plan, img, "aboot", aboot=tmp_path / "missing.img")


def test_loki_absent_dependency_is_named_after_the_input_check(
    tmp_path: Path,
) -> None:
    """A missing aboot.img must be reported even when `patches` does not exist."""
    from aik_py.repack import apply_loki

    img = tmp_path / "image-new.img"
    img.write_bytes(b"image")
    with pytest.raises(RepackError, match="aboot.img required"):
        apply_loki(RepackPlan(ramdisk=tmp_path / "ramdisk"), img, "aboot", None)


def test_loki_is_skipped_when_not_recorded(tmp_path: Path) -> None:
    from aik_py.repack import apply_loki

    img = tmp_path / "image-new.img"
    img.write_bytes(b"image")
    assert apply_loki(RepackPlan(ramdisk=tmp_path / "ramdisk"), img, None, None) == img
    assert img.read_bytes() == b"image"


def test_amonet_is_skipped_when_not_recorded(tmp_path: Path) -> None:
    from aik_py.repack import apply_amonet

    img = tmp_path / "image-new.img"
    img.write_bytes(b"image")
    assert apply_amonet(RepackPlan(ramdisk=tmp_path), img, None) == img
    assert img.read_bytes() == b"image"


def test_signature_forks_the_output_name(work: Path, fake_build: dict) -> None:
    """A recorded signature means build-to-unsigned then sign; else build direct."""
    from aik_py.repack import repack

    repack(work, FakeManifest())
    assert fake_build["build_out"].name == "image-new.img"
    assert "sign_out" not in fake_build

    fake_build.clear()
    repack(work, FakeManifest(signature_type="AVBv1"))
    assert fake_build["build_out"].name == "unsigned-new.img"
    assert fake_build["sign_out"].name == "image-new.img"


def test_signed_image_lands_in_image_new_img(work: Path, fake_build: dict) -> None:
    from aik_py.repack import repack

    repack(work, FakeManifest(signature_type="AVBv1"))
    assert (work / "image-new.img").read_bytes() == b"signed image"
    assert (work / "unsigned-new.img").read_bytes() == b"built image"


def test_dhtb_signature_suppresses_the_footer(work: Path, fake_build: dict) -> None:
    """A DHTB image carries a DHTB footer, so no Bump/SEAndroid tail is added."""
    from aik_py.repack import repack

    repack(work, FakeManifest(signature_type="DHTB", footer_type="Bump"))
    assert (work / "image-new.img").read_bytes() == b"signed image"


def test_footer_is_appended_when_not_dhtb(work: Path, fake_build: dict) -> None:
    from aik_py.repack import repack

    repack(work, FakeManifest(footer_type="Bump"))
    assert (work / "image-new.img").read_bytes() == b"built image" + BUMP_FOOTER


def test_stale_scratch_is_cleared_before_repacking(
    work: Path, fake_build: dict
) -> None:
    from aik_py.repack import repack

    stale = work / "ramdisk-new.cpio.gz"
    stale.write_bytes(b"stale from a previous run")
    other = work / "unlokied-new.img"
    other.write_bytes(b"stale too")

    repack(work, FakeManifest())

    assert gzip.decompress(stale.read_bytes())  # replaced, not left stale
    assert not other.exists()


def test_origsize_pads_the_image(work: Path, fake_build: dict) -> None:
    from aik_py.repack import RepackOptions, repack

    manifest = FakeManifest(original_size=len(b"built image") + 512)
    repack(work, manifest, RepackOptions(origsize=True))

    out = work / "image-new.img"
    assert out.stat().st_size == manifest.original_size
    assert out.read_bytes().startswith(b"built image")
    assert out.read_bytes().endswith(b"\0" * 512)


def test_origsize_refuses_to_truncate(work: Path, fake_build: dict) -> None:
    from aik_py.repack import RepackOptions, repack

    manifest = FakeManifest(signature_type="AVBv1", original_size=4)
    with pytest.raises(RepackError, match="refusing to pad"):
        repack(work, manifest, RepackOptions(origsize=True))

    # The correctly-built signed image survives untouched.
    assert (work / "image-new.img").read_bytes() == b"signed image"


def test_origsize_without_a_recorded_size_is_a_noop(
    work: Path, fake_build: dict
) -> None:
    from aik_py.repack import RepackOptions, repack

    repack(work, FakeManifest(original_size=None), RepackOptions(origsize=True))
    assert (work / "image-new.img").read_bytes() == b"built image"


def test_original_option_uses_the_recorded_archive(
    work: Path, fake_build: dict
) -> None:
    from aik_py.repack import RepackOptions, repack

    original = work / "split_img" / "boot-ramdisk.cpio.gz"
    sentinel = gzip.compress(b"the original archive")
    original.write_bytes(sentinel)

    repack(work, FakeManifest(), RepackOptions(original=True))

    assert fake_build["build_ramdisk"] == original
    assert not (work / "ramdisk-new.cpio.gz").exists()


def test_repack_passes_kernel_and_ramdisk_to_the_builder(
    work: Path, fake_build: dict
) -> None:
    from aik_py.repack import repack

    repack(work, FakeManifest())
    assert fake_build["build_kernel"] == work / "split_img" / "boot-kernel"
    assert fake_build["build_ramdisk"] == work / "ramdisk-new.cpio.gz"
    assert Path(fake_build["build_ramdisk"]).is_file()


# --------------------------------------------------------------------------
# regressions found in review
# --------------------------------------------------------------------------


def test_sign_image_does_not_assign_to_the_readonly_signed_property(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DHTB repack used to raise AttributeError on the slots dataclass."""
    import aik_py.repack as mod

    fake_module = type("M", (), {"sign": staticmethod(lambda **kw: None)})()
    monkeypatch.setattr(mod.importlib, "import_module", lambda n: fake_module)

    out = tmp_path / "image-new.img"
    unsigned = tmp_path / "unsigned-new.img"
    unsigned.write_bytes(b"unsigned")
    plan = RepackPlan(ramdisk=tmp_path / "ramdisk", unsigned=unsigned)

    # DHTB must not blow up here; the real code path raised AttributeError.
    result = mod.sign_image(
        plan, out, RepackOptions(), FakeManifest(signature_type="DHTB")
    )
    assert result == out
    assert plan.output == out


def test_sign_image_falls_back_to_the_bundled_avb_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """repackimg.sh signs with $bin/avb/verity when --avbkey is omitted."""
    import aik_py.repack as mod

    seen: dict[str, object] = {}
    fake_module = type("M", (), {"sign": staticmethod(lambda **kw: seen.update(kw))})()
    monkeypatch.setattr(mod.importlib, "import_module", lambda n: fake_module)
    monkeypatch.setattr(mod, "_default_avbkey", lambda: "/bundled/verity")

    unsigned = tmp_path / "unsigned-new.img"
    unsigned.write_bytes(b"unsigned")
    plan = RepackPlan(ramdisk=tmp_path / "ramdisk", unsigned=unsigned)
    mod.sign_image(
        plan,
        tmp_path / "image-new.img",
        RepackOptions(),
        FakeManifest(signature_type="AVBv1"),
    )

    assert seen["avbkey"] == "/bundled/verity"


def test_sign_image_keeps_an_explicit_avbkey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aik_py.repack as mod

    seen: dict[str, object] = {}
    fake_module = type("M", (), {"sign": staticmethod(lambda **kw: seen.update(kw))})()
    monkeypatch.setattr(mod.importlib, "import_module", lambda n: fake_module)

    unsigned = tmp_path / "unsigned-new.img"
    unsigned.write_bytes(b"unsigned")
    plan = RepackPlan(ramdisk=tmp_path / "ramdisk", unsigned=unsigned)
    mod.sign_image(
        plan,
        tmp_path / "image-new.img",
        RepackOptions(avbkey="mykey"),
        FakeManifest(signature_type="AVBv1"),
    )

    assert seen["avbkey"] == "mykey"


def test_gzip_level_is_honoured() -> None:
    """`--level 1` must actually change the gzip output, not be dropped."""
    payload = b"A" * 200_000
    assert compress_ramdisk(payload, "gzip", 1) != compress_ramdisk(payload, "gzip", 9)


def test_bzip2_level_is_honoured() -> None:
    payload = b"A" * 200_000
    assert compress_ramdisk(payload, "bzip2", 1) != compress_ramdisk(
        payload, "bzip2", 9
    )


def test_options_reject_a_non_integer_level_eagerly() -> None:
    """A float or str must fail in __post_init__, not deep in the pipeline."""
    with pytest.raises(RepackError, match="must be an integer|out of range"):
        RepackOptions(level=3.5)
    with pytest.raises(RepackError, match="must be an integer|out of range"):
        RepackOptions(level="9")


def test_elf_without_rpm_is_downgraded_to_aosp(work: Path, fake_build: dict) -> None:
    from aik_py.repack import repack

    plan = repack(work, FakeManifest(img_type="ELF"))
    assert plan.img_type == "AOSP"


def test_forceelf_keeps_the_elf_format(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aik_py.repack as mod

    seen: dict[str, object] = {}

    def build(**kw):
        seen.update(kw)
        Path(kw["out"]).write_bytes(b"elf image")

    fake_module = type("M", (), {"build_image": staticmethod(build)})()
    monkeypatch.setattr(mod.importlib, "import_module", lambda n: fake_module)

    plan = mod.repack(work, FakeManifest(img_type="ELF"), RepackOptions(forceelf=True))
    assert plan.img_type == "ELF"
    assert seen["img_type"] == "ELF"


def test_dttype_eleven_implies_forceelf(
    work: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """repackimg.sh forces ELF repack whenever the recorded dttype is ELF."""
    import aik_py.repack as mod

    (work / "split_img" / "boot-dttype").write_text("ELF\n")
    seen: dict[str, object] = {}

    def build(**kw):
        seen.update(kw)
        Path(kw["out"]).write_bytes(b"elf image")

    fake_module = type("M", (), {"build_image": staticmethod(build)})()
    monkeypatch.setattr(mod.importlib, "import_module", lambda n: fake_module)

    plan = mod.repack(work, FakeManifest(img_type="ELF"))
    assert plan.img_type == "ELF"


def test_used_original_ramdisk_is_recorded(work: Path, fake_build: dict) -> None:
    from aik_py.repack import repack

    (work / "split_img" / "boot-ramdisk.cpio.gz").write_bytes(gzip.compress(b"orig"))
    plan = repack(work, FakeManifest(), RepackOptions(original=True))
    assert plan.used_original_ramdisk is True


def test_lzop_is_recognised_by_the_extension_table() -> None:
    """lzo has no stdlib codec, but its absence must not be a silent surprise."""
    assert compression_extension("lzop") == ".lzo"
    bundled = Path(__file__).resolve().parents[1] / "bin/linux/x86_64/lzop"
    if shutil.which("lzop") is None and not bundled.is_file():
        with pytest.raises(RepackError, match="lzop binary"):
            compress_ramdisk(b"data", "lzop")
