import lzma
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from aik_py import bin
from aik_py.compression import (
    Compression,
    CorruptImageError,
    MissingToolError,
    UnsupportedCompressionError,
    compress,
    decompress,
    detect,
    extension,
)

PAYLOAD = b"init\n" * 64 + bytes(range(256)) * 8

# cpio detection keys off the newc magic, so the identity round-trip needs a
# payload that actually starts with it.
CPIO_PAYLOAD = b"070701" + b"0" * 1080

PURE_FORMATS = [
    Compression.GZIP,
    Compression.BZIP2,
    Compression.XZ,
    Compression.LZMA,
    Compression.LZ4,
    Compression.LZ4_LEGACY,
    Compression.CPIO,
]


def _lzop_available() -> bool:
    try:
        bin.tool_path("lzop")
    except bin.MissingToolError:
        return False
    return True


LZOP = pytest.mark.skipif(not _lzop_available(), reason="no lzop binary available")


@pytest.mark.parametrize("comp", PURE_FORMATS)
def test_round_trip(comp: Compression) -> None:
    blob = compress(PAYLOAD, comp)
    assert decompress(blob, comp) == PAYLOAD


@pytest.mark.parametrize("comp", PURE_FORMATS)
def test_detect_matches_input(comp: Compression) -> None:
    payload = CPIO_PAYLOAD if comp is Compression.CPIO else PAYLOAD
    assert detect(compress(payload, comp)) is comp


@pytest.mark.parametrize(
    ("magic", "expected"),
    [
        (b"\x1f\x8b\x08\x00", Compression.GZIP),
        (b"BZh9", Compression.BZIP2),
        (b"\xfd7zXZ\x00", Compression.XZ),
        (b"\x5d\x00\x00\x80\x00", Compression.LZMA),
        (b"\x89LZO\x00\r\n\x1a\n", Compression.LZOP),
        (struct.pack("<I", 0x184D2204), Compression.LZ4),
        (struct.pack("<I", 0x184C2102), Compression.LZ4_LEGACY),
        (b"070701" + b"\x00" * 500, Compression.CPIO),
        (b"070702" + b"\x00" * 500, Compression.CPIO),
        (b"070707" + b"\x00" * 500, Compression.CPIO),
        (b"0143561" + b"\x00" * 500, Compression.CPIO),
    ],
)
def test_detect_magic(magic: bytes, expected: Compression) -> None:
    assert detect(magic + b"trailing payload") is expected


def test_detect_empty() -> None:
    assert detect(b"") is Compression.EMPTY


def test_detect_rejects_unknown() -> None:
    with pytest.raises(UnsupportedCompressionError):
        detect(os.urandom(512))


def test_extensions() -> None:
    assert extension(Compression.GZIP) == ".gz"
    assert extension(Compression.BZIP2) == ".bz2"
    assert extension(Compression.XZ) == ".xz"
    assert extension(Compression.LZMA) == ".lzma"
    assert extension(Compression.LZ4) == ".lz4"
    assert extension(Compression.LZ4_LEGACY) == ".lz4"
    assert extension(Compression.LZOP) == ".lzo"
    assert extension(Compression.CPIO) == ""
    assert extension(Compression.EMPTY) == ".empty"


def test_empty_is_a_zero_length_file() -> None:
    # repackimg.sh touches an empty .cpio.empty file, so EMPTY means "no
    # ramdisk" in both directions rather than a passthrough.
    assert compress(b"", Compression.EMPTY) == b""
    assert compress(b"abc", Compression.EMPTY) == b""
    assert decompress(b"", Compression.EMPTY) == b""
    assert detect(compress(b"abc", Compression.EMPTY)) is Compression.EMPTY
    assert decompress(PAYLOAD, Compression.CPIO) == PAYLOAD
    assert compress(PAYLOAD, Compression.CPIO) == PAYLOAD


def test_lz4_frame_versions_are_distinguished() -> None:
    assert detect(compress(PAYLOAD, Compression.LZ4)) is Compression.LZ4
    assert detect(compress(PAYLOAD, Compression.LZ4_LEGACY)) is Compression.LZ4_LEGACY


def test_lz4_legacy_uses_legacy_magic_only() -> None:
    blob = compress(PAYLOAD, Compression.LZ4_LEGACY)
    assert blob[:4] == struct.pack("<I", 0x184C2102)
    # No modern frame magic may appear where the frame starts.
    assert blob[:4] != struct.pack("<I", 0x184D2204)


def test_xz_defaults_to_level_one() -> None:
    assert compress(PAYLOAD, Compression.XZ) == compress(PAYLOAD, Compression.XZ, 1)


def test_lz4_defaults_to_level_nine() -> None:
    assert compress(PAYLOAD, Compression.LZ4) == compress(PAYLOAD, Compression.LZ4, 9)
    assert compress(PAYLOAD, Compression.LZ4_LEGACY) == compress(
        PAYLOAD, Compression.LZ4_LEGACY, 9
    )


def test_explicit_level_overrides_default() -> None:
    assert compress(PAYLOAD, Compression.XZ, 9) != compress(PAYLOAD, Compression.XZ, 1)


def test_xz_uses_crc32_check() -> None:
    blob = compress(PAYLOAD, Compression.XZ)
    # Stream header flags live at offset 7; the low nibble is the check id,
    # where 0x01 is CRC32 and 0x04 is the format default CRC64.
    assert blob[7] & 0x0F == lzma.CHECK_CRC32
    assert blob[7] & 0x0F != lzma.CHECK_CRC64


def test_xz_decompress_accepts_any_check() -> None:
    crc64 = lzma.compress(PAYLOAD, format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64)
    assert detect(crc64) is Compression.XZ
    assert decompress(crc64, Compression.XZ) == PAYLOAD


def test_lzma_uses_alone_format() -> None:
    blob = compress(PAYLOAD, Compression.LZMA)
    assert blob[:1] == b"\x5d"
    with pytest.raises(lzma.LZMAError):
        lzma.decompress(blob, format=lzma.FORMAT_XZ)


@LZOP
def test_lzop_round_trip() -> None:
    blob = compress(PAYLOAD, Compression.LZOP)
    assert detect(blob) is Compression.LZOP
    assert decompress(blob, Compression.LZOP) == PAYLOAD


@LZOP
def test_lzop_accepts_level() -> None:
    for level in (1, 5, 9):
        blob = compress(PAYLOAD, Compression.LZOP, level)
        assert detect(blob) is Compression.LZOP
        assert decompress(blob, Compression.LZOP) == PAYLOAD


@LZOP
def test_lzop_output_matches_reference_binary() -> None:
    blob = compress(PAYLOAD, Compression.LZOP)
    reference = subprocess.run(
        [bin.require("lzop"), "-c", "-f", "-"],
        input=PAYLOAD,
        capture_output=True,
        check=True,
    )
    assert blob == reference.stdout


def test_lzop_missing_tool_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str):
        raise bin.MissingToolError(f"Required tool {name!r} was not found.")

    monkeypatch.setattr(bin, "tool_path", missing)
    with pytest.raises(MissingToolError, match="apt install lzop"):
        decompress(b"\x89LZO\x00\r\n\x1a\n", Compression.LZOP)


@pytest.mark.parametrize(
    ("comp", "blob"),
    [
        (Compression.GZIP, b"\x1f\x8b" + b"garbage" * 4),
        (Compression.BZIP2, b"BZh9" + b"garbage" * 4),
        (Compression.XZ, b"\xfd7zXZ\x00" + b"garbage" * 4),
        (Compression.LZMA, b"\x5d\x00\x00\x80\x00" + b"garbage" * 4),
        (Compression.LZ4, struct.pack("<I", 0x184D2204) + b"garbage" * 4),
        (Compression.LZ4_LEGACY, struct.pack("<I", 0x184C2102) + b"\xff" * 32),
    ],
)
def test_corrupt_payload_raises_corrupt_image_error(
    comp: Compression, blob: bytes
) -> None:
    # Every stdlib backend must be normalised to one exception type, so callers
    # only have to catch CompressionError.
    with pytest.raises(CorruptImageError):
        decompress(blob, comp)


def test_lz4_legacy_truncation_is_rejected() -> None:
    payload = b"ramdisk " * 2_000_000  # spans more than one 8 MiB legacy block
    blob = compress(payload, Compression.LZ4_LEGACY)
    with pytest.raises(CorruptImageError, match="truncated"):
        decompress(blob[: len(blob) // 2], Compression.LZ4_LEGACY)


def test_lz4_legacy_multi_block_round_trip() -> None:
    payload = b"ramdisk " * 2_000_000
    assert len(
        decompress(compress(payload, Compression.LZ4_LEGACY), Compression.LZ4_LEGACY)
    ) == len(payload)


def test_negative_levels_are_accepted() -> None:
    # repackimg.sh talks in -1..-9, so callers plumbing that through must not
    # hit OverflowError.
    assert compress(PAYLOAD, Compression.XZ, -1) == compress(PAYLOAD, Compression.XZ, 1)
    assert compress(PAYLOAD, Compression.LZMA, -9) == compress(
        PAYLOAD, Compression.LZMA, 9
    )
    assert decompress(compress(PAYLOAD, Compression.XZ, -1), Compression.XZ) == PAYLOAD


def test_lz4_v13_magic_is_not_claimed_as_modern() -> None:
    # The v1.0-v1.3 frame is a distinct wire format that lz4.frame cannot read,
    # so it must not be detected as a format we can then fail to decompress.
    with pytest.raises(UnsupportedCompressionError):
        detect(struct.pack("<I", 0x184C2103) + b"\x00" * 32)


def test_bin_resolver_reports_missing_tool() -> None:
    with pytest.raises(bin.MissingToolError):
        bin.tool_path("definitely-not-a-real-aik-tool")


def test_bin_resolver_prefers_bundled_binary() -> None:
    lz4_path = bin.tool_path("lz4")
    assert lz4_path.is_file()
    assert bin.require("lz4") == str(lz4_path)
    assert os.access(bin.require("lz4"), os.X_OK)


def test_bin_lzop_skips_bad_override_and_uses_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIK_LZOP", "/nonexistent/lzop")
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/lzop" if name == "lzop" else None
    )
    assert bin.tool_path("lzop") == Path("/usr/bin/lzop")


def test_bin_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake = tmp_path / "lzop"
    fake.write_text("")
    monkeypatch.setenv("AIK_LZOP", str(fake))
    assert bin.tool_path("lzop") == fake


def test_bin_raises_when_nothing_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIK_LZOP", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(bin.MissingToolError, match="lzop"):
        bin.tool_path("lzop")
