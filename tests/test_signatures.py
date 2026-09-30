"""Tests for the OEM head-signature wrappers.

Tests that need a real device image are skipped rather than faked: a passing
test that never exercises the tool is worse than an honest skip.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from aik_py.patches import signatures
from aik_py.patches.signatures import (
    BLOB_HEADER,
    DHTB_HEADER_SIZE,
    NOOK_BOOTLOADER_SIZE,
    NOOKTAB_BOOTLOADER_SIZE,
    MissingToolError,
    SignatureError,
    SignatureType,
    StripResult,
)


def _tool_available(name: str) -> bool:
    """Ask the library's own resolver so the guards cannot drift from the code."""
    try:
        signatures._tool(name)
    except (MissingToolError, SignatureError):
        return False
    return True


def _tool_path(name: str) -> Path:
    return signatures._tool(name)


requires_blobpack = pytest.mark.skipif(
    not _tool_available("blobpack"), reason="bundled blobpack/blobunpack not present"
)
requires_futility = pytest.mark.skipif(
    not _tool_available("futility"), reason="bundled futility not present"
)
requires_sony_dump = pytest.mark.skipif(
    not _tool_available("sony_dump"), reason="bundled sony_dump not present"
)


# --------------------------------------------------------------------------
# Nook / NookTAB
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [NOOK_BOOTLOADER_SIZE])
def test_nook_roundtrip_at_boundary(size: int) -> None:
    payload = os.urandom(size * 2)
    result = signatures.strip(payload, SignatureType.NOOK)
    assert result.image == payload[size:]
    assert result.sidecar == payload[:size]
    assert signatures.apply(result.image, result) == payload


@pytest.mark.parametrize("size", [NOOK_BOOTLOADER_SIZE])
def test_nook_roundtrip_below_and_above_boundary(size: int) -> None:
    for total in (size, size + 1, size + 4096):
        payload = os.urandom(total)
        result = signatures.strip(payload, SignatureType.NOOK)
        assert len(result.image) == total - size
        assert signatures.apply(result.image, result) == payload


def test_nooktab_roundtrip_at_boundary() -> None:
    size = NOOKTAB_BOOTLOADER_SIZE
    payload = os.urandom(size + 1024)
    result = signatures.strip(payload, SignatureType.NOOKTAB)
    assert result.image == payload[size:]
    assert result.sidecar == payload[:size]
    assert signatures.apply(result.image, result) == payload


def test_nooktab_roundtrip_below_and_above_boundary() -> None:
    size = NOOKTAB_BOOTLOADER_SIZE
    for total in (size, size + 1, size + 512):
        payload = os.urandom(total)
        result = signatures.strip(payload, SignatureType.NOOKTAB)
        assert signatures.apply(result.image, result) == payload


def test_nook_sidecar_recovered_exactly() -> None:
    """The stashed bootloader must be byte-exact, not merely the right length."""
    payload = os.urandom(NOOK_BOOTLOADER_SIZE + 64)
    result = signatures.strip(payload, "NOOK")
    assert result.sidecar is not None
    assert len(result.sidecar) == NOOK_BOOTLOADER_SIZE
    assert result.sidecar == payload[:NOOK_BOOTLOADER_SIZE]
    assert result.exact_inverse is True


def test_nooktab_and_nook_use_different_boundaries() -> None:
    payload = os.urandom(NOOK_BOOTLOADER_SIZE)
    assert len(signatures.strip(payload, SignatureType.NOOK).image) == 0
    assert len(signatures.strip(payload, SignatureType.NOOKTAB).image) == (
        NOOK_BOOTLOADER_SIZE - NOOKTAB_BOOTLOADER_SIZE
    )


def test_nook_below_minimum_raises() -> None:
    payload = os.urandom(NOOK_BOOTLOADER_SIZE - 1)
    with pytest.raises(SignatureError, match="bootloader"):
        signatures.strip(payload, SignatureType.NOOK)


def test_nooktab_below_minimum_raises() -> None:
    payload = os.urandom(NOOKTAB_BOOTLOADER_SIZE - 1)
    with pytest.raises(SignatureError, match="bootloader"):
        signatures.strip(payload, SignatureType.NOOKTAB)


def test_nook_apply_without_sidecar_raises() -> None:
    bare = StripResult(sigtype=SignatureType.NOOK, image=b"x", sidecar=None)
    with pytest.raises(SignatureError, match="stashed bootloader"):
        signatures.apply(b"x", bare)


# --------------------------------------------------------------------------
# DHTB
# --------------------------------------------------------------------------


def test_dhtb_strips_exactly_512_bytes() -> None:
    payload = os.urandom(4096)
    result = signatures.strip(payload, SignatureType.DHTB)
    # The Bash writes `dd bs=4096 skip=512 iflag=skip_bytes`; `iflag=skip_bytes`
    # makes skip count bytes, so this is 512 bytes and NOT 2 MiB.
    assert DHTB_HEADER_SIZE == 512
    assert result.image == payload[512:]
    assert len(result.image) == 4096 - 512


def test_dhtb_does_not_strip_two_mibibytes() -> None:
    """Regression guard for the skip_bytes/bs misreading."""
    payload = os.urandom(3 * 1024 * 1024)
    result = signatures.strip(payload, SignatureType.DHTB)
    assert len(result.image) == len(payload) - 512
    assert len(result.image) != len(payload) - 2 * 1024 * 1024


def test_dhtb_minimum_size_raises() -> None:
    with pytest.raises(SignatureError, match="512-byte header"):
        signatures.strip(os.urandom(DHTB_HEADER_SIZE), SignatureType.DHTB)
    # Exactly 512 bytes is a header and nothing else; 511 is not even a header.
    with pytest.raises(SignatureError, match="512-byte header"):
        signatures.strip(os.urandom(DHTB_HEADER_SIZE - 1), SignatureType.DHTB)


def test_dhtb_is_not_an_exact_inverse() -> None:
    """apply() re-signs via dhtbsign, so the header bytes are regenerated."""
    result = signatures.strip(os.urandom(4096), SignatureType.DHTB)
    assert result.exact_inverse is False, (
        "DHTB strip is a byte slice but apply shells out to dhtbsign, "
        "which rebuilds the 512-byte header; claiming an exact inverse would "
        "make callers trust a byte comparison that cannot hold"
    )


def test_only_nook_variants_are_exact_inverses() -> None:
    sizes = {
        SignatureType.NOOK: NOOK_BOOTLOADER_SIZE,
        SignatureType.NOOKTAB: NOOKTAB_BOOTLOADER_SIZE,
    }
    for st, size in sizes.items():
        result = signatures.strip(os.urandom(size + 32), st)
        assert result.exact_inverse is True
        assert signatures.apply(result.image, result) is not None


# --------------------------------------------------------------------------
# BLOB
# --------------------------------------------------------------------------


def test_blob_header_is_28_bytes() -> None:
    # repackimg.sh emits `-SIGNED-BY-SIGNBLOB-` (20 bytes) + eight \00 = 28.
    assert BLOB_HEADER == b"-SIGNED-BY-SIGNBLOB-" + bytes(8)
    assert len(BLOB_HEADER) == 28
    assert BLOB_HEADER[:20] == b"-SIGNED-BY-SIGNBLOB-"
    assert BLOB_HEADER[20:] == bytes(8)


def test_blob_name_parsing_matches_bash_pipeline() -> None:
    """Mirrors `blobunpack | tail -n+5 | cut -d" " -f2 | dd bs=1 count=3`."""
    stdout = (
        "Header size: 60\n"
        "1 partitions starting at offset 0x3C\n"
        "Blob version: 65536\n"
        "Partition 0\n"
        "Name: boo\n"
        "Offset: 76 (0x4C)\n"
        "Size: 4096 (0x1000)\n"
        "Writing file file.boo (4096 bytes)\n"
    )
    assert signatures._parse_blob_name(stdout) == "boo"


def test_blob_name_truncated_to_three_bytes() -> None:
    stdout = "a\nb\nc\nPartition 0\nName: recoveryish\nx"
    assert signatures._parse_blob_name(stdout) == "rec"


def test_blob_parse_failure_raises() -> None:
    with pytest.raises(SignatureError, match="partition name"):
        signatures._parse_blob_name("only one line\n")


def test_blob_name_never_falls_back_to_arbitrary_text() -> None:
    """Without a `Name:` line the answer must be an error, not `"Off"`.

    A guessed partition name produces a blob the bootloader will reject, and
    the corruption would surface only at flash time.
    """
    stdout = "Header size: 60\n1 partitions\nBlob version: 1\nPartition 0\nOffset: 76\n"
    with pytest.raises(SignatureError, match="partition name"):
        signatures._parse_blob_name(stdout)


def test_blob_apply_without_name_raises() -> None:
    bare = StripResult(sigtype=SignatureType.BLOB, image=b"img", sidecar=None)
    with pytest.raises(SignatureError, match="partition name"):
        signatures.apply(b"img", bare)


@requires_blobpack
def test_blob_real_roundtrip_preserves_payload_and_name() -> None:
    """Real blobpack -> blobunpack -> blobpack round trip."""
    payload = os.urandom(4096)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "in.bin").write_bytes(payload)
        subprocess.run(
            [str(_tool_path("blobpack")), "out.blob", "boo", "in.bin"],
            cwd=work,
            check=True,
            capture_output=True,
        )
        blob = (work / "out.blob").read_bytes()

    result = signatures.strip(blob, SignatureType.BLOB)
    assert result.blob_name == "boo"
    assert result.image == payload

    rebuilt = signatures.apply(result.image, result)
    assert rebuilt[: len(BLOB_HEADER)] == BLOB_HEADER
    # blobpack is deterministic, so the packed body is byte-identical.
    assert rebuilt[len(BLOB_HEADER) :] == blob


@requires_blobpack
def test_blob_multi_partition_does_not_lose_partitions() -> None:
    """A multi-partition blob must keep every partition, not just the last."""
    boot, recovery = os.urandom(1000), os.urandom(1200)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "a.bin").write_bytes(boot)
        (work / "b.bin").write_bytes(recovery)
        subprocess.run(
            [str(_tool_path("blobpack")), "multi.blob", "boo", "a.bin", "rec", "b.bin"],
            cwd=work,
            check=True,
            capture_output=True,
        )
        blob = (work / "multi.blob").read_bytes()

    result = signatures.strip(blob, SignatureType.BLOB)
    assert result.partitions == {"boo": boot, "rec": recovery}
    assert result.image == boot, "the named partition must win, not the last member"
    assert set(result.members) == {boot, recovery}


@requires_blobpack
def test_blob_argv_construction_matches_tool_usage() -> None:
    """`blobpack <outfile> <partitionname> <partitionfile>` per its usage text."""
    payload = os.urandom(2048)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "image").write_bytes(payload)
        proc = subprocess.run(
            [str(_tool_path("blobpack")), "blob.tmp", "boo", "image"],
            cwd=work,
            check=True,
            capture_output=True,
        )
        assert (work / "blob.tmp").is_file()
        assert b"blob.tmp" in proc.stdout or b"boo" in proc.stdout


# --------------------------------------------------------------------------
# CHROMEOS
# --------------------------------------------------------------------------


@requires_futility
def test_chromeos_apply_argv_matches_documented_usage() -> None:
    """futility vbutil_kernel --pack requires the flags repackimg.sh passes."""
    help_text = subprocess.run(
        [str(_tool_path("futility")), "help", "vbutil_kernel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for flag in ("--pack", "--keyblock", "--signprivate", "--version", "--vmlinuz"):
        assert flag in help_text, f"{flag} missing from futility usage"


def test_chromeos_missing_tool_raises_clear_error() -> None:
    result = StripResult(sigtype=SignatureType.CHROMEOS, image=b"vmlinuz")
    monkey = _missing_tool("futility")
    with monkey, pytest.raises((MissingToolError, SignatureError)):
        signatures.apply(b"vmlinuz", result)


@requires_futility
def test_chromeos_roundtrip_skipped_without_device_fixture() -> None:
    pytest.skip(
        "No real ChromeOS kernel-partition fixture in the repo; "
        "futility rejects synthetic blobs as 'too small to be a valid kernel blob'"
    )


# --------------------------------------------------------------------------
# SONY
# --------------------------------------------------------------------------


def test_sin_is_packaging_not_a_signature() -> None:
    for st in (SignatureType.SINV1, SignatureType.SINV2, SignatureType.SINV3):
        assert st.is_sin is True
        result = StripResult(sigtype=st, image=b"x")
        with pytest.raises(SignatureError, match="packaging"):
            signatures.apply(b"x", result)
    assert SignatureType.NOOK.is_sin is False


def test_sin_missing_tool_raises_clear_error() -> None:
    monkey = _missing_tool("sony_dump")
    with monkey, pytest.raises((MissingToolError, SignatureError)):
        signatures.strip(os.urandom(512), SignatureType.SINV3)


def test_sin_roundtrip_skipped_without_device_fixture() -> None:
    pytest.skip(
        "No real SINv1/v2/v3 fixture in the repo; sony_dump rejects synthetic "
        "containers with 'sin header size or sin data size is 0'"
    )


@requires_sony_dump
def test_sin_member_glob_matches_sony_dump_naming() -> None:
    """sony_dump writes ``<name>-<part>``, so the glob must be hyphen-based.

    The Bash gets away with ``"$file."*`` only because ``$file`` is
    ``boot.img`` and the dot supplies the separator; globbing ``image.img.*``
    finds nothing and would make SIN support dead code.
    """
    import struct

    # A complete-enough bootimg v0: sony_dump aborts (SIGABRT, heap corruption)
    # when the size/address fields are left zero, so populate all of them.
    header = bytearray(1632)
    header[0:8] = b"ANDROID!"
    for offset, value in {
        8: 1632,  # kernel_size
        12: 1024,  # kernel_addr
        16: 2048,  # ramdisk_size
        20: 3072,  # ramdisk_addr
        24: 512,  # second_size
        28: 4096,  # second_addr
        32: 0,  # tags_addr
        36: 0,  # page_size
        40: 2,  # dt_size
    }.items():
        struct.pack_into("<I", header, offset, value)
    header[48:64] = b"kerneloff=00000800"
    image = bytes(header) + bytes(4096 + 2048 + 512)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "image.img").write_bytes(image)
        # The bundled sony_dump writes its members and then aborts in teardown
        # (exit 134), so its exit status is not a usable success signal.
        subprocess.run(
            [str(_tool_path("sony_dump")), ".", "image.img"],
            cwd=work,
            capture_output=True,
            check=False,
        )
        members = sorted(p.name for p in work.glob("image.img-*"))
        assert members, "sony_dump produced no hyphen-named members"
        assert not list(work.glob("image.img.*")), "the dotted glob must not match"

    result = signatures.strip(image, SignatureType.SINV3)
    assert result.members, "strip must return the sony_dump members"
    assert result.is_sin is True


# --------------------------------------------------------------------------
# Generic
# --------------------------------------------------------------------------


def test_unknown_signature_type_raises() -> None:
    with pytest.raises(SignatureError, match="unknown signature type"):
        signatures.strip(os.urandom(64), "NOPE")


def test_empty_data_raises() -> None:
    with pytest.raises(SignatureError, match="empty"):
        signatures.strip(b"", SignatureType.NOOK)


def test_all_signature_types_have_a_stripper() -> None:
    assert set(signatures._STRIPPERS) == set(SignatureType)


class _missing_tool:
    """Context manager forcing `_tool()` to report a missing binary."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> None:
        import aik_py.patches.signatures as mod

        self._mod = mod
        self._orig = mod._tool
        mod._tool = self._raise

    def _raise(self, name: str) -> Path:
        if name == self.name:
            raise MissingToolError(f"bundled tool {name!r} not found (simulated)")
        return self._orig(name)

    def __exit__(self, *exc: object) -> None:
        self._mod._tool = self._orig
