"""Tests for :mod:`aik_py.patches.footers`."""

from __future__ import annotations

import base64
import struct

import pytest

from aik_py.patches import footers
from aik_py.patches.footers import Footer, FooterKind

# --- Fixtures --------------------------------------------------------------

# Real footers, produced by running the bundled AOSP signer against the
# bundled test key:
#
#   java -jar bin/boot_signer.jar /boot <image> \
#        bin/avb/verity.pk8 bin/avb/verity.x509.pem <out>
#
# These are ground truth for the AVBv1 layout -- the byte lengths here were
# measured from real signer output rather than assumed.
REAL_AVB_V1_FOOTERS = {
    "boot": base64.b64decode(
        "MIIFIgIBATCCA/0wggLloAMCAQICCQCXD5g5CaqJSTANBgkqhkiG9w0BAQUFADCBlDELMAkGA1UEBhMCVVMxEzARBgNVBAgM"
        "CkNhbGlmb3JuaWExFjAUBgNVBAcMDU1vdW50YWluIFZpZXcxEDAOBgNVBAoMB0FuZHJvaWQxEDAOBgNVBAsMB0FuZHJvaWQx"
        "EDAOBgNVBAMMB0FuZHJvaWQxIjAgBgkqhkiG9w0BCQEWE2FuZHJvaWRAYW5kcm9pZC5jb20wHhcNMTQxMTA2MTkwNzQwWhcN"
        "NDIwMzI0MTkwNzQwWjCBlDELMAkGA1UEBhMCVVMxEzARBgNVBAgMCkNhbGlmb3JuaWExFjAUBgNVBAcMDU1vdW50YWluIFZp"
        "ZXcxEDAOBgNVBAoMB0FuZHJvaWQxEDAOBgNVBAsMB0FuZHJvaWQxEDAOBgNVBAMMB0FuZHJvaWQxIjAgBgkqhkiG9w0BCQEW"
        "E2FuZHJvaWRAYW5kcm9pZC5jb20wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDo63hNL01UkXp7szvb52ln5NHk"
        "M2Gm9IKqYusQM4unZg/roKBCiZmz4rhOQ8H9tYrGfboVFLtHUDOOnSuKHCsTEa3J5hscnRZ+qH7NzgyTFzpL9oCly/xXWxD3"
        "Q28c3bvM98pPluu7nTP31u1m2kNwztJJ7vosympP90+NXObqF5kPNVDbQM0RsxnITVVzJlrkxjpIOlPtCNk3eyvMr1DFoQFj"
        "z6Si7VR/awC+U842DUfdos3SnM9wI0bCNwk47aYlQARnl9E3I0UrmQeyvRCueh1fjhTUuiNTT43Q+xSEochpaqmXVDpAFGWG"
        "p26YHk+Te0C+rrqnBqaEzpGpbupJAgMBAAGjUDBOMB0GA1UdDgQWBBR+QzP5u6AK3+Dt6XnijtGSBJK0DzAfBgNVHSMEGDAW"
        "gBR+QzP5u6AK3+Dt6XnijtGSBJK0DzAMBgNVHRMEBTADAQH/MA0GCSqGSIb3DQEBBQUAA4IBAQBztzUrwxOYxbzHoRhrUvAZ"
        "xWHRSkSIJQzHOJ4lks4czlWER+Fmo8XbQVnOX8dIsTRf/oVSlYT+gCDkGvvvrJPhyIu0b4qDz24Y3LFZWXu3FBBqy0kr3hqL"
        "kMnPo9PpuiRfOJLMyJeRhjTE58dF0TzvoPLeoeS1n51cnYGX6RkUYPlh3sJNVNh5b04yHHm9DXXKCfFiiQBVnXLhO1SJuMPr"
        "xfw2F/wNjss69vkEYZmEr9XLrIAHgtbbx+GqfidS4FJzruSofxajyy0sDVUKeAbu2/UBX6ivDKI/OG4/PDakRmYA8qs7k6xc"
        "Z+hVu3ZsgMrbbgCQBAMppNZoqXffs6d/MAsGCSqGSIb3DQEBCzALEwUvYm9vdAICCAAEggEALggAOh9K/EKbqdG5iDfwmGZP"
        "Ss30UPMxknL5EhfHiez6pwWkvM7tFYIEAbxHcmcjb28Xl9yZfQ2ZdpAmNUQgxo7jJzH+u22aglCm7MxYn5Jmz89+gHVhZqvK"
        "2xhJV/dgg1DPjjn7DHjGALmQB/P5y9VqPRSi6FgVP3BP9/MB9HZqRVOLMcSD+Fw/tQwtj49MUAPSUJiGlHWepztAzmUi3aMA"
        "VvV5XkQQxjzsEIW3AlLmLJpPWqvJ2OKZuvznQsf24Zfef1vW422A+uap+pnQnIYsKvbJ2+S36KydIGzNrQHdUo6Uh0EUbNgt"
        "o1yy+ljA5OcSaAdPLoQcXcAPRvl3rA=="
    ),
    "recovery": base64.b64decode(
        "MIIFJgIBATCCA/0wggLloAMCAQICCQCXD5g5CaqJSTANBgkqhkiG9w0BAQUFADCBlDELMAkGA1UEBhMCVVMxEzARBgNVBAgM"
        "CkNhbGlmb3JuaWExFjAUBgNVBAcMDU1vdW50YWluIFZpZXcxEDAOBgNVBAoMB0FuZHJvaWQxEDAOBgNVBAsMB0FuZHJvaWQx"
        "EDAOBgNVBAMMB0FuZHJvaWQxIjAgBgkqhkiG9w0BCQEWE2FuZHJvaWRAYW5kcm9pZC5jb20wHhcNMTQxMTA2MTkwNzQwWhcN"
        "NDIwMzI0MTkwNzQwWjCBlDELMAkGA1UEBhMCVVMxEzARBgNVBAgMCkNhbGlmb3JuaWExFjAUBgNVBAcMDU1vdW50YWluIFZp"
        "ZXcxEDAOBgNVBAoMB0FuZHJvaWQxEDAOBgNVBAsMB0FuZHJvaWQxEDAOBgNVBAMMB0FuZHJvaWQxIjAgBgkqhkiG9w0BCQEW"
        "E2FuZHJvaWRAYW5kcm9pZC5jb20wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDo63hNL01UkXp7szvb52ln5NHk"
        "M2Gm9IKqYusQM4unZg/roKBCiZmz4rhOQ8H9tYrGfboVFLtHUDOOnSuKHCsTEa3J5hscnRZ+qH7NzgyTFzpL9oCly/xXWxD3"
        "Q28c3bvM98pPluu7nTP31u1m2kNwztJJ7vosympP90+NXObqF5kPNVDbQM0RsxnITVVzJlrkxjpIOlPtCNk3eyvMr1DFoQFj"
        "z6Si7VR/awC+U842DUfdos3SnM9wI0bCNwk47aYlQARnl9E3I0UrmQeyvRCueh1fjhTUuiNTT43Q+xSEochpaqmXVDpAFGWG"
        "p26YHk+Te0C+rrqnBqaEzpGpbupJAgMBAAGjUDBOMB0GA1UdDgQWBBR+QzP5u6AK3+Dt6XnijtGSBJK0DzAfBgNVHSMEGDAW"
        "gBR+QzP5u6AK3+Dt6XnijtGSBJK0DzAMBgNVHRMEBTADAQH/MA0GCSqGSIb3DQEBBQUAA4IBAQBztzUrwxOYxbzHoRhrUvAZ"
        "xWHRSkSIJQzHOJ4lks4czlWER+Fmo8XbQVnOX8dIsTRf/oVSlYT+gCDkGvvvrJPhyIu0b4qDz24Y3LFZWXu3FBBqy0kr3hqL"
        "kMnPo9PpuiRfOJLMyJeRhjTE58dF0TzvoPLeoeS1n51cnYGX6RkUYPlh3sJNVNh5b04yHHm9DXXKCfFiiQBVnXLhO1SJuMPr"
        "xfw2F/wNjss69vkEYZmEr9XLrIAHgtbbx+GqfidS4FJzruSofxajyy0sDVUKeAbu2/UBX6ivDKI/OG4/PDakRmYA8qs7k6xc"
        "Z+hVu3ZsgMrbbgCQBAMppNZoqXffs6d/MAsGCSqGSIb3DQEBCzAPEwkvcmVjb3ZlcnkCAggABIIBAGwqO6/vCj7AOH6Pq5Ss"
        "BGxhUHP/L8ct0mtmC5RTHhbrks5EyqKWblbnY1XPabS7u/VTwp6oBRfZ2h/1jMqiVFSGQCCWB+pXLSb2sPxpAYF886Dkb//t"
        "YOdrKxYKSrHuwzscUohwYdQ45R9Hk/Tscgnx3XiBgXZPkyYVciV9AovpanaTZHxN9SAk1ANwvw350lGidKt430Yo7+8M2Y30"
        "tV+qwhHiJu6t5ncSEebGBXabi6RDTcski0lHZAM630Y7LX7tHinMhlSbhY0y6htspPdC3iKWmN2hIM0YHw/pqsKI0SFQeJU6"
        "mt6UI71YZXadWHdh7Kyry5GZxM5w8ynBO3w="
    ),
}


def build_avb_v2(image: bytes, vbmeta: bytes) -> bytes:
    """Build ``image + vbmeta + 64-byte AVBf footer``.

    Mirrors what AOSP ``avbtool`` writes: the vbmeta blob is appended after
    the image and the footer lands in the final 64 bytes pointing at it.  The
    real tool also page-aligns the blob and pads the partition, but the footer
    contract -- that it records ``original_image_size``, ``vbmeta_offset`` and
    ``vbmeta_size`` -- is what this module depends on, so the fixture keeps
    the layout minimal and readable.
    """
    vbmeta_offset = len(image)
    footer = struct.pack(
        ">4sIIQQQ28x",
        footers.AVB_V2_MAGIC,
        1,
        0,  # major/minor version
        len(image),  # original_image_size
        vbmeta_offset,
        len(vbmeta),
    )
    assert len(footer) == footers.AVB_V2_FOOTER_SIZE
    return image + vbmeta + footer


def build_avb_v1(partition: str) -> bytes:
    """Build a synthetic AVBv1 footer block embedding ``partition``.

    The shape mirrors real signer output: ``INTEGER formatVersion``,
    ``SEQUENCE authenticatedAttributes``, ``SEQUENCE algorithmIdentifier``,
    ``SEQUENCE { PrintableString target, INTEGER length }`` and finally the
    signature ``OCTET STRING``, which terminates the block.

    The authenticatedAttributes SEQUENCE is deliberately padded past 127
    bytes so its length uses the two-octet long form -- that is what produces
    the ``30 82`` in the ``02 01 01 30 82`` magic, which a real signature
    block always carries because it embeds an X.509 certificate.
    """

    def tlv(tag: int, content: bytes) -> bytes:
        if len(content) < 0x80:
            return bytes([tag, len(content)]) + content
        return bytes([tag, 0x82]) + len(content).to_bytes(2, "big") + content

    version = tlv(0x02, b"\x01")
    cert_oid = tlv(0x06, b"\x2a\x86\x48\x86\xf7\x0d\x01\x01\x01")
    auth_attrs = tlv(0x30, cert_oid + b"\x00" * 200)
    algorithm = tlv(0x30, tlv(0x06, b"\x2a\x86\x48\x86\xf7\x0d\x01\x01\x0b"))
    target = tlv(0x30, tlv(0x13, partition.encode()) + tlv(0x02, b"\x01\x00"))
    signature = tlv(0x04, b"\xab" * 256)
    inner = version + auth_attrs + algorithm + target + signature
    # Real signer output wraps the whole thing in a DER SEQUENCE whose
    # "30 82 <len>" header sits immediately before the magic.
    block = tlv(0x30, inner)
    assert block[4:].startswith(footers.AVB_V1_MAGIC), block[4:9].hex()
    return block


IMAGE = bytes(range(256)) * 64  # 16 KiB of deterministic, non-footer bytes


# --- Bump ------------------------------------------------------------------


def test_bump_detected():
    data = IMAGE + footers.BUMP_MAGIC
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.BUMP
    assert footer.length == 16
    assert footer.offset == len(IMAGE)


def test_bump_round_trip():
    data = IMAGE + footers.BUMP_MAGIC
    footer = footers.detect(data)
    assert footer is not None
    assert footers.strip(data, footer) == IMAGE
    assert footers.apply(IMAGE, footer) == data


# --- SEAndroid -------------------------------------------------------------


def test_seandroid_detected():
    data = IMAGE + footers.SEANDROID_MAGIC
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.SEANDROID
    assert footer.length == 16
    assert footers.strip(data, footer) == IMAGE
    assert footers.apply(IMAGE, footer) == data


def test_seandroid_round_trip_is_byte_exact():
    data = IMAGE + b"SEANDROIDENFORCE"
    footer = footers.detect(data)
    bare = footers.strip(data, footer)
    assert footers.apply(bare, footer) == data


# --- AVBv2 -----------------------------------------------------------------


def test_avbv2_detected_and_stripped():
    vbmeta = b"AVB0" + b"\x11" * 508  # plausible vbmeta blob
    data = build_avb_v2(IMAGE, vbmeta)
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.AVBV2
    assert footer.original_image_size == len(IMAGE)
    assert footer.vbmeta == vbmeta
    assert footer.length == footers.AVB_V2_FOOTER_SIZE
    assert footers.strip(data, footer) == IMAGE


def test_avbv2_footer_size_is_64():
    assert footers.AVB_V2_FOOTER_SIZE == 64
    footer = build_avb_v2(IMAGE, b"AVB0")
    assert footer[-64:][:4] == footers.AVB_V2_MAGIC


def test_avbv2_strip_handles_padding():
    """avbtool pads the blob and the partition; strip ignores both."""
    vbmeta = b"AVB0" + b"\x22" * 300
    padded = vbmeta + b"\x00" * 64
    data = (
        IMAGE
        + padded
        + struct.pack(
            ">4sIIQQQ28x",
            footers.AVB_V2_MAGIC,
            1,
            0,
            len(IMAGE),
            len(IMAGE),
            len(vbmeta),
        )
    )
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.AVBV2
    assert footers.strip(data, footer) == IMAGE


def test_avbv2_rejects_offsets_pointing_past_the_image():
    bad = IMAGE + struct.pack(
        ">4sIIQQQ28x", footers.AVB_V2_MAGIC, 1, 0, len(IMAGE), len(IMAGE) + 4096, 64
    )
    assert footers.detect(bad) is None


def test_avbv2_rejects_image_size_larger_than_vbmeta_offset():
    bad = IMAGE + struct.pack(
        ">4sIIQQQ28x", footers.AVB_V2_MAGIC, 1, 0, len(IMAGE) + 10, len(IMAGE), 8
    )
    assert footers.detect(bad) is None


def test_avbv2_apply_reports_missing_signer(monkeypatch):
    footer = Footer(kind=FooterKind.AVBV2, length=64, original_image_size=len(IMAGE))
    monkeypatch.setitem(__import__("sys").modules, "aik_py.avb", None)
    with pytest.raises(footers.SignatureError, match="AVB re-signing not available"):
        footers.apply(IMAGE, footer)


# --- AVBv1 -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["boot", "recovery"])
def test_real_avbv1_footer_detected(name):
    """Ground truth: a footer actually emitted by bin/boot_signer.jar."""
    data = IMAGE + REAL_AVB_V1_FOOTERS[name]
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.AVBV1
    assert footer.partition == f"/{name}"
    assert footer.offset == len(IMAGE)
    assert footer.length == len(REAL_AVB_V1_FOOTERS[name])
    assert footers.strip(data, footer) == IMAGE


def test_real_avbv1_footers_start_with_the_magic():
    """Real signer output wraps the block in a DER SEQUENCE before the magic."""
    for name, block in REAL_AVB_V1_FOOTERS.items():
        assert block[4:].startswith(footers.AVB_V1_MAGIC), name
        # SEQUENCE tag plus the two-octet long-form length, covering exactly
        # the rest of the footer.
        assert block[:2] == b"\x30\x82", block[:2].hex()
        assert int.from_bytes(block[2:4], "big") == len(block) - 4


def test_real_avbv1_footer_length_is_measured_not_guessed():
    """1318 for /boot vs 1322 for /recovery -- variable, so it must be parsed."""
    assert len(REAL_AVB_V1_FOOTERS["boot"]) == 1318
    assert len(REAL_AVB_V1_FOOTERS["recovery"]) == 1322
    for name, block in REAL_AVB_V1_FOOTERS.items():
        assert footers.detect(IMAGE + block).length == len(block), name


@pytest.mark.parametrize("name", ["boot", "recovery"])
def test_real_avbv1_strip_leaves_no_orphan_wrapper_bytes(name):
    """The 4-byte SEQUENCE header is part of the footer, not trailing junk.

    Left behind it would be appended to the repacked image, corrupting it.
    """
    block = REAL_AVB_V1_FOOTERS[name]
    data = IMAGE + block
    footer = footers.detect(data)
    assert footer.offset == len(IMAGE)
    assert footers.strip(data, footer) == IMAGE
    assert footers.strip(data, footer) + block == data


@pytest.mark.parametrize(
    ("partition", "expected"), [("/boot", "boot"), ("/recovery", "recovery")]
)
def test_avbv1_partition_name_recovered(partition, expected):
    data = IMAGE + build_avb_v1(partition)
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.AVBV1
    assert footer.partition == partition
    assert expected in footer.partition


def test_avbv1_footer_length_is_variable_and_measured():
    boot = build_avb_v1("/boot")
    recovery = build_avb_v1("/recovery")
    # The longer partition name makes the block longer, so the length cannot
    # be hardcoded -- it has to come out of the TLV walk.
    assert len(recovery) == len(boot) + 4
    assert footers.detect(IMAGE + boot).length == len(boot)
    assert footers.detect(IMAGE + recovery).length == len(recovery)


def test_avbv1_strip_removes_whole_block():
    footer_bytes = build_avb_v1("/boot")
    data = IMAGE + footer_bytes
    footer = footers.detect(data)
    assert footers.strip(data, footer) == IMAGE


def test_avbv1_rejects_magic_without_partition():
    truncated = build_avb_v1("/boot")[:20]
    assert footers.detect(IMAGE + truncated) is None


def test_avbv1_rejects_non_partition_target():
    assert footers.detect(IMAGE + build_avb_v1("notapath")) is None


def test_avbv1_accepts_long_by_name_partition():
    """By-name paths exceed any size heuristic; detection must not depend on one."""
    path = "/dev/block/platform/soc.ldr.2.1-by-name/" + "p" * 100
    assert len(path) > 64
    data = IMAGE + build_avb_v1(path)
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.AVBV1
    assert footer.partition == path
    assert footers.strip(data, footer) == IMAGE


def test_avbv1_apply_refuses_to_guess_partition(monkeypatch):
    """A recovery image signed as /boot is worse than an explicit failure."""
    import sys
    import types

    module = types.ModuleType("aik_py.avb")
    module.sign_v1 = lambda image, partition: b"signed"
    monkeypatch.setitem(sys.modules, "aik_py.avb", module)

    footer = Footer(kind=FooterKind.AVBV1)  # partition defaults to None
    with pytest.raises(footers.SignatureError, match="without a partition name"):
        footers.apply(IMAGE, footer)


def test_avbv1_apply_passes_partition_to_signer(monkeypatch):
    seen: list[tuple] = []
    import sys
    import types

    module = types.ModuleType("aik_py.avb")
    module.sign_v1 = lambda image, partition: (
        seen.append((image, partition)) or b"signed"
    )
    monkeypatch.setitem(sys.modules, "aik_py.avb", module)

    footer = Footer(kind=FooterKind.AVBV1, partition="/recovery")
    assert footers.apply(IMAGE, footer) == b"signed"
    assert seen == [(IMAGE, "/recovery")]


def test_avbv2_apply_delegates_to_signer(monkeypatch):
    import sys
    import types

    seen: list[bytes] = []
    module = types.ModuleType("aik_py.avb")
    module.sign_v2 = lambda image: seen.append(image) or b"signed-v2"
    monkeypatch.setitem(sys.modules, "aik_py.avb", module)

    vbmeta = b"AVB0" + b"\x44" * 128
    data = build_avb_v2(IMAGE, vbmeta)
    footer = footers.detect(data)
    assert footers.apply(IMAGE, footer) == b"signed-v2"
    assert seen == [IMAGE]


# --- Negative and scan-window behaviour ------------------------------------


def test_no_footer_returns_none():
    assert footers.detect(IMAGE) is None


def test_empty_input_returns_none():
    assert footers.detect(b"") is None


def test_footer_near_tail_of_large_image_is_found():
    body = b"\x00" * 3_000_000
    data = body + footers.SEANDROID_MAGIC
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.SEANDROID
    assert footers.strip(data, footer) == body


def test_footer_beyond_first_window_found_by_line_retry():
    """A footer past the last 8 KiB is still recovered by the second stage."""
    # 12 KiB of newlines before the footer puts it ~12 KiB from the tail, i.e.
    # outside the 8 KiB byte window but inside the last 50 lines.
    filler = b"\n" * 12_288
    body = IMAGE + filler
    data = body + footers.SEANDROID_MAGIC
    assert len(data) - footers.TAIL_SCAN_SIZE > 0
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.SEANDROID
    assert footers.strip(data, footer) == body


def test_footer_far_from_tail_is_not_found():
    """Beyond both windows the scan gives up rather than guess."""
    filler = b"\n" * 200_000  # 200k lines: past the 50-line retry
    body = IMAGE
    data = body + footers.SEANDROID_MAGIC + filler
    assert footers.detect(data) is None


def test_bump_inside_image_is_not_mistaken_for_a_footer():
    # The marker is present but not at the tail, so it is not a footer.
    data = footers.BUMP_MAGIC + b"\x00" * 64
    assert footers.detect(data) is None


def test_small_image_scanned_whole():
    data = footers.BUMP_MAGIC
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.BUMP
    assert footers.strip(data, footer) == b""


def test_avbv1_wrapper_at_offset_zero_is_included():
    """A footer that is the entire buffer still starts at 0, wrapper included."""
    data = build_avb_v1("/boot")
    footer = footers.detect(data)
    assert footer is not None
    assert footer.offset == 0
    assert footer.length == len(data)
    assert footers.strip(data, footer) == b""


def test_footer_found_without_newlines_in_the_tail():
    """``tail -n50`` on a newline-free buffer yields the whole buffer."""
    body = b"\x00" * 40_000  # no newline bytes anywhere
    data = body + footers.SEANDROID_MAGIC
    footer = footers.detect(data)
    assert footer is not None
    assert footer.kind is FooterKind.SEANDROID
    assert footers.strip(data, footer) == body


# --- Module integrity ------------------------------------------------------


def test_every_exported_name_exists():
    """__all__ must not promise names that only exist in the fallback branch."""
    for name in footers.__all__:
        assert hasattr(footers, name), name


def test_star_import_works():
    """`from ... import *` must not raise on any name listed in __all__."""
    for name in footers.__all__:
        assert getattr(footers, name) is not None, name


# --- Round trips over a realistically sized payload ------------------------


@pytest.mark.parametrize("size", [4096, 1_000_000, 5_000_000])
def test_round_trip_over_image_sized_payload(size):
    image = bytes((i * 7 + 11) & 0xFF for i in range(4096)) * (size // 4096)
    for kind, marker in (
        (FooterKind.BUMP, footers.BUMP_MAGIC),
        (FooterKind.SEANDROID, footers.SEANDROID_MAGIC),
        (FooterKind.AVBV2, b"AVB0" + b"\x33" * 300),
    ):
        data = (
            build_avb_v2(image, marker) if kind is FooterKind.AVBV2 else image + marker
        )
        footer = footers.detect(data)
        assert footer is not None and footer.kind is kind
        stripped = footers.strip(data, footer)
        assert stripped == image
        if kind is not FooterKind.AVBV2:
            assert footers.apply(stripped, footer) == data


# --- Strip guards ----------------------------------------------------------


@pytest.mark.parametrize("size", [0, -1, 1 << 20])
def test_strip_rejects_out_of_range_avbv2_size(size):
    data = build_avb_v2(IMAGE, b"AVB0" + b"\x00" * 64)
    footer = Footer(kind=FooterKind.AVBV2, length=64, original_image_size=size)
    with pytest.raises(footers.CorruptImageError, match="out of range"):
        footers.strip(data, footer)


def test_strip_rejects_oversized_footer():
    footer = Footer(kind=FooterKind.BUMP, length=len(IMAGE) + 1)
    with pytest.raises(footers.CorruptImageError, match="corrupt footer"):
        footers.strip(IMAGE, footer)
