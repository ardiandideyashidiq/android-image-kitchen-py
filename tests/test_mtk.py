"""Tests for the MediaTek 512-byte header layer."""

import struct
import subprocess

import pytest

from aik_py.patches import mtk
from aik_py.patches.mtk import (
    HEADER_SIZE,
    MAGIC,
    MtkHeaderError,
    MtkHeaders,
    MtkToolError,
    MtkType,
    apply,
    apply_subprocess,
    build,
    detect_type,
    has_header,
    payload_size,
    record_type,
    resolve_mkmtkhdr,
    strip,
)

MKMTKHDR = resolve_mkmtkhdr()
needs_binary = pytest.mark.skipif(
    MKMTKHDR is None, reason="bundled mkmtkhdr not available"
)


def retag(blob: bytes, tag: str) -> bytes:
    """Overwrite the type tag in an already-built header."""
    raw = bytearray(blob)
    raw[8:40] = tag.encode("ascii").ljust(32, b"\x00")
    return bytes(raw)


class TestDetection:
    def test_has_header_positive(self):
        assert has_header(build(b"payload", MtkType.KERNEL))
        assert has_header(build(b"x" * 4, MtkType.ROOTFS))

    def test_has_header_negative(self):
        assert not has_header(b"")
        assert not has_header(b"not an mtk image at all")
        assert not has_header(MAGIC[::-1])

    def test_has_header_needs_a_full_header(self):
        # A truncated extract must not be classified as headered, or the
        # matching strip() would reject it and lose the component.
        assert not has_header(MAGIC)
        assert not has_header(MAGIC + b"KERNEL" + bytes(100))
        assert not has_header(build(b"payload", MtkType.KERNEL)[: HEADER_SIZE - 1])

    def test_detect_type_kernel(self):
        assert detect_type(build(b"K" * 64, MtkType.KERNEL)) is MtkType.KERNEL

    def test_detect_type_rootfs(self):
        assert detect_type(build(b"R" * 64, MtkType.ROOTFS)) is MtkType.ROOTFS

    def test_detect_type_recovery(self):
        assert detect_type(build(b"D" * 64, MtkType.RECOVERY)) is MtkType.RECOVERY

    def test_detect_type_none_without_magic(self):
        assert detect_type(b"NOHEADER" * 64) is None

    def test_detect_type_none_for_unknown_tag(self):
        # Magic present but the tag is something AIK has no name for: still a
        # header, but the type is undetermined.
        raw = retag(build(b"x" * 16, MtkType.ROOTFS), "SOMETHING ELSE")
        assert has_header(raw)
        assert detect_type(raw) is None

    def test_detect_type_none_for_short_buffer(self):
        assert detect_type(b"") is None
        assert detect_type(MAGIC) is None
        assert detect_type(MAGIC + b"KERNEL" + bytes(100)) is None

    def test_detect_type_ignores_non_ascii_tag(self):
        raw = bytearray(build(b"x" * 16, MtkType.ROOTFS))
        raw[8:40] = b"\xff\xfe\xfd\xfc".ljust(32, b"\x00")
        assert detect_type(bytes(raw)) is None
        assert has_header(bytes(raw))

    def test_payload_size(self):
        assert payload_size(build(b"z" * 1234, MtkType.KERNEL)) == 1234
        assert payload_size(b"no header here") is None


class TestStrip:
    def test_strip_removes_exactly_512_bytes(self):
        payload = bytes(range(256)) * 4
        blob = build(payload, MtkType.KERNEL)
        assert len(blob) == HEADER_SIZE + len(payload)
        assert strip(blob) == payload
        assert len(strip(blob)) == len(blob) - 512

    def test_strip_leaves_payload_byte_identical(self):
        for size in (4, 5, 511, 512, 513, 65536, 100000):
            payload = bytes((i * 31 + 7) & 0xFF for i in range(size))
            assert strip(build(payload, MtkType.ROOTFS)) == payload, size

    def test_strip_is_lossless_for_both_components(self):
        kernel = b"K" * 4096
        ramdisk = b"R" * 1024
        headers = apply(kernel, ramdisk, MtkType.ROOTFS)
        assert strip(headers.kernel) == kernel
        assert strip(headers.ramdisk) == ramdisk

    @pytest.mark.parametrize("size", [0, 1, 511])
    def test_strip_short_buffer_raises(self, size):
        with pytest.raises(MtkHeaderError, match="too short"):
            strip(bytes(size))

    def test_strip_refuses_a_buffer_without_the_magic(self):
        # Truncating 512 bytes off a genuine kernel would silently corrupt it.
        payload = bytes(range(256)) * 8
        with pytest.raises(MtkHeaderError, match="does not start with the MTK magic"):
            strip(payload)

    def test_strip_refuses_a_truncated_header(self):
        truncated = build(b"payload", MtkType.KERNEL)[: HEADER_SIZE - 1]
        with pytest.raises(MtkHeaderError, match="too short"):
            strip(truncated)


class TestBuild:
    def test_header_fields(self):
        blob = build(b"payload!", MtkType.RECOVERY)
        assert blob[:4] == MAGIC
        assert struct.unpack_from("<I", blob, 4)[0] == 8
        assert blob[8:16] == b"RECOVERY"
        assert blob[16:40] == bytes(24)
        assert blob[40:HEADER_SIZE] == b"\xff" * 472

    def test_header_is_512_bytes_regardless_of_payload(self):
        for size in (4, 512, 100_000):
            assert len(build(bytes(size), MtkType.KERNEL)) == 512 + size

    @pytest.mark.parametrize("size", [0, 1, 2, 3])
    def test_build_rejects_payloads_mkmtkhdr_would_refuse(self, size):
        # The binary errors on these, so byte-compatibility means refusing too.
        with pytest.raises(MtkHeaderError, match="at least"):
            build(bytes(size), MtkType.KERNEL)

    def test_build_refuses_to_double_header(self):
        once = build(b"payload" * 8, MtkType.KERNEL)
        with pytest.raises(MtkHeaderError, match="already carries an MTK header"):
            build(once, MtkType.KERNEL)

    def test_type_values_match_repack_flag_concatenation(self):
        # repackimg.sh builds the flag as "--$mtktype", so the recorded value
        # must be exactly the token the tool accepts.
        assert MtkType.ROOTFS == "rootfs"
        assert MtkType.RECOVERY == "recovery"
        assert MtkType.KERNEL == "kernel"
        assert {t.value for t in MtkType} == {"kernel", "rootfs", "recovery"}


class TestApply:
    def test_apply_tags_kernel_and_ramdisk(self):
        headers = apply(b"K" * 1024, b"R" * 100, MtkType.ROOTFS)
        assert isinstance(headers, MtkHeaders)
        assert detect_type(headers.kernel) is MtkType.KERNEL
        assert detect_type(headers.ramdisk) is MtkType.ROOTFS

    def test_apply_recovery_type(self):
        headers = apply(b"K" * 1024, b"R" * 100, MtkType.RECOVERY)
        assert detect_type(headers.ramdisk) is MtkType.RECOVERY

    def test_apply_round_trips_exactly(self):
        kernel = bytes((i * 13) & 0xFF for i in range(9000))
        ramdisk = bytes((i * 17) & 0xFF for i in range(3333))
        headers = apply(kernel, ramdisk, MtkType.RECOVERY)
        assert strip(headers.kernel) == kernel
        assert strip(headers.ramdisk) == ramdisk

    def test_apply_sizes_record_payload_length(self):
        headers = apply(b"K" * 2048, b"R" * 512, MtkType.ROOTFS)
        assert payload_size(headers.kernel) == 2048
        assert payload_size(headers.ramdisk) == 512

    def test_apply_refuses_already_headered_components(self):
        # mkmtkhdr rejects these outright, so the native path must too rather
        # than silently stacking a second header.
        once = apply(b"K" * 100, b"R" * 100, MtkType.ROOTFS)
        with pytest.raises(MtkHeaderError, match="already carries an MTK header"):
            apply(once.kernel, b"R" * 100, MtkType.ROOTFS)
        with pytest.raises(MtkHeaderError, match="already carries an MTK header"):
            apply(b"K" * 100, once.ramdisk, MtkType.ROOTFS)


class TestRecordType:
    def test_no_headers_records_nothing(self):
        assert record_type(b"K" * 100, b"R" * 100) == (None, ())

    def test_no_headers_and_no_ramdisk(self):
        assert record_type(b"K" * 100, None) == (None, ())

    def test_rootfs_ramdisk(self):
        kernel = build(b"K" * 100, MtkType.KERNEL)
        ramdisk = build(b"R" * 100, MtkType.ROOTFS)
        assert record_type(kernel, ramdisk) == (MtkType.ROOTFS, ())

    def test_recovery_ramdisk(self):
        kernel = build(b"K" * 100, MtkType.KERNEL)
        ramdisk = build(b"D" * 100, MtkType.RECOVERY)
        assert record_type(kernel, ramdisk) == (MtkType.RECOVERY, ())

    def test_ramdisk_header_without_kernel_warns(self):
        ramdisk = build(b"R" * 100, MtkType.RECOVERY)
        kind, warnings = record_type(b"K" * 100, ramdisk)
        assert kind is MtkType.RECOVERY
        assert warnings == ("Warning: No MTK header found in kernel!",)

    def test_kernel_header_without_ramdisk_assumes_rootfs(self):
        kernel = build(b"K" * 100, MtkType.KERNEL)
        kind, warnings = record_type(kernel, None)
        assert kind is MtkType.ROOTFS
        assert warnings == (
            'Warning: No MTK header found in ramdisk, assuming "rootfs" type!',
        )

    def test_kernel_header_with_untyped_ramdisk_assumes_rootfs(self):
        kernel = build(b"K" * 100, MtkType.KERNEL)
        ramdisk = retag(build(b"R" * 100, MtkType.ROOTFS), "MYSTERY")
        kind, warnings = record_type(kernel, ramdisk)
        assert kind is MtkType.ROOTFS
        assert len(warnings) == 1

    def test_truncated_component_is_not_treated_as_headered(self):
        # has_header and strip must agree, or a damaged extract takes the MTK
        # path and then hard-fails when the header is removed.
        truncated = build(b"K" * 100, MtkType.KERNEL)[: HEADER_SIZE - 1]
        kind, warnings = record_type(truncated, None)
        assert kind is None
        assert warnings == ()
        with pytest.raises(MtkHeaderError):
            strip(truncated)

    def test_recorded_type_survives_a_regen_round_trip(self):
        kernel = build(b"K" * 100, MtkType.KERNEL)
        ramdisk = build(b"R" * 100, MtkType.RECOVERY)
        kind, _ = record_type(kernel, ramdisk)
        regen = apply(strip(kernel), strip(ramdisk), kind)
        assert detect_type(regen.ramdisk) is MtkType.RECOVERY


class TestSubprocessCrossCheck:
    @needs_binary
    @pytest.mark.parametrize("mtk_type", [MtkType.ROOTFS, MtkType.RECOVERY])
    @pytest.mark.parametrize(
        "ksize,rsize", [(4, 4), (512, 512), (1024, 100), (4096, 65536)]
    )
    def test_native_apply_matches_binary_byte_for_byte(self, mtk_type, ksize, rsize):
        kernel = bytes((i * 7 + 3) & 0xFF for i in range(ksize))
        ramdisk = bytes((i * 11 + 5) & 0xFF for i in range(rsize))
        native = apply(kernel, ramdisk, mtk_type)
        via_binary = apply_subprocess(kernel, ramdisk, mtk_type)
        assert native.kernel == via_binary.kernel
        assert native.ramdisk == via_binary.ramdisk

    @needs_binary
    def test_binary_output_is_detected_by_our_detector(self):
        headers = apply_subprocess(b"K" * 2048, b"R" * 700, MtkType.RECOVERY)
        assert has_header(headers.kernel)
        assert has_header(headers.ramdisk)
        assert detect_type(headers.kernel) is MtkType.KERNEL
        assert detect_type(headers.ramdisk) is MtkType.RECOVERY
        assert strip(headers.kernel) == b"K" * 2048
        assert strip(headers.ramdisk) == b"R" * 700

    @needs_binary
    def test_subprocess_rejects_kernel_as_ramdisk_type(self):
        with pytest.raises(MtkHeaderError):
            apply_subprocess(b"K" * 100, b"R" * 100, MtkType.KERNEL)

    def test_subprocess_raises_when_binary_missing(self, monkeypatch):
        monkeypatch.setattr(mtk, "resolve_mkmtkhdr", lambda tool="mkmtkhdr": None)
        with pytest.raises(MtkToolError, match="could not locate"):
            apply_subprocess(b"K" * 100, b"R" * 100, MtkType.ROOTFS)

    def test_subprocess_wraps_an_unrunnable_binary(self, tmp_path, monkeypatch):
        # A stripped exec bit resolves fine but cannot be spawned; that must
        # surface as MtkToolError, not a raw PermissionError.
        broken = tmp_path / "mkmtkhdr"
        broken.write_bytes(b"#!/bin/sh\nexit 0\n")
        broken.chmod(0o644)
        monkeypatch.setattr(mtk, "resolve_mkmtkhdr", lambda tool="mkmtkhdr": broken)
        with pytest.raises(MtkToolError, match="could not execute"):
            apply_subprocess(b"K" * 100, b"R" * 100, MtkType.ROOTFS)

    @needs_binary
    def test_subprocess_reports_tool_failure(self, monkeypatch):
        # A payload the binary rejects surfaces as MtkToolError, not a raw
        # CalledProcessError.
        with pytest.raises(MtkToolError):
            apply_subprocess(b"", b"R" * 100, MtkType.ROOTFS)


class TestBinaryDrivenHeaderBytes:
    """Guard the layout itself: if mkmtkhdr ever changes, this fails loudly."""

    @needs_binary
    @pytest.mark.parametrize("mtk_type", [MtkType.ROOTFS, MtkType.RECOVERY])
    def test_binary_header_field_layout(self, mtk_type, tmp_path):
        kernel = bytes((i * 7 + 3) & 0xFF for i in range(1500))
        ramdisk = bytes((i * 11 + 5) & 0xFF for i in range(900))
        kpath = tmp_path / "kernel.bin"
        rpath = tmp_path / "rootfs.bin"
        kpath.write_bytes(kernel)
        rpath.write_bytes(ramdisk)
        proc = subprocess.run(
            [str(MKMTKHDR), "--kernel", str(kpath), f"--{mtk_type}", str(rpath)],
            cwd=tmp_path,
            capture_output=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        for produced, payload, kind in (
            (kpath.with_name("kernel.bin-mtk"), kernel, MtkType.KERNEL),
            (rpath.with_name("rootfs.bin-mtk"), ramdisk, mtk_type),
        ):
            data = produced.read_bytes()
            assert detect_type(data) is kind
            assert payload_size(data) == len(payload)
            assert data[:512] == build(payload, kind)[:512]
            assert strip(data) == payload
