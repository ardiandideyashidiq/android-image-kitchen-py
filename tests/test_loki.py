import os
import subprocess
from pathlib import Path

import pytest

from aik_py.patches import loki

TOOL = Path(__file__).resolve().parents[1] / "bin" / "linux" / "x86_64" / "loki_tool"
has_tool = pytest.mark.skipif(
    not TOOL.is_file(), reason="bundled loki_tool binary is not present"
)


@pytest.mark.parametrize("partition", ["boot", "recovery", "BOOT", "Recovery"])
def test_partition_accepted(partition: str) -> None:
    assert loki._check_partition(partition) == partition.lower()


@pytest.mark.parametrize("partition", ["", "system", "kernel", "boot.img"])
def test_partition_rejected(partition: str) -> None:
    with pytest.raises(ValueError):
        loki._check_partition(partition)


def test_unlok_rejects_bad_partition() -> None:
    with pytest.raises(ValueError):
        loki.unlok(b"x", "system")


def test_patch_requires_aboot(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        loki.patch(b"x", "boot", tmp_path / "missing.img")


def test_patch_rejects_empty_aboot(tmp_path: Path) -> None:
    empty = tmp_path / "aboot.img"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        loki.patch(b"x", "boot", empty)


def test_unlok_argv_matches_tool_contract(tmp_path: Path) -> None:
    argv = loki._unlok_argv(TOOL, tmp_path / "in.lok", tmp_path / "out.img")
    assert argv[1:] == ["unlok", str(tmp_path / "in.lok"), str(tmp_path / "out.img")]


def test_patch_argv_matches_tool_contract(tmp_path: Path) -> None:
    argv = loki._patch_argv(
        TOOL, "boot", tmp_path / "aboot.img", tmp_path / "in.img", tmp_path / "out.lok"
    )
    assert argv[1:] == [
        "patch",
        "boot",
        str(tmp_path / "aboot.img"),
        str(tmp_path / "in.img"),
        str(tmp_path / "out.lok"),
    ]


def test_unlok_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loki, "_tool", lambda: TOOL)
    failure = subprocess.CalledProcessError(
        1,
        ["loki_tool"],
        output=b"Loki tool v2.1\n[-] Failed to find function to patch.\n",
    )

    def boom(*a: object, **kw: object) -> None:
        raise failure

    monkeypatch.setattr(loki.subprocess, "run", boom)
    with pytest.raises(loki.LokiError, match="Failed to find function"):
        loki.unlok(os.urandom(4096), "boot")


def test_error_marker_on_exit_zero_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(loki, "_tool", lambda: TOOL)

    def zero_exit(*a: object, **kw: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=["loki_tool"],
            returncode=0,
            stdout=b"Loki tool v2.1\n[-] Input file is not a Loki image.\n",
            stderr=b"",
        )

    monkeypatch.setattr(loki.subprocess, "run", zero_exit)
    with pytest.raises(loki.LokiError, match="not a Loki image"):
        loki.unlok(os.urandom(4096), "boot")


def test_diagnostic_falls_back_to_exit_status() -> None:
    failure = subprocess.CalledProcessError(3, ["loki_tool"], output=b"", stderr=None)
    assert "exit status 3" in loki._diagnostic(failure)


def test_non_executable_tool_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = tmp_path / "loki_tool"
    stub.write_bytes(b"")
    stub.chmod(0o644)
    monkeypatch.setattr(loki, "_resolve_tool", lambda name: stub)
    with pytest.raises(loki.MissingToolError):
        loki._tool()


@has_tool
def test_real_tool_rejects_non_loki_image() -> None:
    # loki_tool exits 0 and copies the input through for a non-Loki image.
    with pytest.raises(loki.LokiError, match="not a Loki image"):
        loki.unlok(os.urandom(4096), "boot")


@has_tool
def test_real_tool_rejects_garbage_aboot(tmp_path: Path) -> None:
    aboot = tmp_path / "aboot.img"
    aboot.write_bytes(os.urandom(4096))
    with pytest.raises(loki.LokiError):
        loki.patch(os.urandom(4096), "boot", aboot)


@has_tool
def test_patch_normalizes_partition_case(tmp_path: Path) -> None:
    aboot = tmp_path / "aboot.img"
    aboot.write_bytes(os.urandom(4096))
    with pytest.raises(loki.LokiError) as excinfo:
        loki.patch(os.urandom(4096), "BOOT", aboot)
    assert "must be boot or recovery" not in str(excinfo.value)


def test_round_trip_requires_signed_image() -> None:
    pytest.skip("no genuine Loki-signed boot/recovery fixture available")
