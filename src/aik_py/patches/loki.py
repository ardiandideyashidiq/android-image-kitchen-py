"""Loki boot-image encryption bypass.

Loki encrypts the boot/recovery partition on some Qualcomm devices by rewriting
the image header and moving the real payload. The bundled ``loki_tool`` binary
(``bin/linux/x86_64/loki_tool``, v2.1) knows the transform; re-deriving it here
would mean reimplementing its crypto, so every operation shells out to it.

The tool only operates on files, so each call stages its input and output in a
temporary directory.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

PARTITIONS = ("boot", "recovery")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOL_NAME = "loki_tool"
_TOOL_CANDIDATE_DIRS = (
    "bin/linux/x86_64",
    "bin/linux/arm64",
    "bin/darwin",
)

try:  # pragma: no cover - depends on which sibling units have landed
    from aik_py.bin import resolve_tool as _resolve_tool
except ImportError:  # pragma: no cover - standalone fallback

    def _resolve_tool(name: str) -> Path | None:
        for relative in _TOOL_CANDIDATE_DIRS:
            candidate = _REPO_ROOT / relative / name
            if candidate.is_file():
                return candidate
        found = shutil.which(name)
        return Path(found) if found else None


try:  # pragma: no cover - depends on which sibling units have landed
    from aik_py.errors import CorruptImageError, MissingToolError
except ImportError:  # pragma: no cover - standalone fallback

    class MissingToolError(RuntimeError):
        """A required external tool is not available."""

    class CorruptImageError(ValueError):
        """An image is not in the shape the operation requires."""


# loki_tool prefixes every failure with this, and reports on stdout.
_ERROR_MARKER = "[-]"
_VERSION_BANNER = re.compile(r"^\s*Loki\b")


class LokiError(CorruptImageError):
    """``loki_tool`` rejected the image or the requested operation failed."""


def _check_partition(partition: str) -> str:
    normalized = partition.lower()
    if normalized not in PARTITIONS:
        raise ValueError(f"Loki applies to {PARTITIONS} images, not {partition!r}")
    return normalized


def _tool() -> Path:
    tool = _resolve_tool(_TOOL_NAME)
    if tool is None:
        raise MissingToolError(
            f"{_TOOL_NAME} not found; install the bundled Android Image Kitchen "
            f"binaries under {(_REPO_ROOT / 'bin').as_posix()}"
        )
    if not os.access(tool, os.X_OK):
        raise MissingToolError(f"{_TOOL_NAME} at {tool} is not executable")
    return tool


def _run(argv: list[str]) -> str:
    try:
        completed = subprocess.run(argv, check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        raise LokiError(f"{_TOOL_NAME} {argv[1]} failed: {_diagnostic(exc)}") from exc
    output = _stream_text(completed.stdout) + _stream_text(completed.stderr)
    # `unlok` reports a non-Loki image on stdout and still exits 0, copying its
    # input to the output file, so the exit status alone cannot be trusted.
    if _ERROR_MARKER in output:
        raise LokiError(f"{_TOOL_NAME} {argv[1]} failed: {_summarize(output)}")
    return output


def _stream_text(stream: bytes | None) -> str:
    return stream.decode(errors="replace") if stream else ""


def _summarize(output: str) -> str:
    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip() and not _VERSION_BANNER.match(line)
    ]
    if not lines:
        return "no diagnostic output"
    errors = [line for line in lines if _ERROR_MARKER in line]
    return " ".join(errors or lines)


def _diagnostic(exc: subprocess.CalledProcessError | OSError) -> str:
    if isinstance(exc, OSError):
        return str(exc)
    streams = (exc.stdout, exc.stderr)
    for stream in streams:
        text = _stream_text(stream)
        if text.strip():
            return _summarize(text)
    return f"exit status {exc.returncode} with no diagnostic output"


def _unlok_argv(tool: Path, source: Path, dest: Path) -> list[str]:
    return [str(tool), "unlok", str(source), str(dest)]


def _patch_argv(
    tool: Path, partition: str, aboot: Path, source: Path, dest: Path
) -> list[str]:
    return [str(tool), "patch", partition, str(aboot), str(source), str(dest)]


def unlok(data: bytes, partition: str) -> bytes:
    """Revert a Loki-patched image, returning the plaintext image."""
    _check_partition(partition)
    if not data:
        raise CorruptImageError("cannot unlok an empty image")
    tool = _tool()
    with tempfile.TemporaryDirectory(prefix="aik-loki-") as tmp:
        root = Path(tmp)
        source = root / "in.lok"
        dest = root / "out.img"
        source.write_bytes(data)
        _run(_unlok_argv(tool, source, dest))
        if not dest.is_file():
            raise LokiError("loki_tool produced no output image")
        return dest.read_bytes()


def patch(data: bytes, partition: str, aboot: Path) -> bytes:
    """Apply the Loki transform to ``data``.

    ``aboot`` is a dump of the device's aboot partition. The tool reads the
    patched-boot header out of it to recover the per-device Loki offset, which
    differs per bootloader build, so no sane default exists and the caller must
    supply their own dump.
    """
    partition = _check_partition(partition)
    aboot = Path(aboot)
    if not aboot.is_file():
        raise FileNotFoundError(
            f"Device aboot.img required to find the Loki patch offset: {aboot} is missing"
        )
    if not aboot.stat().st_size:
        raise CorruptImageError(f"aboot dump is empty: {aboot}")

    tool = _tool()
    with tempfile.TemporaryDirectory(prefix="aik-loki-") as tmp:
        root = Path(tmp)
        source = root / "in.img"
        dest = root / "out.lok"
        source.write_bytes(data)
        _run(_patch_argv(tool, partition, aboot, source, dest))
        if not dest.is_file():
            raise LokiError("loki_tool produced no patched image")
        return dest.read_bytes()
