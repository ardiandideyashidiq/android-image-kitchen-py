"""Subprocess boundary shared by the exotic format wrappers.

The Bash originals discard stderr and check only the exit status of the last
element of a pipeline, so a tool that fails halfway through is routinely
reported as success. Every bundled tool is invoked through :func:`run_tool`,
which captures both streams and folds stderr into the raised error.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ToolError",
    "ToolMissingError",
    "bin_dir",
    "platform_tag",
    "run_tool",
    "tool",
]

_REPO_ROOT = Path(__file__).resolve().parents[3]


class ToolError(RuntimeError):
    """A bundled tool exited non-zero."""


class ToolMissingError(ToolError):
    """A bundled tool is not present on this platform."""


def platform_tag() -> str:
    plat = "macos" if platform.system() == "Darwin" else "linux"
    return f"{plat}/{platform.machine()}"


def _aik_root() -> Path:
    """Locate the checkout that owns ``bin/``.

    Walking up from this file is right for a source checkout and an editable
    install, but a wheel install lands in site-packages where there is no
    ``bin/`` to find. ``AIK_ROOT`` is the escape hatch for that case, the same
    way the Bash anchors on its own directory via ``BASH_SOURCE``.
    """
    override = os.environ.get("AIK_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return _REPO_ROOT


def bin_dir() -> Path:
    return _aik_root() / "bin" / platform_tag()


def tool(name: str) -> Path:
    path = bin_dir() / name
    if not path.is_file():
        raise ToolMissingError(
            f"{name} not found at {path}; the bundled tools are platform specific "
            f"({platform_tag()}) and may not ship for this host. Set AIK_ROOT if "
            f"the aik-py package is installed away from its checkout."
        )
    return path


def _env() -> dict[str, str]:
    """Environment for the bundled tools.

    The macOS binaries link the dylibs that ship beside them (libintl, liblzma,
    liblzo2, libmagic) and dyld will not find those on its own, so the loader
    path has to be set for every invocation -- the Bash wraps each tool the
    same way. Harmless on Linux, where the variable is unused.
    """
    env = dict(os.environ)
    env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
        [str(bin_dir()), env.get("DYLD_LIBRARY_PATH", "")]
    ).rstrip(os.pathsep)
    return env


def run_tool(
    name: str,
    *args: str | os.PathLike[str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    argv = [str(tool(name)), *(str(a) for a in args)]
    proc = subprocess.run(
        argv,
        cwd=cwd,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ToolError(
            f"{' '.join(argv)} failed with exit status {proc.returncode}"
            + (f":\n{detail}" if detail else "")
        )
    return proc


@dataclass(frozen=True, slots=True)
class Unpacked:
    """Minimal shape shared by every exotic unpack result.

    Mirrors what the Bash keeps in ``split_img/<image>-<suffix>``: a directory
    of part files plus a handful of small text files holding parsed header
    fields. Native formats can adopt the same shape so the session layer does
    not need to special-case the wrappers.
    """

    directory: Path
    parts: dict[str, Path]
    fields: dict[str, str]

    def part(self, key: str) -> Path:
        try:
            return self.parts[key]
        except KeyError:
            raise KeyError(
                f"no '{key}' part; unpacked parts: {sorted(self.parts)}"
            ) from None

    def field(self, key: str, default: str | None = None) -> str:
        if key in self.fields:
            return self.fields[key]
        if default is not None:
            return default
        raise KeyError(f"no '{key}' field; parsed fields: {sorted(self.fields)}")

    def part_bytes(self, key: str) -> bytes:
        return self.part(key).read_bytes()
