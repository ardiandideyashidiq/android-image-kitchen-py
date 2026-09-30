"""Resolution of the prebuilt tools vendored under ``bin/``.

The Bash toolkit picks a binary with ``arch=$plat/$(uname -m)`` and then runs
``"$bin/$arch/$tool"``. This module reproduces that layout and, when no binary
ships for the running platform, falls back to the ambient ``PATH`` where that
is safe to do so.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

try:
    from .errors import MissingToolError
except ImportError:  # pragma: no cover - errors module lands in a sibling unit

    class MissingToolError(RuntimeError):
        """A required external tool could not be located."""


def _bin_root() -> Path:
    """Locate the vendored ``bin/`` directory.

    In a source checkout this is ``<repo>/bin``. An installed wheel contains
    only the package, so fall back to the working directory AIK is run from
    and finally to ``$AIK_BIN``, which keeps the resolver from silently
    degrading to whatever happens to be on ``PATH``.
    """
    override = os.environ.get("AIK_BIN")
    if override:
        return Path(override)

    packaged = Path(__file__).resolve().parents[2] / "bin"
    if packaged.is_dir():
        return packaged
    return Path.cwd() / "bin"


BIN_ROOT = _bin_root()


def plat() -> str:
    """Return the AIK platform directory name for the running OS."""
    return "macos" if sys.platform == "darwin" else "linux"


def machine() -> str:
    """Return the AIK architecture directory name, or ``None`` when unsupported.

    Only the four architecture directories AIK actually ships are recognised; an
    ``aarch64`` host has no bundled binaries at all.
    """
    m = platform.machine().lower()
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if m in ("i386", "i486", "i586", "i686", "x86"):
        # AIK names the 32-bit x86 directory differently per platform.
        return "i386" if plat() == "macos" else "i686"
    return None


def candidate_paths(name: str) -> list[Path]:
    """Return every location searched for ``name``, in priority order."""
    candidates: list[Path] = []

    override = os.environ.get(f"AIK_{name.upper().replace('-', '_')}")
    if override:
        candidates.append(Path(override))

    arch = machine()
    if arch is not None:
        candidates.append(BIN_ROOT / plat() / arch / name)

    found = shutil.which(name)
    if found:
        candidates.append(Path(found))

    return candidates


def tool_path(name: str) -> Path:
    """Resolve ``name`` to an existing path, searching ``bin/`` then ``PATH``.

    Raises:
        MissingToolError: if no candidate exists.
    """
    candidates = candidate_paths(name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(c) for c in candidates) or "<no candidates>"
    raise MissingToolError(f"Required tool {name!r} was not found. Tried: {tried}.")


def require(name: str) -> str:
    """Resolve ``name`` to an executable path suitable for ``subprocess``.

    Raises:
        MissingToolError: if the resolved file is not executable.
    """
    path = tool_path(name)
    if not os.access(path, os.X_OK):
        raise MissingToolError(f"Tool {name!r} at {path} is not executable.")
    return str(path)
