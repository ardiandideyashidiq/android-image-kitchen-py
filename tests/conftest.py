"""Shared fixtures for driving the prebuilt AIK binaries as a test oracle.

The tools under ``bin/<platform>/<arch>/`` are an *independent* implementation
of the Android boot image format. Round-tripping our own packer through our own
unpacker would only prove self-consistency; cross-checking against these
binaries catches a shared misreading of the spec, which is the whole point of
``tests/test_differential.py``.

Every fixture here is used by that module, but they live in ``conftest`` so
later test modules (golden-image tests, round-trip tests) can reuse them.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import platform
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

#: Repository root, i.e. the directory that owns ``src/`` and ``bin/``.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: ``ANDROID!`` -- present at offset 0 in *every* header version the bundled
#: tools emit, including v3/v4.
BOOT_MAGIC = b"ANDROID!"

#: Header versions the bundled ``mkbootimg`` accepts.
HEADER_VERSIONS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: A minimal but valid init script: enough structure that the archive is
#: indistinguishable from a real one at the header level under test.
_INIT_SCRIPT = "#!/system/bin/sh\nexec /system/bin/sh\n"


#: The bundled tools are built for x86_64 only. Anything else must skip rather
#: than try to exec a foreign-architecture binary and fail confusingly.
_SUPPORTED_MACHINES = frozenset({"x86_64", "amd64"})


def host_is_supported() -> bool:
    """Whether the running host has a matching prebuilt tool set."""
    return platform.machine().lower() in _SUPPORTED_MACHINES


def platform_tag() -> str:
    """Return the ``bin/`` subdirectory name for the running host.

    Mirrors the shipped tree: ``bin/linux/x86_64``, ``bin/macos/...``.
    """
    plat = "macos" if platform.system() == "Darwin" else "linux"
    return f"{plat}/{platform.machine()}"


def _candidate_bin_dirs() -> list[Path]:
    """Directories that may hold the tools for *this* host.

    On x86_64 both the exact platform tag and the ``linux/x86_64`` fallback are
    considered (macOS hosts may run the Linux build under some setups, and the
    tree only ships x86_64). On any other architecture no candidate is offered
    at all, so the suite skips instead of exec'ing the wrong binary.
    """
    if not host_is_supported():
        return []
    candidates = [REPO_ROOT / "bin" / platform_tag()]
    fallback = REPO_ROOT / "bin" / "linux" / "x86_64"
    if fallback not in candidates:
        candidates.append(fallback)
    return candidates


class OracleError(RuntimeError):
    """A bundled binary exited non-zero or produced unusable output."""


def _describe(proc: subprocess.CompletedProcess[bytes]) -> str:
    return (
        f"exit {proc.returncode}\n"
        f"stdout: {proc.stdout.decode(errors='replace')}\n"
        f"stderr: {proc.stderr.decode(errors='replace')}"
    )


def run_binary(
    argv: Sequence[os.PathLike[str] | str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[bytes]:
    """Run a bundled binary, raising :class:`OracleError` on failure."""
    proc = subprocess.run(
        [str(a) for a in argv], cwd=cwd, capture_output=True, check=False
    )
    if proc.returncode != 0:
        raise OracleError(f"{' '.join(str(a) for a in argv)}\n{_describe(proc)}")
    return proc


# --------------------------------------------------------------------------
# bin/ discovery
# --------------------------------------------------------------------------


def find_bin_dir() -> Path | None:
    """Locate a ``bin/`` directory holding both boot-image tools, or None."""
    for candidate in _candidate_bin_dirs():
        if (candidate / "mkbootimg").is_file() and (
            candidate / "unpackbootimg"
        ).is_file():
            return candidate
    return None


def require_bin_dir() -> Path:
    """Return the ``bin/`` directory, or skip the test with a clear reason."""
    found = find_bin_dir()
    if found is not None:
        return found
    if not host_is_supported():
        pytest.skip(
            f"prebuilt AIK binaries are x86_64-only; this host reports "
            f"{platform.machine()} and has no matching build"
        )
    candidates = _candidate_bin_dirs()
    pytest.skip(
        "prebuilt AIK binaries not found; looked in "
        + (", ".join(str(c) for c in candidates) or "(no candidate for this host)")
    )


@pytest.fixture(scope="session")
def bin_dir() -> Path:
    """Directory holding the prebuilt AIK binaries, skipping if unavailable."""
    return require_bin_dir()


@pytest.fixture(scope="session")
def mkbootimg(bin_dir: Path) -> Path:
    """Path to the bundled ``mkbootimg``."""
    return bin_dir / "mkbootimg"


@pytest.fixture(scope="session")
def unpackbootimg(bin_dir: Path) -> Path:
    """Path to the bundled ``unpackbootimg``."""
    return bin_dir / "unpackbootimg"


# --------------------------------------------------------------------------
# cpio ramdisk helper
# --------------------------------------------------------------------------


def _have_cpio(*, reproducible: bool = False) -> bool:
    if shutil.which("cpio") is None:
        return False
    if reproducible:
        # Must check the exit status, not the text: a cpio that lacks the flag
        # prints "unrecognized option '--reproducible'", which of course
        # contains the substring we might grep for. That is exactly the
        # BusyBox case the fallback below exists to handle.
        probe = subprocess.run(
            ["cpio", "--reproducible", "--help"], capture_output=True, check=False
        )
        return probe.returncode == 0
    probe = subprocess.run(["cpio", "--version"], capture_output=True, check=False)
    # BusyBox cpio has no --version and exits non-zero.
    return probe.returncode in (0, 1)


def build_ramdisk_cpio(entries: Mapping[str, str | bytes] | None = None) -> bytes:
    """Return a gzip-compressed SVR4 ``newc`` cpio archive.

    ``entries`` maps POSIX paths inside the archive to file contents; it
    defaults to a single ``init`` script.

    Determinism needs *three* things pinned, not just the obvious one:

    * ``c_mtime`` -- every entry header embeds it, so we ``touch -d @0``.
    * ``c_ino`` / ``c_dev*`` -- the inode number is allocated by the kernel
      and is reused unpredictably across ``mkdtemp`` calls, which makes a
      fresh archive differ byte-for-byte on an otherwise identical tree.
      ``cpio --reproducible`` zeroes those fields for us; where it is
      unavailable we fall back to patching them in the raw archive.
    * The gzip header mtime, pinned via ``gzip.compress(..., mtime=0)``.

    With all three pinned the archive is byte-stable across processes.
    """
    if entries is None:
        entries = {"init": _INIT_SCRIPT}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        for rel, content in entries.items():
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                target.write_text(content)
            else:
                target.write_bytes(content)
        subprocess.run(
            ["find", str(root), "-exec", "touch", "-d", "@0", "{}", "+"],
            check=True,
            capture_output=True,
        )
        reproducible = _have_cpio(reproducible=True)
        cpio_flag = "--reproducible" if reproducible else ""
        proc = subprocess.run(
            f"find . -print0 | cpio --null {cpio_flag} -o -H newc",
            shell=True,
            cwd=root,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise OracleError(f"cpio failed: {proc.stderr.decode(errors='replace')}")
    archive = proc.stdout
    if not reproducible:
        archive = _zero_inode_fields(archive)
    return gzip.compress(archive, 9, mtime=0)


#: Byte ranges of the inode/device fields inside a 110-byte ``newc`` header.
#: ``c_magic`` is 6 bytes, then 12 eight-char fields, then ``c_check``.
_NEW_INODE_FIELDS = ((6, 14), (62, 70), (70, 78), (78, 86), (86, 94))
_NEW_HEADER_SIZE = 110


def _zero_inode_fields(archive: bytes) -> bytes:
    """Zero ``c_ino``/``c_dev*`` in every ``newc`` entry header.

    Fallback for systems whose ``cpio`` lacks ``--reproducible`` (notably
    BusyBox). Walks the archive by its own header fields and blanks the same
    bytes ``--reproducible`` would.
    """
    out = bytearray(archive)
    offset = 0
    while offset + _NEW_HEADER_SIZE <= len(out):
        if out[offset : offset + 6] != b"070701":
            break
        for start, stop in _NEW_INODE_FIELDS:
            out[offset + start : offset + stop] = b"0" * (stop - start)
        namesize = int(bytes(out[offset + 94 : offset + 102]))
        filesize = int(bytes(out[offset + 54 : offset + 62]))
        name_block = _NEW_HEADER_SIZE + namesize
        if name_block % 4:
            name_block += 4 - (name_block % 4)
        data_block = name_block + filesize
        if data_block % 4:
            data_block += 4 - (data_block % 4)
        name = bytes(
            out[offset + _NEW_HEADER_SIZE : offset + _NEW_HEADER_SIZE + namesize]
        )
        if name.rstrip(b"\x00") == b"TRAILER!!!":
            break
        offset += data_block
    return bytes(out)


@pytest.fixture(scope="session")
def ramdisk() -> bytes:
    """A small, deterministic cpio.gz ramdisk."""
    if not _have_cpio():
        pytest.skip("system cpio is required to build the test ramdisk")
    return build_ramdisk_cpio()


# --------------------------------------------------------------------------
# synthetic image construction
# --------------------------------------------------------------------------

#: Payload name -> ``mkbootimg`` flag.
PART_FLAGS: dict[str, str] = {
    "kernel": "--kernel",
    "ramdisk": "--ramdisk",
    "second": "--second",
    "dtb": "--dtb",
}


@dataclass(frozen=True, slots=True)
class OracleFields:
    """Parsed ``<img>-<field>`` files produced by ``unpackbootimg``."""

    values: dict[str, str]
    parts: dict[str, Path]
    directory: Path
    stdout: str

    def get(self, name: str) -> str | None:
        """Return the text of one field file, or None if absent."""
        return self.values.get(name)

    def int_field(self, name: str) -> int | None:
        """Return a field parsed as an integer (``0x``-prefix aware)."""
        raw = self.values.get(name)
        return None if raw is None else int(raw, 0)

    def part_bytes(self, name: str) -> bytes:
        return self.parts[name].read_bytes()

    def has(self, name: str) -> bool:
        return name in self.values or name in self.parts


@dataclass(frozen=True, slots=True)
class SyntheticImage:
    """An image built by the bundled ``mkbootimg`` plus its oracle view."""

    path: Path
    data: bytes
    inputs: dict[str, bytes] = field(default_factory=dict)
    fields: OracleFields | None = None

    def part(self, name: str) -> bytes:
        """The exact input bytes supplied for a payload."""
        return self.inputs[name]

    @property
    def oracle(self) -> OracleFields:
        if self.fields is None:
            raise AssertionError("image was built without running unpackbootimg")
        return self.fields


#: Suffixes ``unpackbootimg`` writes for header *fields* (small text files).
#:
#: Classification must be driven by this name set, never by file size. A
#: realistic cmdline is 100+ bytes and a test kernel can be 16 bytes, so any
#: size threshold misfiles both directions.
ORACLE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "base",
        "board",
        "cmdline",
        "dtb_offset",
        "hashtype",
        "header_version",
        "kernel_offset",
        "os_patch_level",
        "os_version",
        "pagesize",
        "ramdisk_offset",
        "second_offset",
        "tags_offset",
    }
)

#: Suffixes ``unpackbootimg`` writes for extracted *payloads* (binary files).
ORACLE_PART_NAMES: frozenset[str] = frozenset(
    {
        "kernel",
        "ramdisk",
        "second",
        "dtb",
        "recovery_dtbo",
        "recovery_acpio",
        "extra_cmdline",
    }
)


def _read_oracle_fields(
    directory: Path, image: Path
) -> tuple[dict[str, str], dict[str, Path]]:
    """Split unpackbootimg output into text fields and binary payload parts.

    Classification is by suffix name, not by size: a 99-character cmdline
    exceeds any sane "small file" threshold while a 16-byte kernel falls below
    it, so a size heuristic silently swaps the two.
    """
    values: dict[str, str] = {}
    parts: dict[str, Path] = {}
    prefix = image.name + "-"
    for item in sorted(directory.iterdir()):
        stem = item.name.removeprefix(prefix)
        if stem in ORACLE_FIELD_NAMES:
            values[stem] = item.read_text().strip()
        elif stem in ORACLE_PART_NAMES:
            parts[stem] = item
        else:
            # Unknown suffix: fail loudly rather than guess. A new field in a
            # future build should be classified deliberately, not by accident.
            raise OracleError(
                f"unclassified unpackbootimg output {item.name!r} (suffix {stem!r}); "
                f"add it to ORACLE_FIELD_NAMES or ORACLE_PART_NAMES"
            )
    return values, parts


def make_synthetic_image(
    tool: Path,
    out_dir: Path,
    *,
    header_version: int = 1,
    page_size: int = 2048,
    cmdline: str = "console=ttyS0 androidboot.hardware=differential",
    board: str = "diffboard",
    base: int = 0x80000000,
    hashtype: str = "sha1",
    parts: Mapping[str, bytes],
) -> SyntheticImage:
    """Build a boot image with the bundled ``mkbootimg``.

    Returns a :class:`SyntheticImage` carrying the raw bytes and the exact
    input payloads, so callers can assert byte-identical round-trips.
    """
    out = (
        out_dir
        / f"{_image_stem(header_version, page_size, hashtype, cmdline, board, base, parts)}.img"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as partdir:
        argv: list[str] = [
            str(tool),
            "--cmdline",
            cmdline,
            "--board",
            board,
            "--base",
            hex(base),
            "--pagesize",
            str(page_size),
            "--header_version",
            str(header_version),
            "--hashtype",
            hashtype,
        ]
        part_dir = Path(partdir)
        for part_name, flag in PART_FLAGS.items():
            if part_name not in parts:
                continue
            blob_path = part_dir / f"{part_name}.bin"
            blob_path.write_bytes(parts[part_name])
            argv += [flag, str(blob_path)]
        argv += ["-o", str(out)]
        try:
            run_binary(argv)
        except OracleError as exc:
            raise OracleError(
                f"mkbootimg failed for hv={header_version}\n{exc}"
            ) from exc

    data = out.read_bytes()
    if data[:8] != BOOT_MAGIC:
        raise OracleError(
            f"mkbootimg produced no ANDROID! magic at offset 0 for hv={header_version}: {data[:8]!r}"
        )
    return SyntheticImage(path=out, data=data, inputs=dict(parts))


def _image_stem(
    header_version: int,
    page_size: int,
    hashtype: str,
    cmdline: str,
    board: str,
    base: int,
    parts: Mapping[str, bytes],
) -> str:
    """A filename that is unique per distinct image *and* readable.

    Two tests that share a header version but differ in payload would
    otherwise write the same file, so a parallel run could unpack the wrong
    image and report a bogus differential result. Hashing the full input set
    makes collisions effectively impossible while keeping the human-readable
    prefix for debugging.
    """
    digest = hashlib.sha256()
    digest.update(
        f"{header_version}|{page_size}|{hashtype}|{cmdline}|{board}|{base}".encode()
    )
    for name in sorted(parts):
        digest.update(f"|{name}:{len(parts[name])}:".encode())
        digest.update(parts[name])
    return (
        f"synth-hv{header_version}-ps{page_size}-{hashtype}-{digest.hexdigest()[:12]}"
    )


def unpack_with_oracle(tool: Path, image: Path, out_dir: Path) -> OracleFields:
    """Run the bundled ``unpackbootimg`` and parse its output directory.

    ``unpackbootimg -o`` requires the output directory to already exist; it
    fails with ``Could not stat ...`` otherwise, so we create it first.
    """
    outdir = out_dir / f"unpack-{image.stem}"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    argv = [str(tool), "-i", str(image), "-o", str(outdir)]
    try:
        proc = run_binary(argv)
    except OracleError as exc:
        raise OracleError(f"unpackbootimg failed for {image}\n{exc}") from exc
    values, parts = _read_oracle_fields(outdir, image)
    return OracleFields(
        values=values,
        parts=parts,
        directory=outdir,
        stdout=proc.stdout.decode(errors="replace"),
    )


def build_and_unpack(
    mk: Path, unpack: Path, out_dir: Path, **kwargs: Any
) -> SyntheticImage:
    """Build with ``mkbootimg`` then unpack with ``unpackbootimg``.

    The returned :class:`SyntheticImage` carries both the raw bytes and the
    oracle's view of the header, which is exactly what the differential
    assertions need.
    """
    image = make_synthetic_image(mk, out_dir, **kwargs)
    return SyntheticImage(
        path=image.path,
        data=image.data,
        inputs=image.inputs,
        fields=unpack_with_oracle(unpack, image.path, out_dir),
    )


# --------------------------------------------------------------------------
# pytest fixtures exposing the factories
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def image_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped scratch directory for generated images."""
    return tmp_path_factory.mktemp("oracle-images")


@pytest.fixture
def image_factory(mkbootimg: Path, unpackbootimg: Path, image_dir: Path):
    """Factory building oracle-backed synthetic images.

    Keyword arguments are forwarded to ``mkbootimg`` via
    :func:`make_synthetic_image`; ``parts`` maps payload names (``kernel``,
    ``ramdisk``, ``second``, ``dtb``) to their bytes.
    """

    def factory(*, parts: Mapping[str, bytes], **kwargs: Any) -> SyntheticImage:
        return build_and_unpack(
            mkbootimg, unpackbootimg, image_dir, parts=parts, **kwargs
        )

    return factory


def u32(buf: bytes, offset: int) -> int:
    """Read a little-endian u32, mirroring the on-disk header encoding."""
    return struct.unpack_from("<I", buf, offset)[0]
