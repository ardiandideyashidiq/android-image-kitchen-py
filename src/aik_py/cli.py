"""Command line interface for Android Image Kitchen, replacing the Bash toolkit.

``unpack`` mirrors ``unpackimg.sh``, ``repack`` mirrors ``repackimg.sh`` and ``clean``
mirrors ``cleanup.sh``. Several behaviours of the Bash originals are deliberately not
reproduced; see the notes on each command.

The unpack/repack back ends live in sibling modules owned by other parts of this project
(``aik_py.session``, ``aik_py.repack``, ``aik_py.manifest``). They are imported lazily so
this module stays importable, and testable, before those modules land.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import rich_click as click
from loguru import logger
from rich.console import Console
from rich.table import Table

__all__ = ["cli", "main"]

console = Console()
err_console = Console(stderr=True)


# `unpack` with no argument falls back to the first image in the working directory, but must
# never pick one of these: aboot.img is a companion dump the user unpacks separately, and the
# `*-new.img` files are this tool's own output from a previous run.
AUTO_PICK_PATTERNS = ("*.elf", "*.img", "*.sin")
AUTO_PICK_SKIP = frozenset(
    {"aboot.img", "image-new.img", "unlokied-new.img", "unsigned-new.img"}
)

# cleanup.sh ran `rm -rf ramdisk split_img *new.*`; that glob is anchored nowhere, so it
# matches any name containing "new." -- reproduce that set rather than a stricter pattern.
CLEAN_DIRS = ("ramdisk", "split_img")

# repackimg.sh's default compression level varied by compressor (xz wanted -1, lz4 -9). That
# choice belongs to whichever compressor the back end selects, so an unset --level is passed
# through as None and resolved there rather than pinned to a single value here.

LOG_LEVELS = {"trace", "debug", "info", "success", "warning", "error", "critical"}


class AikError(Exception):
    """An expected, user-facing failure.

    The Bash scripts' `abort()` printed "Error!" and returned, leaving every call site to
    remember an explicit `exit 1`. Raising instead means one handler at the top level owns
    the exit status.
    """


class NotImplementedYet(AikError):
    """A back end module owned by another part of the project is not present yet."""


def configure_logging(verbose: bool, quiet: bool) -> None:
    """Point loguru at stderr at a level the flags select.

    Loguru is a singleton, so re-running this replaces the handler rather than stacking one.
    """
    level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level, format="{message}", colorize=True)


def load_backend(module: str, *, feature: str) -> ModuleType:
    """Import a back end module, or explain that it is not implemented yet."""
    try:
        return importlib.import_module(f"aik_py.{module}")
    except ImportError as exc:
        raise NotImplementedYet(
            f"{feature} is not implemented yet: the 'aik_py.{module}' module is missing ({exc})."
        ) from exc


def tool_dir() -> Path:
    """The AIK installation root, i.e. the directory holding ``bin/``.

    repackimg.sh and cleanup.sh both defaulted to this directory instead of the caller's CWD;
    ``--local`` opts out of that. The installed package lives at ``<root>/src/aik_py``, so walk
    up from it and fall back to CWD when aik_py is not inside a checkout (e.g. a wheel).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "bin").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def work_dir(local: bool) -> Path:
    """The directory a command operates on: CWD with ``--local``, else the tool directory."""
    return Path.cwd() if local else tool_dir()


def pick_image(base: Path) -> Path | None:
    """Auto-detect an image in *base*, mirroring unpackimg.sh's `ls *.elf *.img *.sin` scan.

    Bash expanded all three globs into one argument list and handed it to `ls`, so the whole
    set was sorted together -- `aaa.img` won over `zzz.elf`. Collect every pattern first and
    sort globally, or a directory holding both kinds picks a different image than the script.
    """
    candidates: set[Path] = set()
    for pattern in AUTO_PICK_PATTERNS:
        candidates.update(c for c in base.glob(pattern) if c.is_file())
    for candidate in sorted(candidates):
        if candidate.name in AUTO_PICK_SKIP:
            logger.debug("Skipping {}", candidate.name)
            continue
        return candidate
    return None


def _get(source: Any, key: str) -> Any:
    """Read *key* from a mapping or an attribute-style object, returning None if absent."""
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def emit(message: Any) -> None:
    """Print to stdout, resolving the stream at call time.

    A module-level Console captures ``sys.stdout`` when constructed, which is the wrong stream
    once a test runner has replaced it. See report_error.
    """
    Console().print(message)


_SUMMARY_FIELDS = (
    ("Image type", "imgtype"),
    ("Header version", "header_version"),
    ("Signature", "sigtype"),
    ("AVB type", "avbtype"),
    ("Footer", "tailtype"),
    ("MTK", "mtktype"),
    ("Ramdisk compression", "ramdiskcomp"),
    ("Original size", "origsize"),
)


def render_manifest(manifest: Any) -> None:
    """Print a table describing what was unpacked, or note that there is nothing to show."""
    if manifest is None:
        logger.debug("No manifest available for a summary table")
        return
    rows = [(label, _get(manifest, key)) for label, key in _SUMMARY_FIELDS]
    rows = [(label, value) for label, value in rows if value not in (None, "")]
    if not rows:
        logger.debug("Manifest carried no summary fields")
        return
    table = Table(title="Unpacked image", title_justify="left", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for label, value in rows:
        table.add_row(label, str(value))
    emit(table)


def _local_option(f):
    return click.option(
        "--local",
        is_flag=True,
        help="Operate on the current directory instead of the Android Image Kitchen directory.",
    )(f)


def _log_options(f):
    """Accept -v/-q after the subcommand as well as before it.

    `aik-py --verbose unpack` and `aik-py unpack --verbose` are the same request, and users
    reasonably expect both. The group callback cannot configure logging in time for a
    subcommand-level flag, so each command calls configure_logging() with its own value,
    defaulting to the group's when the flag was not repeated.
    """
    f = click.option(
        "-q",
        "--quiet",
        is_flag=True,
        help="Silence output and report only warnings or errors.",
    )(f)
    return click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")(
        f
    )


def _apply_logging(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """Honour a subcommand-level -v/-q, falling back to whatever the group resolved."""
    if verbose or quiet:
        configure_logging(verbose, quiet)
    else:
        params = ctx.find_root().params
        configure_logging(params.get("verbose", False), params.get("quiet", False))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="aik-py", prog_name="aik-py")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("-q", "--quiet", is_flag=True, help="Only report warnings and errors.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """Android Image Kitchen -- unpack and repack Android boot and recovery images.

    Drop a `boot.img`, `recovery.img` or OEM equivalent into this directory and run
    `aik-py unpack`, edit the extracted tree under `ramdisk/`, then run `aik-py repack` to
    rebuild it. Run `aik-py clean` to reset the working directory.
    """
    configure_logging(verbose, quiet)
    ctx.ensure_object(dict)


@cli.command()
@click.argument(
    "image",
    required=False,
    type=click.Path(path_type=Path),
)
@_local_option
@_log_options
# The Bash accepted "--nosudo"/"--sudo"; keep those spellings rather than click's --no-sudo
# default so existing scripts and muscle memory keep working.
@click.option(
    "--sudo/--nosudo", "sudo", default=True, help="Unpack the ramdisk as root."
)
@click.pass_context
def unpack(
    ctx: click.Context,
    image: Path | None,
    local: bool,
    sudo: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Unpack IMAGE, splitting it into `split_img/` and extracting its ramdisk to `ramdisk/`.

    With no IMAGE, the first image in the working directory is used. `aboot.img` and this
    tool's own `*-new.img` output are skipped, so a re-run never unpacks its own result.
    """
    _apply_logging(ctx, verbose, quiet)
    base = work_dir(local)
    if image is None:
        found = pick_image(base)
        if found is None:
            raise AikError(
                f"No image file supplied and none found in {base}. "
                f"Pass a file explicitly, or drop a {'/'.join(AUTO_PICK_PATTERNS)} here."
            )
        image = found
    else:
        # No click.Path(exists=True) here: it validates before this body runs, and it would
        # reject the unexpanded "~" a user typed. Resolve first, then check, so the failure is
        # our own message rather than a usage error.
        image = image.expanduser()
        if not image.is_absolute():
            image = (Path.cwd() / image).resolve()
        if not image.exists():
            raise AikError(f"No such image file: {image}")
        if not image.is_file():
            raise AikError(f"Not a regular file: {image}")

    # Clear the previous run before opening the input, as unpackimg.sh did, so stale split_img/
    # entries cannot be mistaken for this image's. The chosen image itself is exempt: naming
    # `*new.*` explicitly is legitimate, and cleaning first would delete the input.
    preserve = image if image.parent == base.resolve() else None
    removed = clean_workspace(base, quiet=True, preserve=preserve)
    if removed:
        logger.debug("Removed {} stale artifact(s) from {}", len(removed), base)

    session = load_backend("session", feature="Unpacking")

    logger.info("Unpacking {}", image.name)
    manifest = session.unpack(image, local=local, nosudo=not sudo)
    render_manifest(manifest)
    emit(f"[green]Done.[/green] Unpacked {image.name} in {base}")


def clean_workspace(
    base: Path, *, quiet: bool = False, preserve: Path | None = None
) -> list[Path]:
    """Remove the work directories and scratch files, returning what was deleted.

    The Bash scripts escalated to root when `ramdisk/` was owned by root, because cpio -d
    restores the archived ownership. Do the same.
    """
    resolved_preserve = preserve.resolve() if preserve is not None else None
    removed: list[Path] = []
    for name in CLEAN_DIRS:
        target = base / name
        if target.is_dir():
            _remove(target, root_owned=_is_root_owned(target))
            removed.append(target)

    # cleanup.sh's glob was `*new.*`, which matches any name containing "new." -- including the
    # `unamonet-` and `<type>-` variants unpackimg.sh produces, which no prefix rule would catch.
    for target in sorted(base.glob("*new.*")):
        if resolved_preserve is not None and target.resolve() == resolved_preserve:
            logger.debug("Preserving {} from cleanup", target)
            continue
        _remove(target, root_owned=_is_root_owned(target))
        removed.append(target)

    if not quiet:
        emit("[green]Working directory cleaned.[/green]")
    return removed


def _is_root_owned(path: Path) -> bool:
    try:
        return path.stat().st_uid == 0
    except OSError:
        return False


def _remove(target: Path, *, root_owned: bool) -> None:
    """Delete *target*, escalating to sudo only if it is root-owned and the plain delete fails."""
    try:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    except OSError as exc:
        if not root_owned or os.geteuid() == 0:
            raise AikError(f"Could not remove {target}: {exc.strerror or exc}") from exc
        logger.debug("Retrying removal of {} with sudo", target)
        try:
            subprocess.run(["sudo", "rm", "-rf", "--", str(target)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as sudo_exc:
            raise AikError(
                f"Could not remove {target}: it is owned by root. Retry with sudo, or remove it manually."
            ) from sudo_exc


@cli.command()
@_local_option
# One --quiet flag does both jobs cleanup.sh's did and the group-level -q does: silence the
# confirmation line *and* drop to warnings-only. Two separate flags would collide on the name.
@_log_options
@click.pass_context
def clean(ctx: click.Context, local: bool, quiet: bool, verbose: bool) -> None:
    """Remove `ramdisk/`, `split_img/` and every `*new.*` scratch file from the working directory."""
    _apply_logging(ctx, verbose, quiet)
    base = work_dir(local)
    removed = clean_workspace(base, quiet=quiet)
    logger.debug("Removed {} artifact(s)", len(removed))


def _has_entries(path: Path) -> bool:
    """True when *path* is a directory holding at least one entry; False if it is absent."""
    return path.is_dir() and any(path.iterdir())


def _validate_level(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> int | None:
    """Reject a non-numeric or out-of-range --level.

    repackimg.sh's `[0-9]*` case fell through and dropped a bad value on the floor, so a typo
    silently repacked at the default level. Fail loudly instead.
    """
    if value is None or value == "":
        return None
    try:
        level = int(value)
    except ValueError:
        raise click.BadParameter(f"'{value}' is not a number.") from None
    if not 0 <= level <= 9:
        raise click.BadParameter(f"must be between 0 and 9, got {level}.")
    return level


@cli.command()
@_local_option
@_log_options
@click.option(
    "--original",
    is_flag=True,
    help="Repack using the original ramdisk archive unchanged.",
)
@click.option(
    "--origsize", is_flag=True, help="Pad the rebuilt image back to its original size."
)
@click.option(
    "--level",
    default=None,
    callback=_validate_level,
    metavar="0-9",
    help="Compression level, 0 (fastest) to 9 (smallest).",
)
@click.option(
    "--avbkey",
    metavar="NAME",
    help="AVB key to sign with; needs both <NAME>.pk8 and <NAME>.x509.*.",
)
@click.option("--forceelf", is_flag=True, help="Force the ELF repack path.")
@click.pass_context
def repack(
    ctx: click.Context,
    local: bool,
    original: bool,
    origsize: bool,
    level: int | None,
    avbkey: str | None,
    forceelf: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Rebuild the image from `split_img/` and `ramdisk/`, producing `*-new.img`."""
    _apply_logging(ctx, verbose, quiet)
    base = work_dir(local)
    # Match repackimg.sh: it wanted a non-empty split_img/ or an existing ramdisk/, and a
    # directory that merely exists but is empty is the case that silently produces a broken
    # image, so check for contents rather than existence.
    split_img = base / "split_img"
    ramdisk = base / "ramdisk"
    has_split = _has_entries(split_img)
    has_ramdisk = ramdisk.is_dir()
    if not has_split and not has_ramdisk:
        raise AikError(
            f"No files found to be packed/built: {base} has no populated split_img/ and no ramdisk/. "
            f"Run 'aik-py unpack' first."
        )
    if not original and not _has_entries(ramdisk):
        raise AikError(
            f"No files found to be packed/built: {ramdisk} is missing or empty. Populate it, or pass "
            f"--original to reuse the original ramdisk archive."
        )

    key_path = _resolve_avbkey(avbkey, base) if avbkey else None
    if avbkey and key_path is None:
        raise AikError(
            f"No AVB key named '{avbkey}': expected both '{avbkey}.pk8' and '{avbkey}.x509.*' "
            f"in the given path, {Path.cwd()} or {base}."
        )

    back = load_backend("repack", feature="Repacking")
    logger.info("Repacking in {}", base)
    result = back.repack(
        local=local,
        original=original,
        origsize=origsize,
        level=level,
        avbkey=key_path,
        forceelf=forceelf,
    )
    render_manifest(result)
    emit(f"[green]Done.[/green] Repacked in {base}")


def _resolve_avbkey(name: str, base: Path) -> Path | None:
    """Find an AVB key, accepting a bare name or a path, as repackimg.sh did.

    A usable key needs both halves of the pair; a lone `.pk8` is useless without its cert.
    """
    candidates = [Path(name), Path.cwd() / name, base / name]
    for candidate in candidates:
        # Join with "/" rather than with_name(): Path(".").name is "", and with_name("") raises
        # ValueError, which would escape as a traceback for `--avbkey .` or a trailing slash.
        stem = candidate.name
        parent = candidate.parent
        if (parent / f"{stem}.pk8").is_file() and any(parent.glob(f"{stem}.x509.*")):
            return parent / stem
    return None


def library_errors() -> tuple[type[BaseException], ...]:
    """Expected error types to catch at the top level.

    ``aik_py.errors`` is owned by another unit and may not exist yet; fall back to our own
    ``AikError`` so this stays independently mergeable.
    """
    try:
        errors = importlib.import_module("aik_py.errors")
    except ImportError:
        return (AikError,)
    base = getattr(errors, "AikError", None)
    return (AikError, base) if isinstance(base, type) else (AikError,)


def report_error(exc: BaseException) -> int:
    """Print an expected failure to stderr and return the exit status.

    Errors go to stderr, matching click's own convention for failure output. rich Console
    instances bind their stream at construction, so a module-level one created before a test
    runner swapped ``sys.stderr`` would write to the wrong place -- look the stream up now.
    """
    Console(stderr=True).print(f"[red]Error:[/red] {exc}")
    return 1


def main(argv: Iterable[str] | None = None) -> int:
    """Console-script entry point: run the CLI, mapping expected errors to a nonzero exit.

    Genuinely unexpected exceptions are left to propagate so their traceback stays intact.
    """
    try:
        cli.main(args=list(argv) if argv is not None else None, standalone_mode=False)
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Abort:
        Console(stderr=True).print("[red]Aborted.[/red]")
        return 130
    except library_errors() as exc:
        return report_error(exc)
    except KeyboardInterrupt:
        Console(stderr=True).print("[red]Interrupted.[/red]")
        return 130
    return 0
