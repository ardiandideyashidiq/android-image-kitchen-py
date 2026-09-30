"""Tests for the aik-py command line interface."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from dataclasses import dataclass
from pathlib import Path

import pytest

from aik_py import cli as cli_mod
from aik_py.cli import main

# rich-click wraps click and re-exports its Command/Group types, so click's runner drives it
# unchanged. rich_click.testing is not importable on any released rich-click (1.9.9 is latest
# and ships no such module), so fall back to click's own runner; if rich-click ever adds a
# runner of its own, prefer it.
try:  # pragma: no cover - depends on the installed rich-click
    from rich_click.testing import CliRunner
except ImportError:  # pragma: no cover
    from click.testing import CliRunner

BACKENDS = ("session", "repack", "manifest")


def have_backends() -> bool:
    """True when every back end module owned by another unit is present."""
    return all(
        importlib.util.find_spec(f"aik_py.{name}") is not None for name in BACKENDS
    )


requires_backends = pytest.mark.skipif(
    not have_backends(), reason="unpack/repack back end modules are not implemented yet"
)


def _populate(base: Path, *, ramdisk: bool = True) -> None:
    """Lay out the files a repack expects: a populated split_img/ and optionally ramdisk/."""
    (base / "split_img" / "boot-kernel").write_bytes(b"k")
    (base / "split_img" / "boot-ramdiskcomp").write_text("gzip")
    if ramdisk:
        (base / "ramdisk").mkdir(exist_ok=True)
        (base / "ramdisk" / "init").write_bytes(b"i")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@dataclass(frozen=True)
class Invocation:
    exit_code: int
    output: str


def _invoke(args: list[str]) -> Invocation:
    """Run the CLI through main() so the top-level error handler is exercised.

    runner.invoke() would surface AikError as a caught exception instead of the clean
    "Error: ..." message a user actually sees, so error tests go through main().
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = main(args)
    return Invocation(exit_code=code, output=buffer.getvalue())


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """An isolated working directory, and a tool_dir() that resolves to it.

    Commands default to the AIK installation directory; point it at tmp_path so tests never
    delete anything in the checkout.
    """
    (tmp_path / "bin").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project]\nname='aik-py'\n")
    (tmp_path / "split_img").mkdir()
    return tmp_path


@pytest.fixture
def in_workspace(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cli_mod, "tool_dir", lambda: workspace)
    monkeypatch.chdir(workspace)
    return workspace


@pytest.fixture(autouse=True)
def _isolate_logging():
    """Keep loguru handlers from leaking between tests."""
    yield
    cli_mod.configure_logging(verbose=False, quiet=False)


# --- help exits 0 -----------------------------------------------------------------


@pytest.mark.parametrize("args", [["--help"], ["-h"]])
def test_group_help_exits_zero(runner: CliRunner, args: list[str]) -> None:
    result = runner.invoke(cli_mod.cli, args)
    assert result.exit_code == 0
    assert "Usage" in result.output
    for command in ("unpack", "repack", "clean"):
        assert command in result.output


@pytest.mark.parametrize("command", ["unpack", "repack", "clean"])
def test_subcommand_help_exits_zero(runner: CliRunner, command: str) -> None:
    result = runner.invoke(cli_mod.cli, [command, "--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_main_help_exits_zero() -> None:
    assert main(["--help"]) == 0


# --- flag ordering ----------------------------------------------------------------


@pytest.mark.parametrize("command", ["unpack", "repack", "clean"])
def test_local_is_position_independent(runner: CliRunner, command: str) -> None:
    """unpackimg.sh parsed --local in one case block and --nosudo in a second, so
    `--nosudo --local img` silently dropped --local. A normal flag parser has no such trap.
    """
    # --help wins over everything, so use it to prove both orders parse without a usage error.
    for args in ([command, "--local", "--help"], [command, "--help", "--local"]):
        result = runner.invoke(cli_mod.cli, args)
        assert result.exit_code == 0, args


def test_nosudo_and_local_together_do_not_drop_local(in_workspace: Path) -> None:
    result = _invoke(["unpack", "--nosudo", "--local"])
    # Reaches the back end (missing on this branch) rather than failing flag parsing.
    assert result.exit_code == 1
    assert "no such option" not in result.output.lower()


# --- unpack ----------------------------------------------------------------------


def test_unpack_missing_path_exits_nonzero_without_traceback(
    in_workspace: Path,
) -> None:
    (in_workspace / "image-new.img").write_bytes(b"")
    result = _invoke(["unpack", "does-not-exist.img"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "does-not-exist.img" in result.output


def test_unpack_no_image_found_is_an_error(in_workspace: Path) -> None:
    result = _invoke(["unpack"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_auto_pick_skips_aboot(runner: CliRunner, in_workspace: Path) -> None:
    (in_workspace / "aboot.img").write_bytes(b"a")
    (in_workspace / "boot.img").write_bytes(b"b")
    picked = cli_mod.pick_image(in_workspace)
    assert picked is not None
    assert picked.name == "boot.img"


@pytest.mark.parametrize(
    "skipped", ["aboot.img", "image-new.img", "unlokied-new.img", "unsigned-new.img"]
)
def test_auto_pick_skips_own_output(skipped: str, tmp_path: Path) -> None:
    (tmp_path / skipped).write_bytes(b"a")
    (tmp_path / "recovery.img").write_bytes(b"b")
    picked = cli_mod.pick_image(tmp_path)
    assert picked is not None
    assert picked.name == "recovery.img"


def test_auto_pick_sorts_across_patterns(tmp_path: Path) -> None:
    """`ls *.elf *.img *.sin` sorted the whole expansion together, so aaa.img beat zzz.elf."""
    (tmp_path / "zzz.elf").write_bytes(b"e")
    (tmp_path / "aaa.img").write_bytes(b"i")
    picked = cli_mod.pick_image(tmp_path)
    assert picked is not None
    assert picked.name == "aaa.img"


def test_auto_pick_sorts_elf_and_img_together(tmp_path: Path) -> None:
    (tmp_path / "boot.img").write_bytes(b"b")
    (tmp_path / "zkernel.elf").write_bytes(b"e")
    picked = cli_mod.pick_image(tmp_path)
    assert picked is not None
    assert picked.name == "boot.img"


def test_auto_pick_handles_sin(tmp_path: Path) -> None:
    (tmp_path / "kernel.sin").write_bytes(b"s")
    picked = cli_mod.pick_image(tmp_path)
    assert picked is not None
    assert picked.name == "kernel.sin"


def test_auto_pick_returns_none_when_empty(tmp_path: Path) -> None:
    (tmp_path / "aboot.img").write_bytes(b"a")
    assert cli_mod.pick_image(tmp_path) is None


@pytest.mark.parametrize("command", ["unpack", "repack", "clean"])
@pytest.mark.parametrize("placement", ["before", "after"])
def test_verbose_accepted_on_either_side_of_the_subcommand(
    command: str, placement: str, in_workspace: Path
) -> None:
    args = [command, "--verbose"] if placement == "after" else ["--verbose", command]
    result = _invoke(args)
    # Never a usage error, whichever side the flag lands on.
    assert "no such option" not in result.output.lower()
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", ["unpack", "repack", "clean"])
@pytest.mark.parametrize("placement", ["before", "after"])
def test_quiet_accepted_on_either_side_of_the_subcommand(
    command: str, placement: str, in_workspace: Path
) -> None:
    args = [command, "--quiet"] if placement == "after" else ["--quiet", command]
    result = _invoke(args)
    assert "no such option" not in result.output.lower()
    assert "Traceback" not in result.output


def test_verbose_and_quiet_move_the_log_level(in_workspace: Path) -> None:
    from aik_py.cli import logger

    _invoke(["clean", "--verbose"])
    assert logger._core.min_level == 10  # DEBUG

    _invoke(["clean", "--quiet"])
    assert logger._core.min_level == 30  # WARNING

    _invoke(["clean"])
    assert logger._core.min_level == 20  # INFO


# --- clean -----------------------------------------------------------------------


SCRATCH_NAMES = [
    "image-new.img",
    "unsigned-new.img",
    "unlokied-new.img",
    "unamonet-new.img",
    "unpadded-new.img",
    "ramdisk-new.cpio.gz",
    "kernel-new.mtk",
    "boot-new.mtk",
]


@pytest.mark.parametrize("name", SCRATCH_NAMES)
def test_clean_removes_scratch_file(
    name: str, runner: CliRunner, in_workspace: Path
) -> None:
    (in_workspace / name).write_bytes(b"scratch")
    result = runner.invoke(cli_mod.cli, ["clean"])
    assert result.exit_code == 0
    assert not (in_workspace / name).exists()


def test_clean_removes_work_dirs_and_keeps_unrelated(
    runner: CliRunner, in_workspace: Path
) -> None:
    (in_workspace / "ramdisk").mkdir()
    (in_workspace / "ramdisk" / "init").write_bytes(b"x")
    (in_workspace / "split_img" / "boot-kernel").write_bytes(b"k")
    (in_workspace / "boot.img").write_bytes(b"keep")
    (in_workspace / "notes.txt").write_text("keep me")

    result = runner.invoke(cli_mod.cli, ["clean"])
    assert result.exit_code == 0
    assert not (in_workspace / "ramdisk").exists()
    assert not (in_workspace / "split_img").exists()
    assert (in_workspace / "boot.img").exists()
    assert (in_workspace / "notes.txt").exists()


def test_clean_reports_confirmation(runner: CliRunner, in_workspace: Path) -> None:
    result = runner.invoke(cli_mod.cli, ["clean"])
    assert "cleaned" in result.output.lower()


def test_clean_quiet_suppresses_confirmation(
    runner: CliRunner, in_workspace: Path
) -> None:
    (in_workspace / "image-new.img").write_bytes(b"")
    result = runner.invoke(cli_mod.cli, ["clean", "--quiet"])
    assert result.exit_code == 0
    assert "cleaned" not in result.output.lower()
    assert not (in_workspace / "image-new.img").exists()


def test_clean_on_empty_dir_succeeds(runner: CliRunner, in_workspace: Path) -> None:
    result = runner.invoke(cli_mod.cli, ["clean"])
    assert result.exit_code == 0


# --- repack ----------------------------------------------------------------------


@pytest.mark.parametrize("value", ["abc", "", "3.5", "-1", "10", "9x"])
def test_repack_rejects_bad_level(
    value: str, runner: CliRunner, in_workspace: Path
) -> None:
    result = runner.invoke(cli_mod.cli, ["repack", "--level", value])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_repack_rejects_empty_level(runner: CliRunner, in_workspace: Path) -> None:
    result = runner.invoke(cli_mod.cli, ["repack", "--level", ""])
    assert result.exit_code != 0


def test_repack_accepts_valid_levels(in_workspace: Path) -> None:
    _populate(in_workspace)
    for level in ("0", "9"):
        result = _invoke(["repack", "--level", level])
        # Reaches the guard and the back end rather than failing validation.
        assert "not a number" not in result.output.lower(), level
        assert "Traceback" not in result.output, level


def test_repack_without_work_dirs_is_an_error(in_workspace: Path) -> None:
    (in_workspace / "split_img").rmdir()
    result = _invoke(["repack"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_repack_rejects_empty_split_img(in_workspace: Path) -> None:
    """repackimg.sh wanted a *populated* split_img/, not merely an existing one."""
    result = _invoke(["repack"])
    assert result.exit_code != 0
    assert "no files found" in result.output.lower()
    assert "Traceback" not in result.output


def test_repack_rejects_empty_ramdisk(in_workspace: Path) -> None:
    _populate(in_workspace, ramdisk=False)
    result = _invoke(["repack"])
    assert result.exit_code != 0
    assert "no files found" in result.output.lower()
    assert "Traceback" not in result.output


def test_repack_original_allows_empty_ramdisk(in_workspace: Path) -> None:
    _populate(in_workspace, ramdisk=False)
    result = _invoke(["repack", "--original"])
    # Past the guard: either the back end ran, or it reported itself unimplemented.
    assert "no files found" not in result.output.lower()
    assert "Traceback" not in result.output


def test_repack_missing_avbkey_is_an_error(in_workspace: Path) -> None:
    _populate(in_workspace)
    result = _invoke(["repack", "--avbkey", "nosuchkey"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "nosuchkey" in result.output


def test_repack_avbkey_needs_both_halves(in_workspace: Path) -> None:
    _populate(in_workspace)
    (in_workspace / "halfkey.pk8").write_bytes(b"k")
    result = _invoke(["repack", "--avbkey", "halfkey"])
    assert result.exit_code != 0
    assert "halfkey" in result.output


def test_resolve_avbkey_requires_both_files(tmp_path: Path) -> None:
    (tmp_path / "fullkey.pk8").write_bytes(b"k")
    (tmp_path / "fullkey.x509.pem").write_bytes(b"c")
    assert cli_mod._resolve_avbkey("fullkey", tmp_path) == tmp_path / "fullkey"
    assert cli_mod._resolve_avbkey("absent", tmp_path) is None


@pytest.mark.parametrize("name", [".", "mykeys/", ".."])
def test_resolve_avbkey_survives_degenerate_names(name: str, tmp_path: Path) -> None:
    """with_name() raises ValueError on "." and trailing separators; that must not escape."""
    assert cli_mod._resolve_avbkey(name, tmp_path) is None


def test_repack_avbkey_degenerate_name_is_not_a_traceback(in_workspace: Path) -> None:
    (in_workspace / "split_img" / "boot-kernel").write_bytes(b"k")
    result = _invoke(["repack", "--avbkey", "."])
    assert result.exit_code == 1
    assert "Traceback" not in result.output


# --- regression: cleanup must not delete the image unpack was asked to read -----------


def test_unpack_preserves_an_explicitly_named_scratch_image(
    in_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`aik-py unpack image-new.img` is legitimate; cleaning first would delete the input."""
    victim = in_workspace / "image-new.img"
    victim.write_bytes(b"payload")
    seen: dict[str, bool] = {}

    class StubSession:
        @staticmethod
        def unpack(image: Path, **kwargs: object) -> None:
            seen["exists"] = image.exists()

    monkeypatch.setattr(cli_mod, "load_backend", lambda *a, **k: StubSession())
    result = _invoke(["unpack", "image-new.img"])
    assert result.exit_code == 0, result.output
    assert seen["exists"] is True
    assert victim.exists()


def test_unpack_still_cleans_other_scratch_files(
    in_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = in_workspace / "unpadded-new.img"
    stale.write_bytes(b"stale")
    (in_workspace / "image-new.img").write_bytes(b"payload")

    class StubSession:
        @staticmethod
        def unpack(image: Path, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(cli_mod, "load_backend", lambda *a, **k: StubSession())
    result = _invoke(["unpack", "image-new.img"])
    assert result.exit_code == 0, result.output
    assert not stale.exists()


def test_unpack_expands_tilde(
    in_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = in_workspace / "home"
    home.mkdir()
    target = home / "boot.img"
    target.write_bytes(b"img")
    monkeypatch.setenv("HOME", str(home))
    seen: dict[str, Path] = {}

    class StubSession:
        @staticmethod
        def unpack(image: Path, **kwargs: object) -> None:
            seen["path"] = image

    monkeypatch.setattr(cli_mod, "load_backend", lambda *a, **k: StubSession())
    result = _invoke(["unpack", "~/boot.img"])
    assert result.exit_code == 0, result.output
    assert seen["path"] == target


def test_unpack_rejects_a_directory(in_workspace: Path) -> None:
    result = _invoke(["unpack", "split_img"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "split_img" in result.output


# --- defensive imports -----------------------------------------------------------


def test_missing_backend_reports_not_implemented(
    runner: CliRunner, in_workspace: Path
) -> None:
    if have_backends():
        pytest.skip("back end modules are present on this branch")
    _populate(in_workspace)
    result = _invoke(["repack"])
    assert result.exit_code == 1
    assert "not implemented" in result.output.lower()
    assert "ImportError" not in result.output
    assert "Traceback" not in result.output


def test_load_backend_names_the_missing_module() -> None:
    if have_backends():
        pytest.skip("back end modules are present on this branch")
    with pytest.raises(cli_mod.NotImplementedYet) as excinfo:
        cli_mod.load_backend("session", feature="Unpacking")
    assert "aik_py.session" in str(excinfo.value)


@requires_backends
def test_real_unpack_round_trip(
    runner: CliRunner, in_workspace: Path, tmp_path: Path
) -> None:
    """Exercise unpack/repack for real when the back ends exist."""
    image = in_workspace / "boot.img"
    image.write_bytes(b"not-a-real-boot-image")
    result = runner.invoke(cli_mod.cli, ["unpack", str(image)])
    assert result.exit_code == 0, result.output
    assert (in_workspace / "split_img").is_dir()
    assert (in_workspace / "ramdisk").is_dir()
