"""Tests for the pure-Python ``newc`` cpio reader/writer.

Byte-for-byte comparisons are only meaningful with a pinned ``mtime``: cpio
embeds it in every entry, so archives built from a live filesystem differ on
every run. Fixtures here set ``mtime=0`` unless the test is specifically
exercising mtime preservation.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess

import pytest

from aik_py.cpio import (
    HEADER_SIZE,
    CpioEntry,
    CpioError,
    read_archive,
    write_archive,
)

# --------------------------------------------------------------------------
# helpers


def build(entries: list[CpioEntry]) -> bytes:
    """Canonical archive bytes for ``entries`` (no GNU 512-byte tail pad)."""
    return write_archive(entries)


def hand_header(
    *,
    magic: str = "070701",
    ino: int = 0,
    mode: int = 0,
    uid: int = 0,
    gid: int = 0,
    nlink: int = 1,
    mtime: int = 0,
    filesize: int = 0,
    devmajor: int = 0,
    devminor: int = 0,
    rdevmajor: int = 0,
    rdevminor: int = 0,
    namesize: int = 0,
    check: int = 0,
) -> bytes:
    """Assemble a 110-byte newc header by hand.

    Written longhand rather than via the module so tests are not tautological
    -- this is an independent statement of the on-disk layout.
    """
    return (
        magic.encode("ascii")
        + f"{ino:08X}".encode()
        + f"{mode:08X}".encode()
        + f"{uid:08X}".encode()
        + f"{gid:08X}".encode()
        + f"{nlink:08X}".encode()
        + f"{mtime:08X}".encode()
        + f"{filesize:08X}".encode()
        + f"{devmajor:08X}".encode()
        + f"{devminor:08X}".encode()
        + f"{rdevmajor:08X}".encode()
        + f"{rdevminor:08X}".encode()
        + f"{namesize:08X}".encode()
        + f"{check:08X}".encode()
    )


def pad4(n: int) -> bytes:
    return b"\0" * (-n % 4)


def hand_entry(
    name: str,
    data: bytes,
    *,
    mode: int = 0o100644,
    ino: int = 0,
    nlink: int = 1,
) -> bytes:
    name_raw = name.encode() + b"\0"
    return (
        hand_header(
            mode=mode, ino=ino, nlink=nlink, namesize=len(name_raw), filesize=len(data)
        )
        + name_raw
        + pad4(HEADER_SIZE + len(name_raw))
        + data
        + pad4(len(data))
    )


def hand_trailer() -> bytes:
    # Per spec the trailer is all-zero except c_namesize. (GNU cpio writes
    # nlink=1 here; readers must accept both, and we assert our own convention.)
    name_raw = b"TRAILER!!!\0"
    return (
        hand_header(nlink=0, namesize=len(name_raw))
        + name_raw
        + pad4(HEADER_SIZE + len(name_raw))
    )


# --------------------------------------------------------------------------
# round trips


def test_roundtrip_regular_dir_symlink_is_byte_identical():
    entries = [
        CpioEntry(
            name="init", data=b"#!/bin/sh\nexit 0\n", mode=stat.S_IFREG | 0o755, mtime=0
        ),
        CpioEntry(name="etc", data=b"", mode=stat.S_IFDIR | 0o755, mtime=0),
        CpioEntry(
            name="etc/hosts",
            data=b"127.0.0.1 localhost\n",
            mode=stat.S_IFREG | 0o644,
            mtime=0,
        ),
        CpioEntry(
            name="busybox", data=b"/bin/busybox", mode=stat.S_IFLNK | 0o777, mtime=0
        ),
    ]
    blob = build(entries)
    assert write_archive(read_archive(blob)) == blob

    back = read_archive(blob)
    assert [e.name for e in back] == ["init", "etc", "etc/hosts", "busybox"]
    assert back[0].data == b"#!/bin/sh\nexit 0\n"
    assert back[1].is_dir and back[1].data == b""
    assert back[3].is_symlink and back[3].data == b"/bin/busybox"


def test_roundtrip_preserves_mtime_uid_gid_mode():
    entries = [
        CpioEntry(
            name="a",
            data=b"x",
            mode=stat.S_IFREG | 0o600,
            uid=1000,
            gid=1001,
            mtime=0x5F5E100,
            ino=42,
            nlink=3,
        ),
        CpioEntry(
            name="d",
            data=b"",
            mode=stat.S_IFDIR | 0o750,
            uid=0,
            gid=0,
            mtime=0x5F5E100,
            ino=43,
            nlink=2,
        ),
        CpioEntry(
            name="l",
            data=b"target",
            mode=stat.S_IFLNK | 0o777,
            uid=7,
            gid=8,
            mtime=0x5F5E100,
            rdevmajor=9,
            rdevminor=10,
        ),
    ]
    back = read_archive(build(entries))
    assert back == entries


def test_mtime_is_preserved_exactly():
    # Large but valid 32-bit timestamp; guards against accidental truncation.
    entries = [
        CpioEntry(name="f", data=b"", mode=stat.S_IFREG | 0o644, mtime=0xFFFFFFFF)
    ]
    assert read_archive(build(entries))[0].mtime == 0xFFFFFFFF


def test_defaults_are_root_owned():
    entry = CpioEntry(name="f", data=b"")
    assert entry.uid == 0 and entry.gid == 0


def test_directory_entries_are_preserved_not_synthesized():
    entries = [CpioEntry(name="d", data=b"", mode=stat.S_IFDIR | 0o755, mtime=0)]
    back = read_archive(build(entries))
    assert len(back) == 1
    assert back[0].is_dir


def test_directory_payload_is_dropped_on_write():
    # A directory with stray payload must serialise with c_filesize == 0.
    blob = build(
        [CpioEntry(name="d", data=b"junk", mode=stat.S_IFDIR | 0o755, mtime=0)]
    )
    assert read_archive(blob) == [
        CpioEntry(name="d", data=b"", mode=stat.S_IFDIR | 0o755)
    ]


# --------------------------------------------------------------------------
# hardlinks
#
# newc records each extra name for an inode as its own header sharing c_ino and
# c_nlink but with c_filesize == 0. The reader must resolve those groups or a
# caller materialising entries one at a time truncates every link but the first.


def test_hardlinked_names_both_carry_the_payload():
    blob = (
        hand_entry("init", b"IMPORTANT DATA", ino=42, nlink=2)
        + hand_entry("init_link", b"", ino=42, nlink=2)
        + hand_trailer()
    )
    init, link = read_archive(blob)
    assert init.data == b"IMPORTANT DATA"
    # Regression: the second link used to come back as a 0-byte file, which
    # silently drops the data on an unpack/repack round trip.
    assert link.data == b"IMPORTANT DATA"


def test_hardlinked_names_share_metadata():
    blob = (
        hand_entry("init", b"DATA", ino=42, nlink=2)
        + hand_entry("init_link", b"", ino=42, nlink=2)
        + hand_trailer()
    )
    _, link = read_archive(blob)
    assert link.ino == 42 and link.nlink == 2


def test_hardlink_group_with_three_names():
    blob = (
        hand_entry("a", b"PAYLOAD", ino=7, nlink=3)
        + hand_entry("b", b"", ino=7, nlink=3)
        + hand_entry("c", b"", ino=7, nlink=3)
        + hand_trailer()
    )
    assert [e.data for e in read_archive(blob)] == [b"PAYLOAD"] * 3


def test_payload_after_the_empty_link_is_resolved():
    # Link group where the empty header appears first in the stream.
    blob = (
        hand_entry("b", b"", ino=9, nlink=2)
        + hand_entry("a", b"PAYLOAD", ino=9, nlink=2)
        + hand_trailer()
    )
    assert [e.data for e in read_archive(blob)] == [b"PAYLOAD", b"PAYLOAD"]


def test_directories_are_not_used_as_hardlink_payload_sources():
    # Directories share an inode across '.' and its subdirs in GNU output, but
    # must never adopt another entry's payload.
    blob = (
        hand_entry(".", b"", mode=stat.S_IFDIR | 0o755, ino=1, nlink=2)
        + hand_entry("sub", b"", mode=stat.S_IFDIR | 0o755, ino=1, nlink=2)
        + hand_trailer()
    )
    assert [e.data for e in read_archive(blob)] == [b"", b""]


def test_empty_regular_file_with_nlink_two_stays_empty():
    blob = (
        hand_entry("a", b"", ino=3, nlink=2)
        + hand_entry("b", b"", ino=3, nlink=2)
        + hand_trailer()
    )
    assert [e.data for e in read_archive(blob)] == [b"", b""]


def test_distinct_inodes_are_not_merged():
    blob = (
        hand_entry("a", b"AAA", ino=1, nlink=2)
        + hand_entry("b", b"", ino=2, nlink=2)
        + hand_trailer()
    )
    assert [e.data for e in read_archive(blob)] == [b"AAA", b""]


# --------------------------------------------------------------------------
# padding


@pytest.mark.parametrize("name_len", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("data_len", [0, 1, 2, 3, 4, 5, 6, 7])
def test_padding_is_4_byte_aligned_for_all_name_and_data_lengths(name_len, data_len):
    # name_len counts bytes incl. NUL; data_len forces every payload alignment.
    name = "n" * (name_len - 1)
    data = b"d" * data_len
    entry = CpioEntry(name=name, data=data, mode=stat.S_IFREG | 0o644, mtime=0)
    blob = build([entry])

    # Header (110) + namesize, padded, then payload, padded -> whole archive
    # is a multiple of 4 because every header must start 4-aligned.
    assert len(blob) % 4 == 0
    assert read_archive(blob) == [entry]


def test_name_padding_pad_counts_are_correct():
    # 110 + namesize must be rounded up to the next multiple of 4.
    # namesize 5 -> 115 -> 1 pad byte.
    blob = build([CpioEntry(name="init", data=b"", mode=stat.S_IFREG | 0o644, mtime=0)])
    assert blob[110:115] == b"init\0"
    assert blob[115] == 0  # exactly one NUL pad
    assert blob[116:122] == b"070701"  # next header/trailer starts right after


def test_odd_length_payload_is_padded():
    entry = CpioEntry(name="f", data=b"abc", mode=stat.S_IFREG | 0o644, mtime=0)
    blob = build([entry])
    ns = len("f") + 1
    data_start = HEADER_SIZE + ns + len(pad4(HEADER_SIZE + ns))
    assert blob[data_start : data_start + 3] == b"abc"
    # 3 payload bytes need 1 pad byte to reach a 4-byte boundary.
    assert blob[data_start + 3] == 0


def test_odd_length_name_and_payload_together():
    entry = CpioEntry(name="abc", data=b"xyz", mode=stat.S_IFREG | 0o644, mtime=0)
    blob = build([entry])
    assert read_archive(blob) == [entry]
    assert write_archive(read_archive(blob)) == blob


def test_many_entries_stay_aligned():
    entries = [
        CpioEntry(name=f"f{i}", data=b"z" * (i % 5), mode=stat.S_IFREG | 0o644, mtime=0)
        for i in range(20)
    ]
    blob = build(entries)
    assert len(blob) % 4 == 0
    assert read_archive(blob) == entries


# --------------------------------------------------------------------------
# empty archive


def test_empty_archive_is_just_trailer():
    blob = write_archive([])
    assert blob == hand_trailer()
    assert read_archive(blob) == []
    assert len(blob) % 4 == 0


def test_trailer_namesize_is_eleven():
    blob = write_archive([])
    assert int(blob[94:102], 16) == 11
    assert blob[110:121] == b"TRAILER!!!\0"


# --------------------------------------------------------------------------
# security: absolute paths and traversal


def test_absolute_path_is_stripped_on_extract():
    blob = hand_entry("/etc/passwd", b"root:x:0:0\n") + hand_trailer()
    (entry,) = read_archive(blob)
    assert entry.name == "etc/passwd"


def test_leading_slashes_are_all_stripped():
    blob = hand_entry("///etc/shadow", b"") + hand_trailer()
    (entry,) = read_archive(blob)
    assert entry.name == "etc/shadow"


def test_double_slash_and_dot_segments_are_normalised():
    blob = hand_entry("./etc//hosts", b"") + hand_trailer()
    (entry,) = read_archive(blob)
    assert entry.name == "etc/hosts"


@pytest.mark.parametrize(
    "bad", ["../escape", "a/../../escape", "..", "etc/../../x", "a/b/../../../c"]
)
def test_dotdot_traversal_raises(bad):
    blob = hand_entry(bad, b"") + hand_trailer()
    with pytest.raises(CpioError, match="escapes the extraction root"):
        read_archive(blob)


def test_interior_nul_in_name_raises():
    # Only the final byte is the terminator, so an interior NUL would reach
    # os.path.join() and raise a bare ValueError downstream.
    name = b"ev\x00il"
    raw = hand_header(namesize=len(name) + 1) + name + b"\0"
    blob = raw + pad4(HEADER_SIZE + len(name) + 1) + hand_trailer()
    with pytest.raises(CpioError, match="interior NUL"):
        read_archive(blob)


def test_reserved_trailer_member_name_is_rejected_on_write():
    with pytest.raises(CpioError, match="reserved"):
        write_archive([CpioEntry(name="TRAILER!!!", data=b"x", mtime=0)])


def test_reserved_trailer_member_is_read_as_a_normal_entry():
    # A member named TRAILER!!! that carries data is not the end-of-archive
    # marker, so it must not truncate the archive.
    blob = (
        hand_entry("a", b"A")
        + hand_entry("TRAILER!!!", b"payload", mode=stat.S_IFREG | 0o644)
        + hand_entry("b", b"B")
        + hand_trailer()
    )
    entries = read_archive(blob)
    assert [e.name for e in entries] == ["a", "TRAILER!!!", "b"]
    assert entries[1].data == b"payload"


def test_dot_entry_is_preserved_as_root():
    # GNU cpio emits "." first in every archive; it is not an escape.
    blob = hand_entry(".", b"", mode=stat.S_IFDIR | 0o755) + hand_trailer()
    (entry,) = read_archive(blob)
    assert entry.name == "."
    assert entry.is_dir


def test_sanitize_can_be_disabled_for_exact_rewriting():
    blob = hand_entry("/etc/passwd", b"") + hand_trailer()
    (entry,) = read_archive(blob, sanitize=False)
    assert entry.name == "/etc/passwd"


def test_normalised_names_do_not_round_trip_byte_identically():
    # Documented consequence of sanitising: reading rewrites the name, so a
    # round trip returns an equivalent archive, not the same bytes.
    blob = hand_entry("./etc/hosts", b"data") + hand_trailer()
    assert read_archive(blob)[0].name == "etc/hosts"
    assert write_archive(read_archive(blob)) != blob


# --------------------------------------------------------------------------
# malformed input


def test_bad_magic_raises():
    blob = b"070702" + hand_header()[6:] + hand_trailer()
    with pytest.raises(CpioError, match="bad cpio magic"):
        read_archive(blob)


def test_empty_input_raises():
    with pytest.raises(CpioError, match="truncated"):
        read_archive(b"")


def test_truncated_header_raises():
    blob = build([CpioEntry(name="f", data=b"data", mtime=0)])
    with pytest.raises(CpioError, match="truncated"):
        read_archive(blob[:80])


def test_truncated_payload_raises():
    blob = build([CpioEntry(name="f", data=b"payload", mtime=0)])
    # Chop into the final entry's payload; keep a full header.
    with pytest.raises(CpioError):
        read_archive(blob[:-4])


def test_namesize_past_buffer_raises():
    raw = hand_entry("f", b"")
    # Forge an absurd c_namesize on the first header.
    forged = raw[:94] + b"FFFFFFF0" + raw[102:]
    with pytest.raises(CpioError, match="runs past the archive"):
        read_archive(forged + hand_trailer())


def test_zero_namesize_raises():
    raw = hand_entry("f", b"")
    forged = raw[:94] + b"00000000" + raw[102:]
    with pytest.raises(CpioError, match="c_namesize"):
        read_archive(forged + hand_trailer())


def test_non_hex_field_raises():
    raw = hand_entry("f", b"")
    bad = raw[:54] + b"ZZZZZZZZ" + raw[62:]  # corrupt c_filesize
    with pytest.raises(CpioError, match="not valid hex"):
        read_archive(bad + hand_trailer())


def test_lowercase_hex_is_accepted():
    # GNU cpio writes and accepts either case; real ramdisks contain it.
    payload = b"\0" * 10
    raw = hand_entry("f", payload)
    bad = raw[:54] + b"0000000a" + raw[62:]  # lowercase c_filesize = 10
    (entry,) = read_archive(bad + hand_trailer())
    assert entry.data == payload


def test_lowercase_hex_mtime_is_accepted():
    raw = hand_entry("f", b"")
    bad = raw[:46] + b"05f5e100" + raw[54:]
    (entry,) = read_archive(bad + hand_trailer())
    assert entry.mtime == 0x5F5E100


def test_unterminated_name_raises():
    raw = hand_entry("f", b"")
    # Overwrite the name's trailing NUL with a real byte (inside the name block).
    ns = len(b"f\0")
    pad = pad4(HEADER_SIZE + ns)
    name_start = HEADER_SIZE + ns + len(pad) - 1  # point at the NUL
    bad = raw[:name_start] + b"x" + raw[name_start + 1 :]
    with pytest.raises(CpioError, match="not NUL-terminated"):
        read_archive(bad + hand_trailer())


def test_missing_trailer_raises():
    # An archive that ends after its last member, with no TRAILER!!! entry.
    blob = hand_entry("f", b"data")
    with pytest.raises(CpioError, match="truncated"):
        read_archive(blob)


def test_trailing_garbage_after_trailer_is_ignored():
    blob = build([]) + b"\0" * 512
    assert read_archive(blob) == []


def test_negative_mode_is_rejected_on_construction():
    with pytest.raises(CpioError, match="out of range"):
        CpioEntry(name="f", data=b"", mode=-1)


def test_negative_mode_does_not_leak_overflowerror():
    # stat.S_ISDIR(-1) raises a bare OverflowError; construction must reject it
    # first so callers only ever need to catch CpioError.
    with pytest.raises(CpioError):
        CpioEntry(name="f", data=b"", mode=-1)


def test_oversized_field_is_rejected_on_write():
    entry = CpioEntry(name="f", data=b"", uid=0x1_0000_0000)
    with pytest.raises(CpioError, match="out of range"):
        write_archive([entry])


def test_non_ascii_name_survives_roundtrip():
    entry = CpioEntry(name="café", data=b"", mode=stat.S_IFREG | 0o644, mtime=0)
    assert read_archive(build([entry])) == [entry]


def test_error_is_a_corrupt_image_error():
    # Callers upstream catch CorruptImageError; CpioError must be a subclass.
    try:
        from aik_py.errors import CorruptImageError
    except ImportError:  # pragma: no cover - before unit 1 merges
        pytest.skip("aik_py.errors not available yet")
    assert issubclass(CpioError, CorruptImageError)


# --------------------------------------------------------------------------
# independent oracle: real GNU cpio


@pytest.mark.skipif(shutil.which("cpio") is None, reason="GNU cpio not installed")
def test_gnu_cpio_extracts_our_archive(tmp_path):
    entries = [
        CpioEntry(
            name="init", data=b"#!/bin/sh\nexit 0\n", mode=stat.S_IFREG | 0o755, mtime=0
        ),
        CpioEntry(name="etc", data=b"", mode=stat.S_IFDIR | 0o755, mtime=0),
        CpioEntry(
            name="etc/hosts",
            data=b"127.0.0.1 localhost\n",
            mode=stat.S_IFREG | 0o644,
            mtime=0,
        ),
        CpioEntry(
            name="busybox", data=b"/bin/busybox", mode=stat.S_IFLNK | 0o777, mtime=0
        ),
    ]
    archive = tmp_path / "rd.cpio"
    archive.write_bytes(build(entries))

    out = tmp_path / "out"
    out.mkdir()
    subprocess.run(
        ["cpio", "-idm"],
        cwd=out,
        stdin=archive.open("rb"),
        check=True,
        capture_output=True,
    )

    init = out / "init"
    assert init.is_file() and stat.S_IMODE(init.lstat().st_mode) == 0o755
    assert init.read_bytes() == b"#!/bin/sh\nexit 0\n"

    assert (out / "etc").is_dir()
    assert (out / "etc" / "hosts").read_bytes() == b"127.0.0.1 localhost\n"

    link = out / "busybox"
    assert link.is_symlink()
    assert os.readlink(link) == "/bin/busybox"


@pytest.mark.skipif(shutil.which("cpio") is None, reason="GNU cpio not installed")
def test_gnu_cpio_reads_our_archive(tmp_path):
    # The reverse direction: real cpio output parsed by our reader.
    src = tmp_path / "tree"
    (src / "etc").mkdir(parents=True)
    (src / "init").write_text("#!/bin/sh\nexit 0\n")
    (src / "etc" / "hosts").write_text("127.0.0.1 localhost\n")
    (src / "link").symlink_to("init")

    archive = tmp_path / "gnu.cpio"
    # cwd= plus a fixed argv keeps tmp_path out of any shell string.
    subprocess.run(
        ["sh", "-c", "find . | cpio -o -H newc 2>/dev/null"],
        cwd=src,
        check=True,
        stdout=archive.open("wb"),
    )

    entries = read_archive(archive.read_bytes())
    by_name = {e.name.lstrip("./"): e for e in entries}
    assert by_name["init"].data == b"#!/bin/sh\nexit 0\n"
    assert by_name["init"].is_file
    assert by_name["etc"].is_dir
    assert by_name["etc/hosts"].data == b"127.0.0.1 localhost\n"
    assert by_name["link"].is_symlink
    assert by_name["link"].data == b"init"


@pytest.mark.skipif(shutil.which("cpio") is None, reason="GNU cpio not installed")
def test_gnu_cpio_hardlinks_are_resolved(tmp_path):
    # GNU cpio emits the second link name with c_filesize == 0; confirm against
    # the real thing that we hand both names the payload.
    src = tmp_path / "tree"
    src.mkdir()
    (src / "init").write_bytes(b"IMPORTANT DATA")
    (src / "init_link").hardlink_to(src / "init")

    archive = tmp_path / "gnu.cpio"
    subprocess.run(
        ["sh", "-c", "find . | cpio -o -H newc 2>/dev/null"],
        cwd=src,
        check=True,
        stdout=archive.open("wb"),
    )

    by_name = {e.name.lstrip("./"): e for e in read_archive(archive.read_bytes())}
    assert by_name["init"].data == b"IMPORTANT DATA"
    assert by_name["init_link"].data == b"IMPORTANT DATA"
    assert by_name["init"].ino == by_name["init_link"].ino
    assert by_name["init"].nlink == 2
