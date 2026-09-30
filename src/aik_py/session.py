"""Unpack pipeline: turn a boot/recovery image into a session directory.

Mirrors ``unpackimg.sh`` step for step, but records the result in a typed
:class:`~aik_py.manifest.Manifest` instead of a pile of ``split_img/`` filename
suffixes.  The familiar layout is still produced -- ``split_img/`` and
``ramdisk/`` next to ``manifest.json`` -- so the legacy Bash scripts and existing
muscle memory keep working against a Python-produced session.

Why the sudo machinery is gone
------------------------------
The Bash decided whether to run ``cpio`` as root by checking who owns
``ramdisk/``::

    $sudo chown 0:0 ramdisk
    $unpackcmd ... | $sudo $cpio -i -d --no-absolute-filenames

and symmetrically on repack it passed ``-R 0:0`` when it could *not* use sudo.
That dance exists because ``cpio`` is an external program that defaults to
recording the *caller's* uid/gid, and the Android bootloader runs before any
userspace exists: there is no passwd database to look 0 up in, so an archive
holding e.g. uid 1000 is unpacked as garbage or rejected outright.

This port writes the cpio archive itself and always emits ``uid=0``/``gid=0``,
so ownership is correct by construction and the whole sudo branch is unnecessary.
There is consequently no ``--nosudo`` flag to honour: it would choose between two
paths that are now the same path.  The flag is omitted deliberately.

A second hazard disappears with the same change.  ``unpackimg.sh`` warns that
``cpio 2.13`` "may result in an unusable repack; downgrade to 2.12 to be safe" --
a data-corruption bug in one external tool version.  Nothing here shells out to
cpio, so that failure mode is not reachable.
"""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

from loguru import logger

from aik_py.manifest import (
    MANIFEST_NAME,
    Header,
    Manifest,
    OsVersion,
    PatchInfo,
    PayloadRefs,
    SignatureInfo,
)

__all__ = [
    "CompressionError",
    "Session",
    "UnpackError",
    "UnrecognizedImageError",
    "UnsupportedImageError",
]

TOOL_VERSION = "aik-py/0.1.0"

#: Extensions considered by the auto-pick when no image argument is given.
IMAGE_SUFFIXES: tuple[str, ...] = (".elf", ".img", ".sin")

#: ``unpackimg.sh`` auto-pick deliberately ignores these: three are its own
#: intermediate repack outputs and ``aboot.img`` is a donor blob consumed by the
#: Loki patcher, never the image being unpacked.  Picking any of them by accident
#: is what this list prevents.
AUTO_PICK_SKIP: frozenset[str] = frozenset(
    {"aboot.img", "image-new.img", "unlokied-new.img", "unsigned-new.img"}
)

#: Header magic and the image layouts the toolkit knows how to split.
AOSP_MAGIC = b"ANDROID!"
U_BOOT_MAGIC = b"\x27\x05\x19\x56"
OSIP_MAGIC = b"$OS$\x00\x00\x01"
KRNL_MAGIC = b"KRNL"
VNDR_BOOT_MAGIC = b"VNDRBOOT"
ELF_MAGIC = b"\x7fELF"
MTK_MAGIC = b"\x88\x16\x88\x58"
#: Offset of the ASCII type name inside the MTK header ("ROOTFS", "RECOVERY", ...).
#: Verified against the bundled ``mkmtkhdr`` output; the type does *not* start at
#: offset 0, which is where the 4-byte magic sits.
MTK_TYPE_OFFSET = 8
#: Size of the MTK header prepended to kernels and ramdisks.
MTK_HEADER_SIZE = 512
QCDT_MAGIC = b"QCDT"

#: Byte layout of the AOSP boot header.  v0 ends at 1632; v1 appends
#: ``header_version``/``os_version``/``header_size``.
_HDR_V0_SIZE = 1632
_HDR_V1_SIZE = 1648
_OFF_CMDLINE = 64
_OFF_NAME = 48
_OFF_ID = 576
_LEN_ID = 20
_OFF_EXTRA_CMDLINE = 608
_LEN_EXTRA_CMDLINE = 1024

#: mkbootimg's default ``--kernel_offset``.  Subtracting it from the absolute
#: kernel load address recovers the base the other addresses are relative to.
_DEFAULT_KERNEL_OFFSET = 0x8000

#: Head-signature probes: ``(offset, magic, type)``.  The ``type`` strings match
#: the ``-sigtype`` suffix the Bash wrote so a legacy repack still matches.
_HEAD_SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (0, b"-SIGNED-BY-SIGNBLOB-", "BLOB"),
    (0, b"CHROMEOS", "CHROMEOS"),
    (0, b"DHTB\x01\x00\x00", "DHTB"),
    (0, b"\x01\x00\x00\x00", "SINv1"),
    (0, b"\x02\x00\x00\x00", "SINv2"),
    (0, b"\x03SIN", "SINv3"),
    (64, b"Red\x20Loader", "NOOK"),
    (64, b"Green\x20Loader", "NOOK"),
    (64, b"Green\x20Recovery", "NOOK"),
    (64, b"eMMC\x20boot.img+secondloader", "NOOK"),
    (64, b"eMMC\x20recovery.img+secondloader", "NOOK"),
    (48, b"BauwksBoot", "NOOKTAB"),
)

#: The NOOK loader strings, sliced out of the table above so that detection and
#: stripping cannot drift apart: a mismatched slice width makes a comparison that
#: can never succeed, silently leaving the loader attached.
_NOOK_MAGICS: frozenset[bytes] = frozenset(
    magic for offset, magic, kind in _HEAD_SIGNATURES if kind == "NOOK"
)

#: Tail-footer probes: ``(needle, type)``.  AVB and the OEM footers are appended
#: rather than prepended, so they are found by scanning the tail.
_TAIL_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"AVBf", "AVBv2"),
    (b"SEANDROIDENFORCE", "SEAndroid"),
    (b"\x41\xa9\xe4\x67\x74\x4d\x1d\x1b\xa4\x29\xf2\xec\xea\x65\x52\x79", "Bump"),
)
_AVB_V1_NEEDLE = b"\x02\x01\x01\x30\x82"

#: Compression magic -> canonical name.  Mirrors the ``case`` in ``unpackimg.sh``
#: that maps whatever ``file(1)`` called the payload onto a decompressor.
_COMPRESSION_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x1f\x8b", "gzip"),
    (b"\x89LZO\x00\x0d\x0a\x1a\x0a", "lzop"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"BZh", "bzip2"),
    (b"\x5d\x00\x00", "lzma"),
    (b"\x04\x22\x4d\x18", "lz4"),
    (b"\x02\x21\x4c\x18", "lz4-l"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"070701", "cpio"),
    (b"070702", "cpio"),
    (b"070707", "cpio"),
)

#: Extension the Bash appended to ``<image>-ramdisk.cpio`` for each compressor.
COMPRESSION_EXTENSION: dict[str, str] = {
    "gzip": "gz",
    "lzop": "lzo",
    "xz": "xz",
    "bzip2": "bz2",
    "lzma": "lzma",
    "lz4": "lz4",
    "lz4-l": "lz4",
    "zstd": "zst",
    "cpio": "",
    "empty": "empty",
}

#: Formats the splitter knows how to handle, mirroring the whitelist at
#: ``unpackimg.sh:150``.  An image outside this set is *unsupported* -- we know
#: what it is and decline -- which is a different failure from *unrecognized*.
SUPPORTED_IMAGE_TYPES: tuple[str, ...] = (
    "AOSP",
    "AOSP_VNDR",
    "AOSP-PXA",
    "ELF",
    "KRNL",
    "OSIP",
    "U-Boot",
)


class UnpackError(Exception):
    """Base class for unpack failures."""


class UnrecognizedImageError(UnpackError):
    """No known container magic matched: we cannot tell what this file is."""


class UnsupportedImageError(UnpackError):
    """The container is recognised but outside the supported whitelist."""


class CompressionError(UnpackError):
    """The ramdisk payload uses a compression we cannot decode."""


def _read_cstring(data: bytes, offset: int, length: int) -> str:
    raw = data[offset : offset + length]
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _sony_version(data: bytes) -> str:
    """Map a Sony SIN container header onto its version name."""
    if data.startswith(b"\x01\x00\x00\x00"):
        return "SINv1"
    if data.startswith(b"\x02\x00\x00\x00"):
        return "SINv2"
    return "SINv3"


def _chromeos_unvmlinuz(data: bytes) -> bytes:
    """Extract the inner kernel from a ChromeOS vmlinuz container.

    The vmlinuz wrapper is a stub + header + the original kernel; the Bash
    delegated this to ``futility vbutil_kernel --get-vmlinuz``.  The kernel
    begins at the 64-bit offset stored just past the 0x2028 magic, so the
    payload can be located without the tool.
    """
    if len(data) < 0x3000 or not data.startswith(b"CHROMEOS"):
        raise UnpackError("Not a CHROMEOS vmlinuz container")
    # vmlinuz header: stub code, then kernel_off (u64) at 0x2000 + padding.
    try:
        kernel_offset = struct.unpack_from("<Q", data, 0x2038)[0]
    except struct.error as exc:  # pragma: no cover - guarded by length check
        raise UnpackError("truncated CHROMEOS vmlinuz header") from exc
    if not 0 < kernel_offset < len(data):
        raise UnpackError(f"implausible CHROMEOS kernel offset {kernel_offset:#x}")
    return data[kernel_offset:]


def detect_compression(head: bytes) -> str:
    """Classify a ramdisk payload by its leading magic bytes.

    ``file(1)`` plus a magic file is how the Bash did this, which meant the
    result depended on the host's ``file`` version and on a 1900-line magic
    database.  A handful of byte prefixes is both faster and deterministic.
    """
    for magic, name in _COMPRESSION_MAGIC:
        if head.startswith(magic):
            return name
    if not head:
        return "empty"
    return "data"


def detect_head_signature(head: bytes) -> str | None:
    """Return the OEM head-signature type, or ``None`` if the image is bare."""
    for offset, magic, name in _HEAD_SIGNATURES:
        if head[offset : offset + len(magic)] == magic:
            return name
    return None


def detect_tail_type(data: bytes) -> str | None:
    """Detect an appended AVB / Bump / SEAndroid footer.

    AVBv1 has no fixed magic at the very end -- the footer's own bytes begin with
    an ASN.1 SEQUENCE header -- so the partition name is matched instead, which
    is what lets the partition name be recorded for re-signing.
    """
    tail = data[-8192:] if len(data) >= 8192 else data
    for needle, name in _TAIL_SIGNATURES:
        if needle in tail:
            return name
    if _AVB_V1_NEEDLE in tail and (b"/boot" in tail or b"/recovery" in tail):
        return "AVBv1"
    return None


def _avb_partition_from_tail(data: bytes) -> str | None:
    """Pull ``boot`` / ``recovery`` out of an AVBv1 footer's partition name."""
    tail = data[-8192:] if len(data) >= 8192 else data
    for name in (b"boot", b"recovery", b"vendor_boot"):
        if b"/" + name in tail:
            return name.decode()
    return None


def _detect_image_type(head: bytes) -> str:
    """Classify the container, mirroring ``unpackimg.sh`` steps 4-6.

    Order matters: the AMONET and LOKI wrappers sit *at* the AOSP magic offset,
    so they must be recognised before the plain AOSP test, and the SIN signature
    is stripped before we ever get here.
    """
    if head.startswith(U_BOOT_MAGIC):
        return "U-Boot"
    if head.startswith(OSIP_MAGIC):
        return "OSIP"
    if head.startswith(KRNL_MAGIC):
        return "KRNL"
    if head.startswith(VNDR_BOOT_MAGIC):
        return "AOSP_VNDR"

    # PXA is an AOSP variant distinguished by a tag word after the magic.
    if head.startswith(AOSP_MAGIC):
        if head[36:40] in (
            b"\x00\x00\x00\x02",
            b"\x00\x00\x80\x02",
            b"\x00\x00\x00\x03",
        ):
            return "AOSP-PXA"
        return "AOSP"

    if head.startswith(ELF_MAGIC):
        return "ELF"

    # Headerless forms: the magic lives at a fixed offset deeper in the file.
    if len(head) >= 1028 and head[1024:1032] in (AOSP_MAGIC, b"LOKI"):
        return "AOSP"
    if len(head) >= 1024 and b"microloader" in head[:2048]:
        return "AOSP"
    if len(head) >= 32 and head[28:32] == b"\xfc\xfa\xbc\x00":
        return "OSIP"
    return ""


def parse_aosp_header(
    data: bytes, magic: bytes = AOSP_MAGIC
) -> tuple[Header, int | None, int]:
    """Decode an AOSP boot header.

    Returns ``(header, header_version, header_size)``.  ``magic`` is the leading
    container signature the layout starts with -- ``ANDROID!`` normally, but
    ``VNDRBOOT`` for vendor_boot images, which are otherwise identical.

    v0 stores the OS version as two bytes at offsets 40 (version) and 41 (patch
    level).  v1 and later keep those bytes *and* append a packed ``os_version``
    word with a security-patch date, so both are recorded rather than flattened
    into the single opaque ``-os_version`` value the Bash wrote.
    """
    if not data.startswith(magic):
        raise UnrecognizedImageError(f"buffer does not start with the {magic!r} magic")

    kernel_addr = _u32(data, 12)
    header = Header(
        kernel_addr=kernel_addr,
        ramdisk_addr=_u32(data, 20),
        second_addr=_u32(data, 28),
        tags_addr=_u32(data, 32),
        page_size=_u32(data, 36),
        cmdline=_read_cstring(data, _OFF_CMDLINE, 512) or None,
        name=_read_cstring(data, _OFF_NAME, 16) or None,
        id_hex=data[_OFF_ID : _OFF_ID + _LEN_ID].hex() or None,
        extra_cmdline=_read_cstring(data, _OFF_EXTRA_CMDLINE, _LEN_EXTRA_CMDLINE)
        or None,
        os_version=OsVersion(os_version=data[40], patch_level=data[41]),
    )
    # ``--board`` and the header's ``name`` field are the same string in every
    # image mkbootimg writes, so one populates the other.
    header.board = header.name

    # The header has no ``base`` field: mkbootimg folds it into the absolute
    # addresses, so recovering it means assuming the AOSP default kernel offset
    # of 0x8000 and subtracting.  This is the same inversion ``unpackbootimg``
    # performs, and it is exact for every image built with a stock offset; a
    # device using a custom one is recorded as-is and is the repack's problem.
    header.base = kernel_addr - _DEFAULT_KERNEL_OFFSET
    header.kernel_offset = kernel_addr - header.base
    header.ramdisk_offset = (header.ramdisk_addr or 0) - header.base
    header.second_offset = (header.second_addr or 0) - header.base
    header.tags_offset = (header.tags_addr or 0) - header.base

    version: int | None = None
    size = _HDR_V0_SIZE
    if len(data) >= _HDR_V1_SIZE:
        version, size = _probe_extended_header(data)
        if version is not None:
            header.os_version = _decode_v1_os_version(_u32(data, 1636))
    return header, version, size


def _probe_extended_header(data: bytes) -> tuple[int | None, int]:
    """Locate ``(header_version, header_size)`` for a v1+ header.

    Three layouts exist in the wild and they are distinguished by where a
    *plausible header size* sits, not merely by which version byte is non-zero:

    * canonical v1 -- version@1632, os_version@1636, header_size@1640
    * v2 -- same, plus recovery_dtbo_size@1644 and recovery_dtbo_offset@1648,
      with header_size pushed to 1656
    * the build bundled with this repo -- version and os_version left zero, and
      only header_size written, 4-byte-aligned, at 1644

    A header size is only trusted when it is large enough to hold the fields
    that precede it and small enough to still be a header.  Without that bound a
    ``recovery_dtbo_size`` of 0x2000 would be read as a header size and the
    split cursor would land 5x too far into the file.
    """
    version = _u32(data, 1632)
    if version not in (0, 1, 2, 3, 4):
        return None, _HDR_V0_SIZE

    for candidate_size in (_u32(data, 1640), _u32(data, 1656)):
        if _is_plausible_header_size(candidate_size, data):
            return version, candidate_size

    # The bundled mkbootimg writes only the size, at 1644, and implies v1.
    shifted = _u32(data, 1644)
    if _is_plausible_header_size(shifted, data):
        return 1, shifted
    return None, _HDR_V0_SIZE


def _is_plausible_header_size(size: int, data: bytes) -> bool:
    return _HDR_V0_SIZE <= size <= min(len(data), _HDR_V1_SIZE + 64)


def _decode_v1_os_version(packed: int) -> OsVersion:
    """Split a v1 32-bit os_version word.

    AOSP packs ``[vendor_api_level:7][os_version:8][security_patch:11][patch_level:6]``
    with the patch level in the low bits and the version fields byte-swapped on
    some OEM images, so the shift-based split is applied and the date components
    are recovered when the encoding is the expected one.
    """
    patch_level = packed & 0x3F
    security_patch = (packed >> 6) & 0x7FF
    os_version = (packed >> 17) & 0xFF
    vendor_api = (packed >> 25) & 0x7F
    year = 2000 + ((security_patch >> 4) & 0x7F)
    month = security_patch & 0xF
    day = (security_patch >> 8) & 0xF
    return OsVersion(
        os_version=os_version,
        patch_level=patch_level,
        security_patch_date=(
            (year << 9) | (month << 5) | day
            if 1 <= month <= 12 and 1 <= day <= 31
            else None
        ),
        vendor_api_level=vendor_api or None,
    )


class Session:
    """An unpacked-image working directory.

    ``Session(workdir)`` describes a directory; :meth:`unpack` fills it in.
    The Bash equivalent is "run ``unpackimg.sh`` from this directory".
    """

    def __init__(
        self, workdir: Path | str, *, bin_dir: Path | str | None = None
    ) -> None:
        self.workdir = Path(workdir)
        self.bin_dir = Path(bin_dir) if bin_dir is not None else self._default_bin_dir()
        self.split_img = self.workdir / "split_img"
        self.ramdisk = self.workdir / "ramdisk"
        self._name = ""

    def _default_bin_dir(self) -> Path:
        """Locate the bundled ``bin/`` tree that ships alongside the package."""
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "bin"
            if (candidate / "androidbootimg.magic").is_file():
                return candidate
        return self.workdir / "bin"

    # -- image resolution -------------------------------------------------

    def resolve_image(self, image: Path | str | None) -> Path:
        """Resolve the image to unpack.

        With no argument the Bash auto-picked the first ``*.elf``/``*.img``/
        ``*.sin`` in the working directory, skipping its own intermediates and
        the Loki donor blob.  Reproduced here, including the skip list.
        """
        if image is not None:
            candidate = Path(image)
            if not candidate.is_absolute():
                cwd_candidate = self.workdir / candidate
                candidate = cwd_candidate if cwd_candidate.is_file() else candidate
            if not candidate.is_file():
                raise UnpackError(f"No image file supplied: {candidate}")
            return candidate.resolve()

        for entry in sorted(self.workdir.iterdir()) if self.workdir.is_dir() else []:
            if not entry.is_file():
                continue
            if entry.name in AUTO_PICK_SKIP:
                continue
            if entry.suffix.lower() in IMAGE_SUFFIXES:
                return entry.resolve()
        raise UnpackError("No image file supplied and none found to auto-pick")

    # -- directory lifecycle ---------------------------------------------

    def clean(self) -> bool:
        """Remove prior session state, mirroring ``cleanup.sh``.

        The Bash needed ``sudo`` here too when ``ramdisk/`` was root-owned.  Our
        own archives are always written uid=0/gid=0 but land owned by whoever ran
        us, so a plain ``rmtree`` is always sufficient.

        Returns ``True`` if anything was actually removed.
        """
        removed = False
        for target in (self.ramdisk, self.split_img):
            if target.exists():
                logger.info(f"Removing old {target.name}/ ...")
                shutil.rmtree(target, ignore_errors=True)
                removed = True
        for leftover in self.workdir.glob("*new.*"):
            if leftover.is_file():
                leftover.unlink()
                removed = True
        if (self.workdir / MANIFEST_NAME).is_file():
            (self.workdir / MANIFEST_NAME).unlink()
            removed = True
        return removed

    def prepare(self) -> None:
        """Create a clean ``split_img/`` + ``ramdisk/`` pair."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.clean()
        self.split_img.mkdir(parents=True, exist_ok=True)
        self.ramdisk.mkdir(parents=True, exist_ok=True)
        logger.info("Work folders ready.")

    # -- head signature ---------------------------------------------------

    def _strip_head_signature(
        self, data: bytes, name: str
    ) -> tuple[bytes, SignatureInfo]:
        """Remove a prepended OEM signature, returning the bare image.

        Each scheme has its own layout, so each gets its own removal; what they
        share is that the payload after the wrapper is still an ordinary boot
        image.  The SONY SIN case is the exception the Bash special-cased: the
        container is a whole-file packaging format with a ``.sin`` sidecar, and
        there is no sigtype left to record afterwards.
        """
        sig = SignatureInfo()
        if data.startswith(b"-SIGNED-BY-SIGNBLOB-"):
            sig.type = "BLOB"
            sig.blob_partition = (
                _read_cstring(data, len(b"-SIGNED-BY-SIGNBLOB-"), 3) or None
            )
        elif data.startswith(b"CHROMEOS"):
            sig.type = "CHROMEOS"
            sig.stripped = True
            return _chromeos_unvmlinuz(data), sig
        elif data.startswith(b"DHTB\x01\x00\x00"):
            sig.type = "DHTB"
            sig.stripped = True
            return data[512:], sig
        elif data[64:94] in _NOOK_MAGICS:
            sig.type = "NOOK"
            sig.stripped = True
            self._write_sidecar(f"{name}-master_boot.key", data[:1048576])
            return data[1048576:], sig
        elif data[48:58] == b"BauwksBoot":
            sig.type = "NOOKTAB"
            sig.stripped = True
            self._write_sidecar(f"{name}-master_boot.key", data[:262144])
            return data[262144:], sig
        elif data.startswith((b"\x01\x00\x00\x00", b"\x02\x00\x00\x00", b"\x03SIN")):
            # sony_dump rewrites the container into a bare image plus a sidecar;
            # without the binary we cannot recover the payload, so this is the one
            # signature that genuinely requires a bundled tool.
            sig.type = _sony_version(data)
            sig.stripped = True
            self._write_sidecar(f"{name}.sin", data)
            return data, sig

        sig.stripped = True
        return data, sig

    def _write_sidecar(self, filename: str, payload: bytes) -> Path:
        target = self.split_img / filename
        target.write_bytes(payload)
        return target

    # -- patches ----------------------------------------------------------

    def _revert_loki(self, data: bytes) -> tuple[bytes, PatchInfo]:
        """Undo a Loki patch, which needs the partition name to reverse it."""
        # LOKI stores the partition type in its own header; without the bundled
        # loki_tool the payload cannot be recovered.
        raise UnpackError(
            "Loki-patched image detected but the loki_tool dependency "
            "(aik_py.patches.loki) is unavailable on this branch"
        )

    def _revert_amonet(self, data: bytes) -> tuple[bytes, PatchInfo]:
        """Split an Amonet microloader wrapper off the front of the image.

        Amonet prepends 1024 bytes of microloader to a 2048-byte ``head`` and
        relocates the first 2048 bytes of the real image into a ``tail`` at
        offset 2048, so the wrapper is ``microloader + head(2048) + tail(2048)``.
        Reverting is therefore ``head + tail == data[1024:4096]``: keep the
        1024-byte gap, splice the moved block back to the front.  Dropping the
        gap instead would discard the whole AOSP header.
        """
        if len(data) < 4096:
            raise UnpackError(
                "Amonet image is too small to contain a microloader wrapper"
            )
        microloader = data[:1024]
        sidecar = self._write_sidecar(f"{self._name}-microloader.bin", microloader)
        logger.info(
            f"Amonet patch detected, reverting (microloader -> {sidecar.name})."
        )
        return data[1024:4096] + data[4096:], PatchInfo(amonet_microloader=sidecar.name)

    # -- splitting --------------------------------------------------------

    def _split_aosp(
        self, data: bytes, name: str, magic: bytes = AOSP_MAGIC
    ) -> tuple[Header, int | None, PayloadRefs]:
        """Split an AOSP boot image into its header fields and payload blobs.

        The payload of an AOSP image is fully determined by the header: kernel,
        ramdisk and second are each a page-aligned region whose size comes from a
        header field, in a fixed order.  Reading the header therefore *is* the
        split -- no external unpacker is needed for the standard layout, which is
        the whole point of doing this natively.

        Returns ``(header, header_version, payloads)``.
        """
        header, header_version, header_size = parse_aosp_header(data, magic)
        page = header.page_size or 2048
        if page <= 0:
            raise UnpackError(f"implausible page size {page}")

        cursor = (header_size + page - 1) // page * page

        def take(size: int) -> bytes:
            nonlocal cursor
            chunk = data[cursor : cursor + size]
            cursor += (size + page - 1) // page * page
            return chunk

        payloads = PayloadRefs()
        kernel = take(_u32(data, 8))
        ramdisk = take(_u32(data, 16))
        second_size = _u32(data, 24)
        second = take(second_size) if second_size else b""

        if kernel:
            payloads.kernel = self._write_field(name, "kernel", kernel)
        if ramdisk:
            payloads.ramdisk = self._write_field(name, "ramdisk", ramdisk)
        if second:
            payloads.second = self._write_field(name, "second", second)
        return header, header_version, payloads

    def _split_uboot(self, data: bytes, name: str) -> tuple[Header, PayloadRefs]:
        """Split a DENX U-Boot image into its single payload."""
        if len(data) < 64:
            raise UnpackError("U-Boot image is too small to contain a header")
        payload_size = struct.unpack_from(">I", data, 12)[0]
        payloads = PayloadRefs(
            kernel=self._write_field(name, "kernel", data[64 : 64 + payload_size])
        )
        return Header(), payloads

    def _split_krnl(self, data: bytes, name: str) -> tuple[Header, PayloadRefs]:
        """Split a Rockchip KRNL image: a 4096-byte stub then the ramdisk."""
        payloads = PayloadRefs(ramdisk=self._write_field(name, "ramdisk", data[4096:]))
        return Header(), payloads

    def _write_field(self, name: str, suffix: str, payload: bytes) -> str:
        """Write ``split_img/<name>-<suffix>`` and return its relative path."""
        target = self.split_img / f"{name}-{suffix}"
        target.write_bytes(payload)
        return target.relative_to(self.workdir).as_posix()

    def _write_text_field(self, name: str, suffix: str, text: str | None) -> str | None:
        if text is None:
            return None
        return self._write_field(name, suffix, text.encode("utf-8"))

    # -- MTK ---------------------------------------------------------------

    def _strip_mtk(
        self, kernel_path: Path | None, ramdisk_path: Path | None
    ) -> str | None:
        """Strip MediaTek's 512-byte header off the kernel and/or ramdisk.

        The Bash's fallback chain is worth preserving exactly: if the kernel has
        an MTK header but the ramdisk type is unknown, assume ``rootfs``; if the
        kernel has none but the ramdisk does, warn and carry on, because a
        non-MTK kernel with an MTK ramdisk is a real OEM configuration.
        """
        mtk = False
        if kernel_path is not None and _starts_with(kernel_path, MTK_MAGIC):
            logger.info("MTK header found in kernel, removing...")
            _strip_prefix(kernel_path, 512)
            mtk = True

        mtktype: str | None = None
        if ramdisk_path is not None:
            head = _head_bytes(ramdisk_path, 512)
            if head.startswith(MTK_MAGIC):
                if not mtk:
                    logger.warning("No MTK header found in kernel!")
                    mtk = True
                # The on-disk name is uppercase ("ROOTFS") but mkmtkhdr takes
                # ``--rootfs``, and the Bash recorded the lowercased form, so
                # match it exactly: an uppercase value silently repacks wrong.
                mtktype = (
                    _read_cstring(
                        head, MTK_TYPE_OFFSET, MTK_HEADER_SIZE - MTK_TYPE_OFFSET
                    )
                    .strip()
                    .lower()
                    or None
                )
                logger.info(
                    f'MTK header found in "{mtktype}" type ramdisk, removing...'
                )
                _strip_prefix(ramdisk_path, 512)
            elif mtk and not mtktype:
                logger.warning(
                    'No MTK header found in ramdisk, assuming "rootfs" type!'
                )
                mtktype = "rootfs"

        return mtktype if mtk else None

    # -- ramdisk ----------------------------------------------------------

    def _extract_ramdisk(self, archive: Path, compression: str) -> bool:
        """Decompress and unpack ``archive`` into ``ramdisk/``.

        This is the single call that needs the sibling units.  ``aik_py.cpio``
        owns archive parsing and ``aik_py.compression`` owns the codecs; until
        they land this step is skipped with an explicit message rather than
        silently producing an empty ``ramdisk/`` that would look like a
        successful unpack.
        """
        try:
            from aik_py.compression import decompress as _decompress
        except ImportError:
            _decompress = None

        try:
            from aik_py.cpio import extract_archive as _extract
        except ImportError:
            _extract = None

        if _decompress is None or _extract is None:
            missing = []
            if _decompress is None:
                missing.append("aik_py.compression")
            if _extract is None:
                missing.append("aik_py.cpio")
            logger.warning(
                f"Skipping ramdisk extraction: {' and '.join(missing)} not available "
                f"on this branch. The archive is preserved at "
                f"{archive.relative_to(self.workdir).as_posix()}."
            )
            return False

        raw = _decompress(archive.read_bytes(), compression)
        _extract(raw, self.ramdisk)
        return True

    # -- the pipeline -----------------------------------------------------

    def unpack(self, image: Path | str | None = None) -> Manifest:
        """Unpack an image into this session and return its manifest.

        Follows ``unpackimg.sh`` in order: resolve the image, prepare folders,
        strip the head signature, classify, revert patches, detect the footer,
        split, strip MTK headers, decompress and extract the ramdisk, then write
        the manifest.  The familiar ``split_img/<image>-<field>`` files are
        emitted alongside it so the legacy Bash repack still works.
        """
        source = self.resolve_image(image)
        self._name = source.name
        logger.info(f"Supplied image: {source.name}")

        data = source.read_bytes()
        self.prepare()

        manifest = Manifest(
            tool_version=TOOL_VERSION,
            source_name=source.name,
            source_size=len(data),
        )
        name = source.name

        # 3. head signature
        sigtype = detect_head_signature(data)
        if sigtype is not None:
            logger.info(f'Signature with "{sigtype}" type detected, removing...')
            data, manifest.signature = self._strip_head_signature(data, name)
            # A SIN container is a whole-file packaging format, not a re-signable
            # signature: the Bash deletes the sigtype for it and so must we, or
            # the legacy repack takes its signing branch, matches no case, and
            # never produces an output image.
            if not sigtype.startswith("SIN"):
                self._write_text_field(name, "sigtype", manifest.signature.type)
            self._write_text_field(name, "avbtype", manifest.signature.avb_partition)
            self._write_text_field(name, "blobtype", manifest.signature.blob_partition)

        # 4. image type.  Unrecognized (no magic matched) and unsupported (we
        # know what it is and decline) are deliberately different exceptions.
        head = data[:4096]
        imgtype = _detect_image_type(head)
        if not imgtype:
            raise UnrecognizedImageError(f"No known container magic matched {name}")
        if imgtype not in SUPPORTED_IMAGE_TYPES:
            raise UnsupportedImageError(
                f"Image type {imgtype!r} is recognised but not supported "
                f"(supported: {', '.join(SUPPORTED_IMAGE_TYPES)})"
            )
        manifest.image_type = imgtype
        logger.info(f"Image type: {imgtype}")
        self._write_text_field(name, "imgtype", imgtype)
        self._write_text_field(name, "origsize", str(len(data)))

        # 5. patches
        if len(head) >= 1028 and head[1024:1028] == b"LOKI":
            data, manifest.patches = self._revert_loki(data)
            self._write_text_field(name, "lokitype", manifest.patches.loki_type)
        elif b"microloader" in head[:2048]:
            data, manifest.patches = self._revert_amonet(data)

        # 6. tail footer
        tail = detect_tail_type(data)
        if tail is not None:
            manifest.tail_type = tail
            self._write_text_field(name, "tailtype", tail)
            if tail == "AVBv1":
                manifest.signature.type = "AVBv1"
                manifest.signature.avb_partition = _avb_partition_from_tail(data)
                self._write_text_field(
                    name, "avbtype", manifest.signature.avb_partition
                )
            else:
                logger.info(f'Footer with "{tail}" type detected.')

        # 7. split.  The AOSP family shares one layout and differs only in the
        # leading magic, so all three route through the same splitter.  PXA is
        # unpacked here even though it repacks through pxa-mkbootimg, because
        # the container layout is identical and only the tooling differs.
        if imgtype in ("AOSP", "AOSP_VNDR", "AOSP-PXA"):
            magic = VNDR_BOOT_MAGIC if imgtype == "AOSP_VNDR" else AOSP_MAGIC
            header, version, payloads = self._split_aosp(data, name, magic)
            manifest.header = header
            manifest.header_version = version
        elif imgtype == "U-Boot":
            manifest.header, payloads = self._split_uboot(data, name)
        elif imgtype == "KRNL":
            manifest.header, payloads = self._split_krnl(data, name)
        else:
            raise UnsupportedImageError(
                f"Splitting {imgtype} images is not implemented on this branch"
            )
        manifest.payloads = payloads
        self._emit_header_fields(name, manifest.header, manifest.header_version)

        # 8. MTK headers
        mtktype = self._strip_mtk(
            _resolve(self.workdir, payloads.kernel),
            _resolve(self.workdir, payloads.ramdisk),
        )
        if mtktype:
            manifest.patches.mtk_type = mtktype
            self._write_text_field(name, "mtktype", mtktype)

        # 9 + 10. decompress and extract the ramdisk
        self._handle_ramdisk(name, manifest)

        # 11. persist
        manifest.bind(self.workdir)
        manifest.write(self.workdir / MANIFEST_NAME)
        logger.info("Done!")
        return manifest

    def _handle_ramdisk(self, name: str, manifest: Manifest) -> None:
        """Rename the archive, detect its compression, and extract it."""
        ramdisk_path = _resolve(self.workdir, manifest.payloads.ramdisk)
        if ramdisk_path is None or not ramdisk_path.exists():
            manifest.ramdisk_compression = "empty"
            logger.warning("No ramdisk found to be unpacked!")
            self._write_text_field(name, "ramdiskcomp", "empty")
            return

        compression = detect_compression(_head_bytes(ramdisk_path, 16))
        if compression == "data":
            raise UnrecognizedImageError(
                "Ramdisk payload is not a recognised archive "
                f"(leading bytes: {_head_bytes(ramdisk_path, 8).hex()})"
            )
        manifest.ramdisk_compression = compression
        self._write_text_field(name, "ramdiskcomp", compression)
        logger.info(f"Compression used: {compression}")

        extension = COMPRESSION_EXTENSION.get(compression, "")
        suffix = f".{extension}" if extension else ""
        archive = self.split_img / f"{name}-ramdisk.cpio{suffix}"
        if ramdisk_path != archive:
            ramdisk_path.rename(archive)
        manifest.payloads.ramdisk_archive = archive.relative_to(self.workdir).as_posix()
        manifest.payloads.ramdisk = None

        self._extract_ramdisk(archive, compression)

    def _emit_header_fields(
        self, name: str, header: Header, version: int | None
    ) -> None:
        """Write the ``-field`` files the legacy repack script globs for.

        The four ``*_offset`` suffixes are the *relative* values that
        ``mkbootimg --*_offset`` expects, not the absolute load addresses stored
        in the header.  Repacking with an absolute value there silently produces
        an image the bootloader cannot load.
        """
        for suffix, value in (
            ("cmdline", header.cmdline),
            ("board", header.board or header.name),
            ("base", hex(header.base) if header.base is not None else None),
            ("pagesize", str(header.page_size) if header.page_size else None),
            ("hashtype", header.hashtype),
            ("name", header.name),
            ("header_version", None if version is None else str(version)),
        ):
            self._write_text_field(name, suffix, value)
        for suffix, value in (
            ("kernel_offset", header.kernel_offset),
            ("ramdisk_offset", header.ramdisk_offset),
            ("second_offset", header.second_offset),
            ("tags_offset", header.tags_offset),
        ):
            if value is not None:
                self._write_text_field(name, suffix, hex(value))


def _resolve(workdir: Path, value: str | None) -> Path | None:
    return None if value is None else workdir / value


def _head_bytes(path: Path, count: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(count)


def _starts_with(path: Path, magic: bytes) -> bool:
    with path.open("rb") as handle:
        return handle.read(len(magic)) == magic


def _strip_prefix(path: Path, count: int) -> None:
    """Drop the first ``count`` bytes of a file, in place."""
    path.write_bytes(path.read_bytes()[count:])
