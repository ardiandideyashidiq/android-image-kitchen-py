"""Typed manifest describing a fully unpacked Android boot/recovery image.

The Bash toolkit this replaces has no manifest at all: the state of an unpacked
image lives entirely in the *names* of files inside ``split_img/``.  A file named
``boot.img-cmdline`` exists iff a non-empty cmdline was present, and its contents
are the value.  ``repackimg.sh`` then rebuilds the entire ``mkbootimg`` command
line by testing ``[ -f *-field ]`` over thirty globs.

That encoding is lossy and untyped.  A field that is legitimately empty is
indistinguishable from a field that was never parsed, integers and hex strings
are interchangeable depending on which ``cat`` fed them, and several suffixes are
written but never read while others are read but never written.  This module
replaces that with an explicit, typed, losslessly round-trippable record.

The Bash convention is still honoured: :mod:`aik_py.session` writes the familiar
``split_img/<image>-<field>`` files alongside ``manifest.json`` so muscle memory
and the legacy scripts keep working.  The manifest is the source of truth; the
filenames are the compatibility surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path, PurePosixPath
from typing import Any, Self

__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_VERSION",
    "Header",
    "Manifest",
    "OsVersion",
    "PatchInfo",
    "PayloadRefs",
    "SignatureInfo",
    "load_manifest",
]

#: Schema version of this manifest.  Bumped whenever a field changes meaning or a
#: new required field appears.  A reader seeing a higher number than it knows
#: should refuse rather than silently mis-interpret a field it does not model.
MANIFEST_VERSION = 1

#: Filename written into the session directory.
MANIFEST_NAME = "manifest.json"


def _as_str(value: Any) -> str | None:
    """Normalise an optional string field, treating empty/blank as absent."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    return text or None


def _as_int(value: Any) -> int | None:
    """Parse an integer that may arrive as ``int``, ``"0x100"`` or ``"4096"``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text, 0)
        except ValueError:
            try:
                return int(text, 10)
            except ValueError:
                return None
    return None


def _rel(path: Path | str | None) -> str | None:
    """Store paths POSIX-relative so a session directory stays portable."""
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    pure = PurePosixPath(text.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"manifest stores session-relative paths only, got {text!r}")
    return str(pure)


def _abs(workdir: Path, value: str | None) -> Path | None:
    """Resolve a stored relative path back against the session directory."""
    if value is None:
        return None
    return workdir / PurePosixPath(value)


# Raw binary does not appear in a manifest.  The one blob-like field, the image
# ID, is a 20-byte digest and is stored hex-encoded as ``id_hex``; the name says
# so, and JSON stays text-safe without base64 escape noise.


@dataclass(slots=True)
class PayloadRefs:
    """Locations of the extracted payload blobs inside the session directory.

    Every value is a POSIX-relative path such as ``split_img/boot.img-kernel``
    so the whole directory can be moved or archived without rewriting the
    manifest.  ``None`` means the image genuinely had no such component, which is
    distinct from "we did not look".
    """

    kernel: str | None = None
    ramdisk: str | None = None
    ramdisk_archive: str | None = None
    second: str | None = None
    dtb: str | None = None
    dt: str | None = None
    recovery_dtbo: str | None = None
    header: str | None = None
    cmdline: str | None = None
    blob_partition: str | None = None
    master_boot_key: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {f.name: _rel(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Self:
        """Validate paths on read, not just on write.

        A ``manifest.json`` is untrusted input: it can be shipped inside a shared
        session directory or hand-edited.  Validating only when serialising would
        let an absolute or ``..``-escaping path through, and a repack trusting
        :meth:`Manifest.resolve` would then read a file outside the session.
        """
        data = data or {}
        return cls(**{f.name: _rel(_as_str(data.get(f.name))) for f in fields(cls)})

    def path(self, workdir: Path, attr: str) -> Path | None:
        """Resolve one member against ``workdir``."""
        return _abs(workdir, getattr(self, attr, None))


@dataclass(slots=True)
class SignatureInfo:
    """Signature stripped from, or detected on, the image.

    ``type`` mirrors the ``-sigtype`` suffix and carries the AVB variant suffix
    when one applies (``AVBv1 boot``, ``AVBv2``, ...).  ``avb_partition`` and
    ``blob_partition`` record the partition names AVB / the ASUS BLOB scheme
    need in order to re-sign, which the Bash scattered across ``-avbtype`` and
    ``-blobtype``.
    """

    type: str | None = None
    avb_partition: str | None = None
    blob_partition: str | None = None
    stripped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "avb_partition": self.avb_partition,
            "blob_partition": self.blob_partition,
            "stripped": self.stripped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Self:
        data = data or {}
        return cls(
            type=_as_str(data.get("type")),
            avb_partition=_as_str(data.get("avb_partition")),
            blob_partition=_as_str(data.get("blob_partition")),
            stripped=bool(data.get("stripped", False)),
        )


@dataclass(slots=True)
class PatchInfo:
    """Patches applied to the image that must be re-applied on repack."""

    loki_type: str | None = None
    loki_partition: str | None = None
    amonet_microloader: str | None = None
    mtk_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "loki_type": self.loki_type,
            "loki_partition": self.loki_partition,
            "amonet_microloader": self.amonet_microloader,
            "mtk_type": self.mtk_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Self:
        data = data or {}
        return cls(
            loki_type=_as_str(data.get("loki_type")),
            loki_partition=_as_str(data.get("loki_partition")),
            amonet_microloader=_rel(data.get("amonet_microloader")),
            mtk_type=_as_str(data.get("mtk_type")),
        )


@dataclass(slots=True)
class OsVersion:
    """Android OS version triple.

    AOSP header v0 packs ``(os_version << 11) | patch_level`` into the first
    four bytes of the header; v1+ adds a security patch *date* (year/month/day)
    and a vendor API level.  The Bash dumped the whole word to ``-os_version``
    and let ``mkbootimg`` re-parse it, so the split was never recorded at all.
    """

    os_version: int | None = None
    patch_level: int | None = None
    security_patch_date: int | None = None
    vendor_api_level: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "os_version": self.os_version,
            "patch_level": self.patch_level,
            "security_patch_date": self.security_patch_date,
            "vendor_api_level": self.vendor_api_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Self:
        data = data or {}
        return cls(
            os_version=_as_int(data.get("os_version")),
            patch_level=_as_int(data.get("patch_level")),
            security_patch_date=_as_int(data.get("security_patch_date")),
            vendor_api_level=_as_int(data.get("vendor_api_level")),
        )


#: Header fields that must survive as integers.  ``from __future__ import
#: annotations`` turns ``field.type`` into the *string* ``"int | None"``, so the
#: coercion is driven by an explicit set rather than by introspecting types.
_INT_HEADER_FIELDS = frozenset(
    {
        "page_size",
        "base",
        "kernel_addr",
        "ramdisk_addr",
        "second_addr",
        "tags_addr",
        "version",
        "recovery_dtbo_size",
        "kernel_offset",
        "ramdisk_offset",
        "second_offset",
        "tags_offset",
    }
)


@dataclass(slots=True)
class Header:
    """The mkbootimg header fields needed to reconstruct the image.

    Field names mirror the ``mkbootimg`` command-line options and the
    ``split_img/<image>-<field>`` suffixes, so a repack can translate this
    straight back into flags without a lookup table keyed on filename strings.

    The header stores *absolute* load addresses, but ``mkbootimg`` takes
    ``--*_offset`` flags that are relative to ``base``.  Both are kept: the
    addresses are what the file says, the offsets are what a repack needs, and
    conflating them yields an image that boots nowhere.
    """

    page_size: int | None = None
    base: int | None = None
    kernel_addr: int | None = None
    ramdisk_addr: int | None = None
    second_addr: int | None = None
    tags_addr: int | None = None
    kernel_offset: int | None = None
    ramdisk_offset: int | None = None
    second_offset: int | None = None
    tags_offset: int | None = None
    cmdline: str | None = None
    extra_cmdline: str | None = None
    board: str | None = None
    name: str | None = None
    id_hex: str | None = None
    hashtype: str | None = None
    boot_name: str | None = None
    version: int | None = None
    recovery_dtbo_size: int | None = None
    os_version: OsVersion = field(default_factory=OsVersion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_size": self.page_size,
            "base": self.base,
            "kernel_addr": self.kernel_addr,
            "ramdisk_addr": self.ramdisk_addr,
            "second_addr": self.second_addr,
            "tags_addr": self.tags_addr,
            "kernel_offset": self.kernel_offset,
            "ramdisk_offset": self.ramdisk_offset,
            "second_offset": self.second_offset,
            "tags_offset": self.tags_offset,
            "cmdline": self.cmdline,
            "extra_cmdline": self.extra_cmdline,
            "board": self.board,
            "name": self.name,
            "id_hex": self.id_hex,
            "hashtype": self.hashtype,
            "boot_name": self.boot_name,
            "version": self.version,
            "recovery_dtbo_size": self.recovery_dtbo_size,
            "os_version": self.os_version.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Self:
        data = data or {}
        values: dict[str, Any] = {
            f.name: (
                _as_int(data.get(f.name))
                if f.name in _INT_HEADER_FIELDS
                else _as_str(data.get(f.name))
            )
            for f in fields(cls)
            if f.name != "os_version"
        }
        values["os_version"] = OsVersion.from_dict(data.get("os_version"))
        return cls(**values)


@dataclass(slots=True)
class Manifest:
    """Complete description of an unpacked image session.

    Round-trips through JSON without loss: :meth:`to_dict` ->
    ``json.dumps`` -> :meth:`from_dict` yields an equal object, and all
    optional fields survive as ``None`` rather than being dropped.
    """

    manifest_version: int = MANIFEST_VERSION
    tool_version: str | None = None
    source_name: str | None = None
    source_size: int | None = None
    image_type: str | None = None
    header_version: int | None = None
    dt_type: str | None = None
    ramdisk_compression: str | None = None
    tail_type: str | None = None
    signature: SignatureInfo = field(default_factory=SignatureInfo)
    patches: PatchInfo = field(default_factory=PatchInfo)
    header: Header = field(default_factory=Header)
    payloads: PayloadRefs = field(default_factory=PayloadRefs)
    notes: list[str] = field(default_factory=list)
    #: Session directory used to resolve relative payload paths.  Derived state,
    #: not part of the serialised document.
    workdir: Path | None = field(default=None, repr=False, compare=False)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "tool_version": self.tool_version,
            "source_name": self.source_name,
            "source_size": self.source_size,
            "image_type": self.image_type,
            "header_version": self.header_version,
            "dt_type": self.dt_type,
            "ramdisk_compression": self.ramdisk_compression,
            "tail_type": self.tail_type,
            "signature": self.signature.to_dict(),
            "patches": self.patches.to_dict(),
            "header": self.header.to_dict(),
            "payloads": self.payloads.to_dict(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if not isinstance(data, dict):
            raise TypeError("manifest must decode to an object")
        version = _as_int(data.get("manifest_version")) or 0
        if version > MANIFEST_VERSION:
            raise ValueError(
                f"manifest schema version {version} is newer than this tool "
                f"understands ({MANIFEST_VERSION}); refusing to guess at fields"
            )
        return cls(
            manifest_version=version or MANIFEST_VERSION,
            tool_version=_as_str(data.get("tool_version")),
            source_name=_as_str(data.get("source_name")),
            source_size=_as_int(data.get("source_size")),
            image_type=_as_str(data.get("image_type")),
            header_version=_as_int(data.get("header_version")),
            dt_type=_as_str(data.get("dt_type")),
            ramdisk_compression=_as_str(data.get("ramdisk_compression")),
            tail_type=_as_str(data.get("tail_type")),
            signature=SignatureInfo.from_dict(data.get("signature")),
            patches=PatchInfo.from_dict(data.get("patches")),
            header=Header.from_dict(data.get("header")),
            payloads=PayloadRefs.from_dict(data.get("payloads")),
            notes=[str(n) for n in (data.get("notes") or [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> Self:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- path helpers -----------------------------------------------------

    def resolve(self, attr: str) -> Path | None:
        """Resolve a :class:`PayloadRefs` attribute against the session dir."""
        if self.workdir is None:
            raise ValueError("manifest is not bound to a workdir; call bind() first")
        return self.payloads.path(self.workdir, attr)

    def bind(self, workdir: Path) -> Self:
        """Attach a session directory so :meth:`resolve` can be used."""
        self.workdir = Path(workdir)
        return self


def load_manifest(workdir: Path) -> Manifest | None:
    """Read ``<workdir>/manifest.json`` if present, else ``None``.

    Binds the result to ``workdir`` so payload paths resolve immediately.
    """
    path = Path(workdir) / MANIFEST_NAME
    if not path.is_file():
        return None
    return Manifest.read(path).bind(Path(workdir))
