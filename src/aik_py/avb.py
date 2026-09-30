"""Android Verified Boot (AVB) signing.

The Bash toolkit this project replaces signed images with the prebuilt
``bin/boot_signer.jar``. That tool only emits the legacy AVB **v1** signature
block (``02 01 01 30 82 ...``) and drops any AVB **v2** vbmeta it found on the
input, so a repacked image was not genuinely AVB-signed. This module signs with
real AVB v2 instead: the image is padded and followed by an ``AVB0`` vbmeta
blob and a 64-byte ``AVBf`` footer, both of which a bootloader under
``dm-verity``/AVB 2.0 accepts.

Signing is delegated to AOSP's ``avbtool.py`` (vendored under
:mod:`aik_py._vendor`) because it is not installable from PyPI. See
``_vendor/README`` for provenance. The vendored tool is driven through its
Python API (:class:`~aik_py._vendor.avbtool.Avb`), not by shelling out to its
CLI, but it does shell out to ``openssl(1)`` for raw RSA operations.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ._vendor import avbtool

__all__ = (
    "AvbError",
    "AvbKey",
    "MissingToolError",
    "SignatureError",
    "add_hash_footer",
    "partition_name_from_footer",
    "resolve_key",
)


class AvbError(Exception):
    """Base class for AVB signing failures."""


class SignatureError(AvbError):
    """Raised when an image cannot be signed or a signature cannot be checked."""


class MissingToolError(AvbError):
    """Raised when a required external tool such as ``openssl`` is unavailable."""


DEFAULT_KEY_NAME = "verity"
_PK8_SUFFIX = ".pk8"
_KEY_CERT_STEM = ".x509"


@dataclass(frozen=True)
class AvbKey:
    """An AVB signing key pair: a PKCS#8 private key and its X.509 certificate.

    ``pk8`` is DER-encoded PKCS#8 (``verity.pk8``) and ``certificate`` is a PEM
    X.509 file (``verity.x509.pem``). Both must live beside each other under a
    shared basename, which is what :func:`resolve_key` searches for.
    """

    name: str
    pk8: Path
    certificate: Path

    def __str__(self) -> str:
        return f"{self.name} ({self.pk8})"


def _certificate_for(directory: Path, name: str) -> Path | None:
    """Return the certificate next to ``directory/name.pk8``, or None.

    The Bash toolkit globbed ``$keytest.x509.*`` rather than a fixed
    ``.x509.pem`` name, so any extension is accepted to stay compatible.
    """
    stem = f"{name}{_KEY_CERT_STEM}"
    try:
        candidates = sorted(directory.glob(f"{stem}*"))
    except ValueError:
        # A name containing glob metacharacters cannot be globbed; the
        # common ``.x509.pem`` spelling is the only sensible fallback.
        candidates = [directory / f"{stem}.pem"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _key_at(directory: Path, name: str) -> AvbKey | None:
    """Return the key pair in ``directory`` named ``name``, or None if incomplete.

    Both halves are required: a ``.pk8`` without a certificate (or the reverse)
    is rejected, mirroring the Bash test in ``repackimg.sh``.
    """
    pk8 = directory / f"{name}{_PK8_SUFFIX}"
    if not pk8.is_file():
        return None
    certificate = _certificate_for(directory, name)
    if certificate is None:
        return None
    return AvbKey(name=name, pk8=pk8, certificate=certificate)


def resolve_key(name: str | None, search_dirs: Sequence[Path]) -> AvbKey:
    """Locate an AVB key pair, reproducing the Bash ``--avbkey`` lookup.

    ``repackimg.sh`` probed ``"$2"``, ``"$cur/$2"`` and ``"$aik/$2"`` in order
    and took the first candidate directory holding both a ``<name>.pk8`` and a
    ``<name>.x509.*`` file. Here ``search_dirs`` is that ordered probe list and
    ``name`` is the user-supplied key name; when ``name`` is None the bundled
    ``bin/avb/verity`` key is used, resolved against each search dir in turn.

    Raises:
        SignatureError: if no candidate holds a complete key pair. The message
            lists every path probed.
    """
    key_name = name or DEFAULT_KEY_NAME
    if name is None:
        # The default key ships in the toolkit's bin/avb/ directory, so each
        # search root may either be the toolkit root or the bin directory
        # itself; probe both so callers can pass either.
        candidates = []
        for root in search_dirs:
            candidates.append(Path(root) / "bin" / "avb" / key_name)
            candidates.append(Path(root) / "avb" / key_name)
    else:
        candidates = [Path(name), *(Path(d) / name for d in search_dirs)]

    probed: list[str] = []
    for candidate in candidates:
        key = _key_at(candidate.parent, candidate.name)
        if key is not None:
            return key
        probed.append(str(candidate))

    raise SignatureError(
        f"AVB key {key_name!r} not found. Probed for both "
        f"{key_name}{_PK8_SUFFIX} and {key_name}{_KEY_CERT_STEM}.*: "
        + ", ".join(probed)
    )


def _require_openssl() -> str:
    """Return the path to ``openssl``, raising if it is not on PATH."""
    found = shutil.which("openssl")
    if found is None:
        raise MissingToolError(
            "openssl(1) is required for AVB signing (raw RSA sign/verify) but "
            "was not found on PATH."
        )
    return found


def _pk8_to_pem(pk8: Path, dest: Path) -> Path:
    """Convert a DER PKCS#8 private key to PEM, which is what avbtool expects.

    ``avbtool`` feeds keys straight to ``openssl rsa -in``, which only reads
    PEM/SEC1. The bundled keys ship as DER PKCS#8, so they must be converted
    before use.
    """
    pem = dest / f"{pk8.stem}.pem"
    result = subprocess.run(
        [
            _require_openssl(),
            "pkcs8",
            "-inform",
            "DER",
            "-in",
            str(pk8),
            "-nocrypt",
            "-out",
            str(pem),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not pem.is_file():
        detail = result.stderr.decode(errors="replace").strip()
        raise SignatureError(f"Could not convert PKCS#8 key {pk8} to PEM: {detail}")
    return pem


def _algorithm_for(pem: Path) -> str:
    """Pick the AVB algorithm matching the size of a PEM private key.

    avbtool's ``--algorithm`` defaults to ``NONE``, which silently produces an
    *unsigned* vbmeta (algorithm_type 0, no public key, no signature) while
    still exiting 0. That is exactly the kind of quiet downgrade this rewrite
    exists to avoid, so the algorithm is always derived from the key.
    """
    result = subprocess.run(
        [_require_openssl(), "rsa", "-in", str(pem), "-noout", "-text"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise SignatureError(f"Could not read RSA key {pem}: {detail}")

    first_line = result.stdout.decode(errors="replace").splitlines()[:1]
    text = " ".join(first_line)
    for candidate, bits in (
        ("SHA256_RSA8192", 8192),
        ("SHA256_RSA4096", 4096),
        ("SHA256_RSA2048", 2048),
    ):
        if f"{bits} bit" in text:
            return candidate
    raise SignatureError(
        f"Unsupported RSA key size in {pem} ({text.strip()!r}); "
        "AVB signing requires a 2048, 4096 or 8192 bit key."
    )


@contextmanager
def _workdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="aik-avb-") as tmp:
        yield Path(tmp)


def add_hash_footer(
    image: bytes,
    partition_name: str,
    key: Path,
    certificate: Path,
    *,
    hash_algorithm: str = "sha256",
) -> bytes:
    """Sign ``image`` and return it with an AVB v2 hash footer appended.

    The result is the original image, zero-padded to a 4096-byte boundary,
    followed by the ``AVB0`` vbmeta blob, a DONT_CARE region, and a 64-byte
    ``AVBf`` footer describing the layout. ``avbtool`` sizes the output from
    the image rather than taking a fixed partition size, so the signed image is
    only as large as it needs to be.

    Args:
        image: Raw partition image, e.g. a ``boot.img`` from ``mkbootimg``.
        partition_name: Partition recorded in the hash descriptor, such as
            ``"boot"`` or ``"recovery"``. The Bash unpacker recovered this from
            the original footer and passed it through, so repacked images keep
            the same name.
        key: DER PKCS#8 private key (``.pk8``).
        certificate: PEM X.509 certificate (``.x509.pem``). Present for parity
            with the Bash tool and for callers that need the chain; the
            certificate is not embedded in an AVB v2 hash footer, which
            carries the public key in the vbmeta auxiliary block instead.
        hash_algorithm: Digest for the image hash descriptor.

    Returns:
        The signed image bytes.

    Raises:
        SignatureError: if signing fails or produces a vbmeta that is not
            actually signed.
        MissingToolError: if ``openssl`` is unavailable.
    """
    if not image:
        raise SignatureError("Cannot sign an empty image.")
    if not partition_name:
        raise SignatureError(
            "A partition name is required to build the AVB hash descriptor."
        )
    for path in (Path(key), Path(certificate)):
        if not path.is_file():
            raise SignatureError(f"AVB key material not found: {path}")

    _require_openssl()

    with _workdir() as work:
        pem = _pk8_to_pem(Path(key), work)
        algorithm = _algorithm_for(pem)
        image_path = work / "image.img"
        image_path.write_bytes(image)

        avb = avbtool.Avb()
        try:
            avb.add_hash_footer(
                str(image_path),
                None,  # partition_size
                1,  # dynamic_partition_size: grow only as much as needed
                partition_name,
                hash_algorithm,
                None,  # salt: let avbtool pick random bytes of digest size
                None,  # chain_partitions_use_ab
                None,  # chain_partitions_do_not_use_ab
                algorithm,
                str(pem),
                None,  # public_key_metadata_path
                0,  # rollback_index
                0,  # flags
                0,  # rollback_index_location
                None,  # props
                None,  # props_from_file
                None,  # kernel_cmdlines
                None,  # setup_rootfs_from_kernel
                None,  # include_descriptors_from_image
                False,  # calc_max_image_size
                None,  # signing_helper
                None,  # signing_helper_with_files
                None,  # release_string
                None,  # append_to_release_string
                None,  # output_vbmeta_image
                False,  # do_not_append_vbmeta_image
                False,  # print_required_libavb_version
                False,  # use_persistent_digest
                False,  # do_not_use_ab
            )
        except avbtool.AvbError as exc:
            raise SignatureError(f"avbtool failed to add a hash footer: {exc}") from exc

        signed = image_path.read_bytes()

    # Re-signing an already-signed image does not necessarily grow it: avbtool
    # truncates back to the stored original size first, so the result can be
    # the same length. Only a shrink is a failure.
    if len(signed) < len(image):
        raise SignatureError(
            f"AVB signing shrank the image: {len(signed)} bytes from "
            f"{len(image)} bytes."
        )
    if signed[-64:-60] != avbtool.AvbFooter.MAGIC:
        raise SignatureError("AVB signing did not emit a trailing AVBf footer.")

    footer = avbtool.AvbFooter(signed[-64:])
    if footer.original_image_size == 0 or footer.original_image_size > len(signed):
        raise SignatureError(
            f"AVB footer records an implausible original image size "
            f"({footer.original_image_size}) for a {len(signed)} byte image."
        )
    if not footer.vbmeta_size or footer.vbmeta_offset + footer.vbmeta_size > len(
        signed
    ):
        raise SignatureError("AVB footer points outside the signed image.")
    return signed


def partition_name_from_footer(image: bytes) -> str | None:
    """Recover the partition name recorded in an image's AVB v2 hash footer.

    ``unpackimg.sh`` detected the AVBv1 signature block and recorded the
    partition name to pass back to the signer at repack time. This reads the
    same information out of a v2 vbmeta hash descriptor, returning None when
    the image carries no AVB v2 footer.
    """
    if len(image) < avbtool.AvbFooter.SIZE:
        return None
    footer = image[-avbtool.AvbFooter.SIZE :]
    if footer[:4] != avbtool.AvbFooter.MAGIC:
        return None
    vbmeta_offset, vbmeta_size = struct.unpack_from("!2Q", footer, 20)
    if vbmeta_size < avbtool.AvbVBMetaHeader.SIZE or vbmeta_offset + vbmeta_size > len(
        image
    ):
        return None
    vbmeta = image[vbmeta_offset : vbmeta_offset + vbmeta_size]
    try:
        header = avbtool.AvbVBMetaHeader(vbmeta[: avbtool.AvbVBMetaHeader.SIZE])
    except (avbtool.AvbError, struct.error, IndexError, ValueError):
        return None

    aux_start = avbtool.AvbVBMetaHeader.SIZE + header.authentication_data_block_size
    aux = vbmeta[aux_start : aux_start + header.auxiliary_data_block_size]
    descriptors = aux[
        header.descriptors_offset : header.descriptors_offset + header.descriptors_size
    ]
    try:
        descriptor = avbtool.AvbHashDescriptor(descriptors)
    except (avbtool.AvbError, struct.error, IndexError, ValueError):
        return None
    return descriptor.partition_name or None
