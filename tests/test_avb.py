"""Tests for :mod:`aik_py.avb`.

Signing tests need a real RSA key and the ``openssl`` binary. The bundled
AOSP test key is used when present; otherwise those tests skip.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from aik_py._vendor import avbtool
from aik_py.avb import (
    AvbKey,
    SignatureError,
    add_hash_footer,
    partition_name_from_footer,
    resolve_key,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_KEY = REPO_ROOT / "bin" / "avb" / "verity.pk8"
BUNDLED_CERT = REPO_ROOT / "bin" / "avb" / "verity.x509.pem"
VENDORED_AVBTOOL = Path(avbtool.__file__)

requires_key = pytest.mark.skipif(
    not BUNDLED_KEY.is_file() or not BUNDLED_CERT.is_file(),
    reason="bundled AVB test key not present",
)
requires_openssl = pytest.mark.skipif(
    shutil.which("openssl") is None, reason="openssl not installed"
)


def _make_key(directory: Path, name: str) -> AvbKey:
    """Copy the bundled key into ``directory`` under a new ``name``."""
    directory.mkdir(parents=True, exist_ok=True)
    pk8 = directory / f"{name}.pk8"
    cert = directory / f"{name}.x509.pem"
    pk8.write_bytes(BUNDLED_KEY.read_bytes())
    cert.write_bytes(BUNDLED_CERT.read_bytes())
    return AvbKey(name=name, pk8=pk8, certificate=cert)


def _parse_vbmeta(signed: bytes) -> tuple[object, bytes, bytes, bytes]:
    """Return (header, vbmeta, auth block, aux block) for a signed image."""
    footer = signed[-64:]
    assert footer[:4] == avbtool.AvbFooter.MAGIC
    vbmeta_offset, vbmeta_size = struct.unpack_from("!2Q", footer, 20)
    vbmeta = signed[vbmeta_offset : vbmeta_offset + vbmeta_size]
    header = avbtool.AvbVBMetaHeader(vbmeta[: avbtool.AvbVBMetaHeader.SIZE])
    header_size = avbtool.AvbVBMetaHeader.SIZE
    auth = vbmeta[header_size : header_size + header.authentication_data_block_size]
    aux = vbmeta[header_size + header.authentication_data_block_size :]
    return header, vbmeta, auth, aux


class TestResolveKey:
    def test_finds_bundled_default_key(self) -> None:
        key = resolve_key(None, [REPO_ROOT])
        assert key.name == "verity"
        assert key.pk8 == BUNDLED_KEY
        assert key.certificate == BUNDLED_CERT

    @requires_key
    def test_finds_user_key_in_directory(self, tmp_path: Path) -> None:
        expected = _make_key(tmp_path, "custom")
        found = resolve_key("custom", [tmp_path])
        assert found.pk8 == expected.pk8
        assert found.certificate == expected.certificate

    @requires_key
    def test_finds_key_by_bare_path(self, tmp_path: Path) -> None:
        expected = _make_key(tmp_path, "bare")
        found = resolve_key(str(tmp_path / "bare"), [])
        assert found.pk8 == expected.pk8

    @requires_key
    def test_prefers_bare_path_over_search_dir(self, tmp_path: Path) -> None:
        """repackimg.sh probed "$2" before "$cur/$2" and "$aik/$2"."""
        first = _make_key(tmp_path / "first", "dup")
        _make_key(tmp_path / "second", "dup")
        found = resolve_key(str(tmp_path / "first" / "dup"), [tmp_path / "second"])
        assert found.pk8 == first.pk8

    def test_missing_pk8_raises_with_probed_paths(self, tmp_path: Path) -> None:
        # Certificate present, private key missing.
        (tmp_path / "nopk8.x509.pem").write_bytes(BUNDLED_CERT.read_bytes())
        with pytest.raises(SignatureError) as excinfo:
            resolve_key("nopk8", [tmp_path])
        assert "nopk8" in str(excinfo.value)
        assert "nopk8.pk8" in str(excinfo.value)

    def test_missing_certificate_raises(self, tmp_path: Path) -> None:
        # Private key present, certificate missing.
        (tmp_path / "nocert.pk8").write_bytes(BUNDLED_KEY.read_bytes())
        with pytest.raises(SignatureError) as excinfo:
            resolve_key("nocert", [tmp_path])
        assert "nocert.x509.*" in str(excinfo.value)

    def test_lists_every_probed_path(self, tmp_path: Path) -> None:
        with pytest.raises(SignatureError) as excinfo:
            resolve_key("absent", [tmp_path / "a", tmp_path / "b"])
        message = str(excinfo.value)
        assert str(tmp_path / "a" / "absent") in message
        assert str(tmp_path / "b" / "absent") in message

    @requires_key
    def test_accepts_alternate_certificate_extension(self, tmp_path: Path) -> None:
        """The Bash globbed ``$keytest.x509.*``, so the extension is not fixed."""
        _make_key(tmp_path, "altext")
        (tmp_path / "altext.x509.pem").unlink()
        (tmp_path / "altext.x509.der").write_bytes(BUNDLED_CERT.read_bytes())
        found = resolve_key("altext", [tmp_path])
        assert found.certificate == tmp_path / "altext.x509.der"

    def test_glob_metacharacters_in_name_do_not_crash(self, tmp_path: Path) -> None:
        """A name like ``we[i]rd`` must not blow up certificate resolution."""
        with pytest.raises(SignatureError):
            resolve_key("we[i]rd", [tmp_path])


@requires_key
@requires_openssl
class TestAddHashFooter:
    image = b"boot image contents" * 32

    def test_produces_larger_image_with_footer(self) -> None:
        signed = add_hash_footer(self.image, "boot", BUNDLED_KEY, BUNDLED_CERT)
        assert len(signed) > len(self.image)
        assert signed[-64:-60] == avbtool.AvbFooter.MAGIC
        assert signed.startswith(self.image)

    def test_footer_records_original_image_size(self) -> None:
        signed = add_hash_footer(self.image, "boot", BUNDLED_KEY, BUNDLED_CERT)
        original_size = struct.unpack_from("!Q", signed, len(signed) - 64 + 12)[0]
        assert original_size == len(self.image)

    def test_malformed_sparse_image_raises_signature_error(self) -> None:
        """A truncated sparse header must not leak struct.error/ValueError."""
        truncated = struct.pack("<I4H4I", 0xED26FF3A, 1, 0, 0, 0, 0, 0, 0, 0)[:6]
        with pytest.raises(SignatureError):
            add_hash_footer(truncated, "system", BUNDLED_KEY, BUNDLED_CERT)

    def test_resigning_padded_image_does_not_raise(self) -> None:
        """A signed image with extra trailing slack re-signs to a smaller output."""
        signed = add_hash_footer(self.image, "boot", BUNDLED_KEY, BUNDLED_CERT)
        padded = signed + b"\0" * 8192
        assert len(padded) > len(signed)
        resigned = add_hash_footer(padded, "boot", BUNDLED_KEY, BUNDLED_CERT)
        assert resigned[-64:-60] == avbtool.AvbFooter.MAGIC
        # avbtool keeps the padded bytes as the image and re-appends metadata.
        assert struct.unpack_from("!Q", resigned, len(resigned) - 64 + 12)[0] == len(
            padded
        )
        assert resigned.startswith(padded)

    def test_explicit_partition_size_pads_to_that_size(self) -> None:
        """Passing the real partition size controls the final image size."""
        trimmed = add_hash_footer(
            self.image, "boot", BUNDLED_KEY, BUNDLED_CERT, partition_size=131072
        )
        assert len(trimmed) == 131072
        assert trimmed[: len(self.image)] == self.image
        assert partition_name_from_footer(trimmed) == "boot"
        assert struct.unpack_from("!Q", trimmed, len(trimmed) - 64 + 12)[0] == len(
            self.image
        )

    def test_default_output_reserves_metadata_allowance(self) -> None:
        """Without a partition size the output carries avbtool's 72 KiB slack."""
        signed = add_hash_footer(self.image, "boot", BUNDLED_KEY, BUNDLED_CERT)
        slack = len(signed) - len(self.image)
        assert slack >= 4096, "expected room for the vbmeta and footer block"

    def test_vbmeta_parses_and_carries_partition_name(self) -> None:
        for name in ("boot", "recovery"):
            signed = add_hash_footer(self.image, name, BUNDLED_KEY, BUNDLED_CERT)
            header, _vbmeta, _auth, _aux = _parse_vbmeta(signed)
            assert header.magic == avbtool.AvbVBMetaHeader.MAGIC
            assert partition_name_from_footer(signed) == name

    def test_vbmeta_is_signed_not_algorithm_none(self) -> None:
        """avbtool silently emits an unsigned vbmeta when algorithm is NONE."""
        signed = add_hash_footer(self.image, "boot", BUNDLED_KEY, BUNDLED_CERT)
        header, _vbmeta, _auth, _aux = _parse_vbmeta(signed)
        assert (
            header.algorithm_type == avbtool.ALGORITHMS["SHA256_RSA2048"].algorithm_type
        )
        assert header.signature_size > 0
        assert header.public_key_size > 0

    def test_digest_covers_input_image(self) -> None:
        """The hash descriptor must match a fresh salted digest of the input."""
        signed = add_hash_footer(self.image, "boot", BUNDLED_KEY, BUNDLED_CERT)
        footer = avbtool.AvbFooter(signed[-64:])
        vbmeta = signed[
            footer.vbmeta_offset : footer.vbmeta_offset + footer.vbmeta_size
        ]
        header = avbtool.AvbVBMetaHeader(vbmeta[: avbtool.AvbVBMetaHeader.SIZE])
        aux = vbmeta[
            avbtool.AvbVBMetaHeader.SIZE + header.authentication_data_block_size :
        ]
        descriptors = aux[
            header.descriptors_offset : header.descriptors_offset
            + header.descriptors_size
        ]
        hash_descriptor = avbtool.AvbHashDescriptor(descriptors)

        assert hash_descriptor.partition_name == "boot"
        assert hash_descriptor.image_size == len(self.image)
        expected = hashlib.new(hash_descriptor.hash_algorithm, hash_descriptor.salt)
        expected.update(self.image)
        assert expected.digest() == hash_descriptor.digest

    def test_vbmeta_digest_matches_header_plus_aux(self) -> None:
        """AVB signs sha256(header_block + aux_block); verify that link."""
        signed = add_hash_footer(self.image, "boot", BUNDLED_KEY, BUNDLED_CERT)
        footer = avbtool.AvbFooter(signed[-64:])
        vbmeta = signed[
            footer.vbmeta_offset : footer.vbmeta_offset + footer.vbmeta_size
        ]
        header_size = avbtool.AvbVBMetaHeader.SIZE
        header = avbtool.AvbVBMetaHeader(vbmeta[:header_size])
        auth = vbmeta[header_size : header_size + header.authentication_data_block_size]
        aux = vbmeta[header_size + header.authentication_data_block_size :]
        stored = auth[header.hash_offset : header.hash_offset + header.hash_size]
        assert hashlib.sha256(header.encode() + aux).digest() == stored

    def test_signature_verifies_against_embedded_public_key(
        self, tmp_path: Path
    ) -> None:
        """Recompute the RSA check the bootloader performs."""
        signed = add_hash_footer(self.image, "boot", BUNDLED_KEY, BUNDLED_CERT)
        footer = avbtool.AvbFooter(signed[-64:])
        vbmeta = signed[
            footer.vbmeta_offset : footer.vbmeta_offset + footer.vbmeta_size
        ]
        header = avbtool.AvbVBMetaHeader(vbmeta[: avbtool.AvbVBMetaHeader.SIZE])
        header_blob = vbmeta[: avbtool.AvbVBMetaHeader.SIZE]
        header_size = avbtool.AvbVBMetaHeader.SIZE
        auth = vbmeta[header_size : header_size + header.authentication_data_block_size]
        aux = vbmeta[header_size + header.authentication_data_block_size :]

        signature = auth[
            header.signature_offset : header.signature_offset + header.signature_size
        ]
        pubkey = aux[
            header.public_key_offset : header.public_key_offset + header.public_key_size
        ]

        num_bits = struct.unpack_from("!I", pubkey, 0)[0]
        key_bytes = num_bits // 8
        # The AVB public key blob stores the modulus first, then r^2 mod N.
        modulus = int.from_bytes(pubkey[8 : 8 + key_bytes], "big")

        def _der_len(n: int) -> bytes:
            if n < 0x80:
                return bytes([n])
            raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
            return bytes([0x80 | len(raw)]) + raw

        def _der_int(value: int) -> bytes:
            raw = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
            return b"\x02" + _der_len(len(raw)) + raw

        body = _der_int(modulus) + _der_int(65537)
        der = b"\x30" + _der_len(len(body)) + body

        der_path = tmp_path / "pub.der"
        pem_path = tmp_path / "pub.pem"
        data_path = tmp_path / "signed_data.bin"
        sig_path = tmp_path / "signature.bin"
        der_path.write_bytes(der)
        data_path.write_bytes(header_blob + aux)
        sig_path.write_bytes(signature)

        subprocess.run(
            [
                "openssl",
                "rsa",
                "-pubin",
                "-inform",
                "DER",
                "-in",
                str(der_path),
                "-out",
                str(pem_path),
            ],
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            [
                "openssl",
                "dgst",
                "-sha256",
                "-verify",
                str(pem_path),
                "-signature",
                str(sig_path),
                str(data_path),
            ],
            check=True,
            capture_output=True,
        )
        assert b"Verified OK" in result.stdout

    def test_does_not_mutate_input_bytes(self) -> None:
        """avbtool edits files in place; callers pass a copy instead."""
        original = bytes(self.image)
        add_hash_footer(self.image, "boot", BUNDLED_KEY, BUNDLED_CERT)
        assert self.image == original

    def test_resigning_is_idempotent(self) -> None:
        """avbtool unwraps an existing footer, so re-signing returns to the base."""
        first = add_hash_footer(self.image, "boot", BUNDLED_KEY, BUNDLED_CERT)
        second = add_hash_footer(first, "boot", BUNDLED_KEY, BUNDLED_CERT)
        assert len(second) == len(first)
        assert struct.unpack_from("!Q", second, len(second) - 64 + 12)[0] == len(
            self.image
        )
        assert partition_name_from_footer(second) == "boot"

    def test_empty_image_rejected(self) -> None:
        with pytest.raises(SignatureError):
            add_hash_footer(b"", "boot", BUNDLED_KEY, BUNDLED_CERT)

    def test_empty_partition_name_rejected(self) -> None:
        with pytest.raises(SignatureError):
            add_hash_footer(self.image, "", BUNDLED_KEY, BUNDLED_CERT)

    def test_missing_key_file_rejected(self) -> None:
        with pytest.raises(SignatureError):
            add_hash_footer(self.image, "boot", REPO_ROOT / "nope.pk8", BUNDLED_CERT)


class TestPartitionNameFromFooter:
    def test_returns_none_for_unsigned_image(self) -> None:
        assert partition_name_from_footer(b"just some bytes" * 100) is None

    def test_returns_none_for_empty_input(self) -> None:
        assert partition_name_from_footer(b"") is None

    @requires_key
    @requires_openssl
    def test_round_trips_through_footer(self) -> None:
        signed = add_hash_footer(b"payload" * 16, "recovery", BUNDLED_KEY, BUNDLED_CERT)
        assert partition_name_from_footer(signed) == "recovery"

    @requires_key
    @requires_openssl
    def test_reads_hashtree_descriptor(self, tmp_path: Path) -> None:
        """dm-verity partitions carry a hashtree descriptor, not a hash one.

        AvbHashDescriptor raises LookupError on that blob, so this used to
        escape instead of returning the name.
        """
        raw = tmp_path / "hashtree.img"
        raw.write_bytes(b"Z" * 4096)
        pem = tmp_path / "key.pem"
        subprocess.run(
            [
                "openssl",
                "pkcs8",
                "-inform",
                "DER",
                "-in",
                str(BUNDLED_KEY),
                "-nocrypt",
                "-out",
                str(pem),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(VENDORED_AVBTOOL),
                "add_hashtree_footer",
                "--image",
                str(raw),
                "--partition_size",
                "131072",
                "--partition_name",
                "system",
                "--hash_algorithm",
                "sha256",
                "--do_not_generate_fec",
                "--key",
                str(pem),
                "--algorithm",
                "SHA256_RSA2048",
            ],
            check=True,
            capture_output=True,
        )
        assert partition_name_from_footer(raw.read_bytes()) == "system"

    def test_returns_none_for_sparse_image(self) -> None:
        """Footer offsets are unsparsified, so they cannot index raw bytes."""
        footer = avbtool.AvbFooter()
        footer.original_image_size = 4096
        footer.vbmeta_offset = 4096
        footer.vbmeta_size = 1344
        image = bytearray(b"\xab" * 8192)
        image[-64:] = footer.encode()
        sparse = bytearray(struct.pack("<I", 0xED26FF3A)) + bytearray(image)
        assert partition_name_from_footer(bytes(sparse)) is None
