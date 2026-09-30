"""Amonet boot-image transform.

Amonet prepends a 1 KiB microloader to the real image. Unpacking peels the two
halves apart: the microloader (bytes 0..1023) is kept aside, and everything from
byte 1024 on — the 1 KiB head stub of the first 2 KiB block, then the payload —
stays in front. So the reconstructed image is simply ``data[1024:]``.
Re-packing is the exact inverse splice: shift everything right by 1 KiB, then
write the saved microloader back over the hole.
"""

from __future__ import annotations

try:  # pragma: no cover - depends on which sibling units have landed
    from aik_py.errors import CorruptImageError
except ImportError:  # pragma: no cover - standalone fallback

    class CorruptImageError(ValueError):
        """An image is not in the shape the operation requires."""


BLOCK = 1024
MIN_SIZE = 2 * BLOCK


def strip(data: bytes, *, strict: bool = False) -> tuple[bytes, bytes]:
    """Return ``(reconstructed_image, microloader)``.

    The transform is a plain 1 KiB splice, so by default a truncated input
    still round-trips: whatever sits in the first 1 KiB is taken as the
    microloader. Pass ``strict`` when the caller has already identified the
    image as Amonet and wants a short buffer rejected outright.
    """
    if strict and len(data) < MIN_SIZE:
        raise CorruptImageError(
            f"Amonet image needs at least {MIN_SIZE} bytes, got {len(data)}"
        )
    return data[BLOCK:], data[:BLOCK]


def apply(data: bytes, microloader: bytes) -> bytes:
    """Prepend ``microloader`` to a stripped image.

    A short microloader is accepted so that a truncated input still round-trips
    through :func:`strip`; only an oversized one is a caller error.
    """
    if len(microloader) > BLOCK:
        raise CorruptImageError(
            f"Amonet microloader cannot exceed {BLOCK} bytes, got {len(microloader)}"
        )
    return microloader + data
