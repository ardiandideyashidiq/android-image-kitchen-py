class AikError(Exception):
    """Base class for every error aik-py raises deliberately."""


class UnsupportedFormatError(AikError):
    """The image is structurally readable but its format is not one AIK handles."""


class CorruptImageError(AikError):
    """The image is truncated, misaligned, or otherwise fails its structural checks."""


class MissingToolError(AikError):
    """A required external binary is not on PATH or in the bundled ``bin/`` tree."""

    def __init__(self, tool: str, *, purpose: str = "", hint: str = "") -> None:
        parts = [f"Required tool not found: {tool!r}."]
        if purpose:
            parts.append(f"It is needed to {purpose}.")
        if hint:
            parts.append(hint)
        super().__init__(" ".join(parts))
        self.tool = tool
        self.purpose = purpose
        self.hint = hint


class CompressionError(AikError):
    """A compressed stream (usually the ramdisk) could not be decompressed or written."""


class UnsupportedCompressionError(CompressionError):
    """The ramdisk uses a compression codec AIK has no packer/unpacker for."""


class SignatureError(AikError):
    """An image signature, AVB footer, or OEM signing envelope could not be handled."""
