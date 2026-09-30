# Format support matrix

Android boot images come in many vendor-specific shapes. This page states
plainly which ones `aik-py` handles **natively** — parsing and rebuilding the
bytes in pure Python — and which ones are still **shims** that shell out to the
prebuilt binaries bundled in `bin/`.

The distinction matters: native formats work anywhere Python does, are
testable, and are covered by our own test suite. Shimmed formats still require
the corresponding ELF binary for your platform (`bin/linux/x86_64`,
`bin/linux/i686`, `bin/macos/x86_64`, `bin/macos/i386`) to be present and
executable.

> **Status.** The library is under active development. The "Native" column
> reflects the intended target design; where work is still in flight it is
> marked *(in progress)*. The "Shim" column reflects what genuinely still
> depends on `bin/`.

---

## Image formats

| Format | Magic / marker | Status | Notes |
|--------|----------------|--------|-------|
| **AOSP boot v0** | `ANDROID!` at 0, version `0` at `0x28` | **Native** | No `header_size` field; header ends at the page boundary. |
| **AOSP boot v1** | `ANDROID!` at 0, version `1` at `0x28` | **Native** | Adds `os_version`, `os_patch_level`, `recovery_dtbo`. Reported header size 1648. |
| **AOSP boot v2** | `ANDROID!` at 0, version `2` at `0x28` | **Native** | Adds `recovery_acpio`. Reported header size 1660. |
| **AOSP boot v3** | `ANDROID!` at 0, version `3` at `0x28` | **Native** | **Keeps the magic** — see the trap note below. Reported header size 1580. |
| **AOSP boot v4** | `ANDROID!` at 0, version `4` at `0x28` | **Native** | Same as v3. Reported header size 1580. |
| **`vendor_boot`** | `VNDRBOOT` at 0 | **Native** | v3 and v4. Cmdline/name are **fixed-width NUL-padded** fields (2048/16 B), so `tags_addr@0x81C`, `name@0x820`, `header_size@0x830`, `dtb_size@0x834`, `dtb_addr@0x838` sit at **fixed offsets** — never locate them by scanning for a NUL. `dtb_addr` is a **uint64**. See [architecture.md](architecture.md#the-vendor_boot-header). |
| Sony ELF | `\x7FELF` at 0 | **Shim** | `elftool` + `unpackelf` |
| DENX U-Boot | `\x27\x05\x19\x56` at 0 | **Shim** | `dumpimage` (unpack), `mkimage` (pack) |
| Rockchip KRNL | `KRNL` at 0 | **Shim** | `rkcrc`; payload is `dd` from a fixed offset |
| Intel OSIP | `$OS$\x00\x00\x01` at 0 | **Shim** | `mboot` |
| PXA | `ANDROID!` + PXA variant bytes at `0x24` | **Shim** | `pxa-unpackbootimg` / `pxa-mkbootimg` |

### The v3/v4 trap

The AOSP specification says v3/v4 **drop the `ANDROID!` magic** and begin with
`header_version` at offset 0. **The bundled AIK `mkbootimg` does the opposite**
— it keeps `ANDROID!` at offset 0 with the classic field offsets, and dispatches
on the u32 at `0x28`. Confirmed by building images at every version and reading
the bytes back:

```
v3: 414e 4452 4f49 4421 ... 0300 0000   ANDROID!  version 3 @ 0x28
v4: 414e 4452 4f49 4421 ... 0400 0000   ANDROID!  version 4 @ 0x28
```

Any reader that keys on "no magic ⇒ v3/v4" will mis-parse these. See
[architecture.md](architecture.md#header-versions-3-and-4).

### The `vendor_boot` fixed-width trap

`vendor_boot` (magic `VNDRBOOT`) has a separate, easily-misread hazard: the
`vendor_cmdline` and `name` fields are **fixed-width and NUL-padded** (2048 and
16 bytes), **not** length-prefixed. Every field after them therefore sits at a
**fixed offset** past the cmdline region — `tags_addr@0x81C`, `name@0x820`,
`header_size@0x830`, `dtb_size@0x834`, `dtb_addr@0x838` — no matter how long
the actual text is.

**Never locate these fields by scanning for the NUL terminator.** Doing so
works only on an image whose cmdline happens to fill the field exactly, and
silently produces garbage on everything else. `header_size` exists precisely to
tell the loader how far the populated fields reach (2112 for v3, 2128 for v4).

Also: **`dtb_addr` is a uint64, not 4 bytes.** Reading it as a u32 truncates
load addresses above 4 GiB — a real failure on 64-bit devices, where the dtb
is routinely loaded high.

Real v4 `vendor_boot.img` files from a current device are **spec-conformant**
and parse exactly against this table; nothing is truncated. They also carry an
AVB vbmeta partition after the payloads, which is preserved verbatim so
parse/pack is lossless.

> The v4 multi-fragment ramdisk table (`vendor_ramdisk_table_size` and friends at
> `0x840`–`0x84C`, plus the v4 payload ordering) is **spec-derived and not
> empirically verified**: the bundled `mkbootimg` predates
> `--vendor_ramdisk_fragment` and cannot produce a fragmented image to check
> against. The v3 layout is fully verified. See
> [architecture.md](architecture.md#the-vendor_boot-header).

---

## Patch layers

| Layer | Magic | Status | Requirements |
|-------|-------|--------|--------------|
| **MTK** | `\x88\x16\x88\x58` + `KERNEL`/`ROOTFS`/`RECOVERY` | **Native** | 512-byte header; strip/re-prepend around kernel and/or ramdisk |
| **Loki** | `LOKI` at 1024, patched into the gap between `microloader` (at 96) and `ANDROID!` (at 1024) | **Shim** | bundled `loki_tool` **and a device-specific `aboot.img` dump** |
| **Amonet** | `microloader` at 96, `ANDROID!` at 1024 | **Native** | Pure byte relocation; no external tool |

> **Loki cannot be self-contained.** The patch offset is derived from the target
> device's own bootloader binary. You must supply a dump of *your* device's
> `aboot.img` in the working directory. Without it, repack fails. The legacy
> scripts warn about this during unpack and abort during repack.

---

## Signature wrappers

| Type | Magic | Status | Tool / keys |
|------|-------|--------|-------------|
| **AVB v1** | `\x02\x01\x01\x30\x82` (descriptor) | **Tool-backed** | `bin/boot_signer.jar` + `bin/avb/verity.{pk8,x509.pem}` |
| **BLOB** (ASUS) | `-SIGNED-BY-SIGNBLOB-` at 0 | **Tool-backed** | `blobpack` / `blobunpack` |
| **CHROMEOS** | `CHROMEOS` at 0 | **Tool-backed** | `futility` + `bin/chromeos/` keys |
| **DHTB** (Samsung/Spreadtrum) | `DHTB\x01\x00\x00` at 0 | **Tool-backed** | `dhtbsign` |
| **NOOK / NOOKTAB** | strings at offset 64 / 48 | **Native** | Stored key block prepended; no tool needed |
| **SIN** (Sony) | `\x01\x00\x00\x00` / `\x02\x00\x00\x00` / `\x03SIN` at 0 | **Tool-backed** | `sony_dump` |

---

## Footers

| Footer | Magic | Status |
|--------|-------|--------|
| **Bump** (LG) | `\x41\xA9\xE4\x67\x74\x4D\x1D\x1B\xA4\x29\xF2\xEC\xEA\x65\x52\x79` | **Native** — literal byte string appended |
| **SEAndroid** (Samsung) | `SEANDROIDENFORCE` | **Native** — literal byte string appended |
| **AVB v1** | `\x02\x01\x01\x30\x82` | **Tool-backed** — `boot_signer.jar` |
| **AVB v2** | `AVBf` | **Native** *(in progress)* |

### AVB v2 was broken in the Bash toolkit

The legacy scripts **detected AVB v2 but never re-signed it**. During unpack,
`unpackimg.sh` matches `AVB*` in the tail test and writes a `-sigtype` file —
but the inner `case` only records a type for `*v1`:

```sh
case $tailtype in
  AVB*)
    case $tailtype in
      *v1)
        echo $tailtype > "$file-sigtype";
        echo $tailtest | awk '{ print $4 }' > "$file-avbtype";
      ;;
    esac;
  ;;
```

So for an AVB **v2** image no `-sigtype` file is written. In `repackimg.sh`,
signing is gated entirely on that file:

```sh
if [ -f split_img/*-sigtype ]; then
  ...sign...
fi
```

With no `-sigtype`, the whole signing block is skipped and the rebuilt image is
written **unsigned** — silently, with no warning.

> **This is a potential brick.** An AVB v2 image that is repacked without its
> footer will fail to boot on a device with verified boot enforcing.
> `aik-py` detects AVB v2 and handles it explicitly. If you are migrating from
> the Bash toolkit, re-verify the signature state of any AVB v2 image you have
> previously repacked.

---

## Compression

All of these are handled natively by `aik-py` — no external binary is invoked.
Detection is by magic bytes only.

| Format | Magic | Extension | Default level |
|--------|-------|-----------|--------------:|
| gzip | `1F 8B` | `.gz` | 9 |
| bzip2 | `42 5A 68` | `.bz2` | 9 |
| xz | `FD 37 7A 58 5A 00` | `.xz` | 1 (CRC32) |
| lzma | `5D 00 00` | `.lzma` | 1 |
| lzop | `89 4C 5A 4F` | `.lzo` | — |
| lz4 (modern frame) | `04 22 4D 18` | `.lz4` | 9 |
| lz4-legacy frame | `02 21 4C 18` | `.lz4` | 9 |
| cpio (uncompressed) | `070701` | *(none)* | — |

Two details that are easy to get wrong and are handled explicitly:

- **xz uses a CRC32 check, not CRC64.** Both share the same six-byte magic; the
  check type lives in the stream flags (low nibble of the second byte: `0x01` =
  CRC32, `0x04` = CRC64). Some bootloaders only implement CRC32. The legacy
  scripts hard-code `-Ccrc32` for exactly this reason.
- **lz4 vs lz4-legacy is decided by frame format, never by filename.** Modern
  frames start `04 22 4D 18`, legacy frames `02 21 4C 18`; both are named
  `.lz4`. Swapping them yields a ramdisk the bootloader cannot inflate.

---

## cpio `newc`

Handled natively by a pure-Python writer — no `cpio` subprocess, no dependency
on a system cpio version. See
[architecture.md](architecture.md#cpio-newc) for the 110-byte ASCII header,
the `TRAILER!!!` terminator, and why uid/gid must be 0.

This removes the 2.12/2.13/2.15 hazard the Bash scripts warned about, where a
newer `cpio` could emit an archive the bootloader rejects.

---

## Summary: what still needs `bin/`

If you want a dependency-free install, these are the operations that will not
work:

| Operation | Blocker |
|-----------|---------|
| Sony ELF pack/unpack | `elftool`, `unpackelf` |
| DENX U-Boot pack/unpack | `dumpimage`, `mkimage` |
| Rockchip KRNL pack | `rkcrc` |
| Intel OSIP pack/unpack | `mboot` |
| PXA pack/unpack | `pxa-mkbootimg`, `pxa-unpackbootimg` |
| Loki patch/revert | `loki_tool` **+ your device's `aboot.img`** |
| AVB v1 signing | `boot_signer.jar` (needs a JVM) |
| BLOB signing | `blobpack` |
| CHROMEOS signing | `futility` |
| DHTB signing | `dhtbsign` |
| Sony SIN unpacking | `sony_dump` |

Everything else — all AOSP boot versions, `vendor_boot`, MTK, Amonet, NOOK, the
Bump and SEAndroid footers, every compression codec, and the whole cpio layer —
is native.
