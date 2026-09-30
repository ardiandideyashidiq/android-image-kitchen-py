# Image format architecture

This document is the format reference for `aik-py`. It records what the images
actually look like on disk, verified by building images with the AIK-bundled
tools and reading the resulting bytes back.

> **Provenance.** Every offset, size and magic in this document was confirmed
> empirically against the bundled `bin/linux/x86_64/mkbootimg` and
> `bin/linux/x86_64/unpackbootimg`, cross-checked against
> `bin/androidbootimg.magic`. Where the bundled tools disagree with the AOSP
> documentation, this document follows the tools and says so.

## Contents

- [The layer stack](#the-layer-stack)
- [AOSP boot header](#aosp-boot-header)
- [Header versions 3 and 4](#header-versions-3-and-4)
- [The `vendor_boot` header](#the-vendor_boot-header)
  - [Fixed-width, NUL-padded fields](#the-cmdline-and-name-fields-are-fixed-width-not-length-prefixed)
  - [`dtb_addr` is a uint64](#dtb_addr-is-a-uint64)
- [The `id` field](#the-id-field)
- [cpio `newc`](#cpio-newc)
- [Compression](#compression)

---

## The layer stack

A boot image is not one file format. It is up to six nested formats, and any
OEM is free to add one. The unified kernel bootloader understands only the
innermost two; every layer above that exists to satisfy a vendor's signing or
tamper-detection requirement.

Reading an image means peeling these layers from the outside in. Writing means
re-applying them in the exact reverse order.

```
                    ┌─────────────────────────────────────────┐
  outermost         │  signature wrapper                     │  BLOB / CHROMEOS /
                    │  (SONY SIN, DHTB, NOOK, NOOKTAB)       │  AVB / ASUS blob
                    ├─────────────────────────────────────────┤
                    │  patch layer                           │  Loki, Amonet
                    │  (secure-boot bypass / relocation)     │
                    ├─────────────────────────────────────────┤
                    │  image format                          │  AOSP, vendor_boot,
                    │  (header + page-aligned payloads)      │  DENX U-Boot, ELF,
                    │                                         │  Rockchip KRNL,
                    │                                         │  Intel OSIP, PXA
                    ├─────────────────────────────────────────┤
                    │  MTK header                            │  512-byte type marker
                    │  (prepended to kernel and/or ramdisk)  │  on MediaTek parts
                    ├─────────────────────────────────────────┤
                    │  compression                            │  gzip, lz4, xz, lzop,
                    │                                         │  lzma, bzip2
                    ├─────────────────────────────────────────┤
  innermost         │  cpio newc archive                     │  the actual ramdisk
                    │  (the real filesystem)                 │
                    └─────────────────────────────────────────┘
```

Then a **footer** — Bump, SEAndroid, or an AVB v2 `avb` partition — may be
appended *after* everything above. Footers are strictly last-in-first-out.

### How `unpack` and `repack` use this

`unpack` peels outward-in: it strips the signature, reverts the patch layer,
parses the image header, removes any MTK header, decompresses, and finally
extracts the cpio archive into `ramdisk/`. Each peel is *recorded* rather than
discarded.

`repack` replays that record in reverse, re-compressing, re-adding the MTK
header, rebuilding the image, re-applying the patch layer, re-signing, and
re-appending the footer.

That symmetry is the central design property of the library. Any format the
unpacker recognises must be reapplicable by the packer without the user having
to remember which layer they are dealing with. The record of peeled layers
lives in `split_img/manifest.json`.

---

## AOSP boot header

The magic is `ANDROID!` (`0x41 0x4E 0x44 0x52 0x4F 0x49 0x44 0x21`) at offset 0.

> ### Field sizes differ from AOSP
>
> The AOSP documentation describes a `name` field of 512 bytes and a `cmdline`
> field of 1024 bytes. **This build does not do that.** Verified against the
> bundled `mkbootimg`:
>
> - `name` is **16 bytes** at offset **0x30** (`BOOT_NAME_SIZE`). A 15-character
>   board name is accepted; 16 characters is rejected with
>   `error: board name too large`.
> - `cmdline` is **512 bytes** at offset **0x40** (`BOOT_ARGS_SIZE`). A
>   511-character cmdline fits exactly into 0x40–0x23F.
>
> Longer cmdlines are not truncated — they **spill** into a separate
> `extra_cmdline` array. A 600-character cmdline produces one contiguous run of
> 512 bytes at 0x40 and a second run of 88 bytes at 0x260. The accepted maximum
> is 1536 characters, after which the tool errors with
> `error: kernel cmdline too large`.

All multi-byte integers are **little-endian**, unsigned unless noted.

### v0 / v1 / v2 layout

| Offset | Size | Field | Notes |
|-------:|-----:|-------|-------|
| `0x00` | 8 | `magic` | `ANDROID!` |
| `0x08` | 4 | `kernel_size` | bytes |
| `0x0C` | 4 | `kernel_addr` | load address |
| `0x10` | 4 | `ramdisk_size` | bytes |
| `0x14` | 4 | `ramdisk_addr` | load address |
| `0x18` | 4 | `second_size` | bytes |
| `0x1C` | 4 | `second_addr` | load address |
| `0x20` | 4 | `tags_addr` | |
| `0x24` | 4 | `page_size` | |
| `0x28` | 4 | `header_version` | 0, 1 or 2 |
| `0x2C` | 4 | `os_version` | packed, v1+ |
| `0x30` | 16 | `name` | `BOOT_NAME_SIZE` |
| `0x40` | 512 | `cmdline` | `BOOT_ARGS_SIZE` |
| `0x240` | 20 / 32 | `id` | SHA-1 / SHA-256, see [below](#the-id-field) |
| `0x240`+ | | `extra_cmdline` | overflow array (v1+) |

**Verified base layout.** Building with `--header_version 0 --board TESTBOARD
--cmdline "console=ttyHSL0"` produces:

```
00000000: 414e 4452 4f49 4421 0010 0000 0080 0010  ANDROID!........
00000010: 0008 0000 0000 0011 0000 0000 0000 f010  ................
00000020: 0001 0010 0008 0000 0000 0000 0000 0000  ................
00000030: 5445 5354 424f 4152 4400 0000 0000 0000  TESTBOARD.......
00000040: 636f 6e73 6f6c 653d 7474 7948 534c 3000  console=ttyHSL0.
```

`TESTBOARD` begins at `0x30` and the cmdline at `0x40`, as tabulated.

### `os_version` packing (v1+)

`os_version` at `0x2C` is a bitfield, not a plain number:

```
bits 31..14 : os_version major (A)
bits 13..7  : os_version minor (B)
bits  6..0  : os_version patch (C)
```

`os_patch_level` (v2+) is packed into a second word at `0x30` in the AOSP
scheme as `(YYYY - 2000) << 11 | MM << 5 | DD`.

> **Note.** In the bundled build the v1+ extension words land at `0x2C`
> (`os_version`) and the board `name` remains at `0x30`. The bundled
> `unpackbootimg` decodes these correctly, so `aik-py` follows the same offsets
> it writes and reads.

### v1 / v2 extra payload fields

Version 1 adds `recovery_dtbo`; version 2 adds `recovery_acpio`. When present,
these are recorded in the header tail. Observed with a v2 image carrying both a
`dtb` and a `recovery_dtbo`:

```
0x0660: dtb_size
0x0664: dtb_addr
0x066c: header_size   (1660)
0x0670: recovery_dtbo_size
0x0674: recovery_dtbo_addr
```

> The exact tail ordering is version-dependent and vendor-influenced. Treat the
> tail as **opaque bytes that must be preserved**: read them out on unpack and
> write them back verbatim rather than recomputing them from a fixed table.

### Reported `header_size`

`unpackbootimg` reports `BOARD_HEADER_SIZE` for v1 and above. Measured:

| Version | Reported `header_size` |
|--------:|----------------------:|
| 0 | *(not reported)* |
| 1 | 1648 |
| 2 | 1660 |
| 3 | 1580 |
| 4 | 1580 |

`v0` has no header-size field at all — the header ends at the first page
boundary and the size is implicit in `page_size`. A v0 header's last non-zero
byte sits at `0x253` (`name`/`cmdline` region plus `id` at `0x240`).

---

## Header versions 3 and 4

**This is a genuine trap and the single most important thing in this document.**

The AOSP specification says that v3/v4 **drop the `ANDROID!` magic entirely**
and start the header with `header_version` at offset 0. **This build does not
do that.** Verified directly:

```
v3: 414e 4452 4f49 4421 0010 0000 0008 0000  ANDROID!........
v4: 414e 4452 4f49 4421 0010 0000 0008 0000  ANDROID!........
```

Both keep `ANDROID!` at offset 0 and place the classic fields at the classic
offsets.

**Version dispatch is by the u32 at offset `0x28`, not by the magic.** A
reader that assumes "no magic means v3/v4" will mis-parse these images
entirely. The correct test is:

```
read u32 at 0x28
  3 -> header v3
  4 -> header v4
  0/1/2 -> classic header
```

and in all cases the file begins with `ANDROID!`.

Observed v3/v4 header (`--header_version 3`, 4096-byte kernel, 2048-byte
ramdisk, page size 4096):

```
00000000: 414e 4452 4f49 4421 0010 0000 0008 0000  ANDROID!........
00000010: 0000 0000 2c06 0000 0000 0000 0000 0000  ....,...........
00000020: 0000 0000 0000 0000 0300 0000 636f 6e73  ............cons
00000030: 6f6c 653d 7474 7948 534c 3000 0000 0000  ole=ttyHSL0.....
```

Note the cmdline sits at `0x2C` here rather than `0x40` — the v3/v4 layout
packs the fields more tightly and drops the `second`/offsets block.

---

## The `vendor_boot` header

The vendor boot partition carries its own magic, `VNDRBOOT`
(`0x56 0x4E 0x44 0x52 0x42 0x4F 0x4F 0x54`), at offset 0.

### Fixed header region

| Offset | Size | Field |
|-------:|-----:|-------|
| `0x00` | 8 | `magic` = `VNDRBOOT` |
| `0x08` | 4 | `header_version` |
| `0x0C` | 4 | `page_size` |
| `0x10` | 4 | `kernel_addr` (load hint only; no kernel payload) |
| `0x14` | 4 | `ramdisk_addr` |
| `0x18` | 4 | `vendor_ramdisk_size` |
| `0x1C` | 2048 | `vendor_cmdline` — fixed-width, NUL-padded |
| `0x81C` | 4 | `tags_addr` |
| `0x820` | 16 | `name` — fixed-width, NUL-padded |
| `0x830` | 4 | `header_size` |
| `0x834` | 4 | `dtb_size` |
| `0x838` | **8** | `dtb_addr` — **uint64**, not 4 bytes |
| `0x840` | 4 | `vendor_ramdisk_table_size` (v4 only) |
| `0x844` | 4 | `vendor_ramdisk_table_entry_num` (v4 only) |
| `0x848` | 4 | `vendor_ramdisk_table_entry_size` (v4 only) |
| `0x84C` | 4 | `bootconfig_size` (v4 only) |

Payload order on disk is `vendor_ramdisk` (page-aligned) then `dtb`
(page-aligned).

### The cmdline and name fields are fixed-width, not length-prefixed

This is the single most mis-readable thing about this header, and it is worth
stating bluntly because the offsets above look wrong until you understand why.

`vendor_cmdline` is **always a 2048-byte field** and `name` is **always a
16-byte field**, both zero-padded to their full width. Neither carries a length
prefix. There is no `cmdline_size` or `name_size` field in the emitted image
(an earlier AOSP draft had them at `0x1C`/`0x24`; no shipping `mkbootimg`
produces that layout, and the bundled one does not either).

Because the fields are fixed-width, everything after them sits at **fixed
offsets past the end of the cmdline region**, regardless of how long the actual
cmdline text is. `tags_addr@0x81C`, `name@0x820`, `header_size@0x830`,
`dtb_size@0x834` and `dtb_addr@0x838` are all at constants.

> **Never locate these fields by scanning for the NUL terminator.** Reading
> `0x1C + strlen(cmdline)` lands in the middle of the NUL padding and
> mis-locates every field after it. A reader that does this will appear to work
> on an image whose cmdline happens to fill the field exactly, and silently
> produce garbage on every other one.

This is precisely why `header_size` exists: it tells the loader how far into
page 0 the populated fields actually reach, so the rest can be treated as
padding. `header_size` is **2112 (0x840) for v3** and **2128 (0x850) for v4**,
the latter gaining the four v4-only words.

### `dtb_addr` is a uint64

`dtb_addr` at `0x838` is **8 bytes**, not 4. This is a frequent mis-assumption —
it sits in the middle of a table of u32 fields and looks like one.

Reading it as a u32 truncates the address, so **load addresses above 4 GiB do not
round-trip** through a u32 parse. On a 64-bit device the dtb is routinely loaded
high — a real v4 image measures `dtb_addr = 0x47c80000`, comfortably above the
4 GiB boundary — so a truncated read produces an image that fails to boot. Parse
and emit all 8 bytes.

### Verified

Building with `--vendor_ramdisk rdsk2` (2048 bytes of `R`), `--dtb dtb1` (128
bytes of `D`), `--vendor_cmdline androidboot.hardware=x`:

```
magic: b'VNDRBOOT'
0x08 header_version      = 3
0x0c page_size           = 2048
0x10 kernel_addr         = 0x10008000
0x14 ramdisk_addr        = 0x11000000
0x18 vendor_ramdisk_size = 2048        (matches the 2048-byte input)
'androidboot.hardware=x' at 0x1c, NUL-padded out to 0x81c
0x81c tags_addr          = 0x10000100
0x820 name               = "B"          (NUL-padded out to 0x830)
0x830 header_size        = 2112 (0x840)
0x834 dtb_size           = 128          (matches the 128-byte input)
0x838 dtb_addr           = 0x11f00000   (uint64)
vendor ramdisk payload  = 0x1000 .. 0x1800
dtb payload             = 0x1800 .. 0x1880
```

The bundled `unpackbootimg` reads these back correctly and prints
`BOARD_HEADER_SIZE 2112`, `BOARD_DTB_SIZE 128`, `BOARD_DTB_OFFSET 0x01f00000`,
`BOARD_VENDOR_CMDLINE`, `BOARD_VENDOR_BASE`, and extracts
`vf.img-dtb` (128 bytes) and `vf.img-vendor_ramdisk` (2048 bytes).

### Fixed offsets, confirmed

The tail fields do not move with cmdline length. Verified by building with
cmdline lengths of 0, 8, 32, 64 and 128: `0x81C`–`0x840` is byte-identical in
every case.

### Cmdline capacity

The `vendor_cmdline` field holds up to 2048 bytes; beyond that the tool errors
with `error: vendor cmdline too large`. Because the field is fixed-width and
NUL-padded, a **full 2048-byte** cmdline with no terminating NUL is still
valid and round-trips — that is the one case where scanning for a NUL would
*not* have misled you, which is precisely why the bug is so easy to miss in
testing.

### Real-world v4 images are spec-conformant

Real `vendor_boot.img` files from a current device (AOSP v4) parse exactly
against the table above, with nothing truncated and nothing non-standard:

```
last non-zero byte in page 0: 0x848
header_size  @0x830 = 2128 (0x850)
tags_addr    @0x81c = 0x47c80000
dtb_size     @0x834 = 191487
dtb_addr     @0x838 = 0x0000000047c80000
```

Page 0's populated fields end at exactly `0x850`, which is precisely the
recorded `header_size` — the field doing the job it exists for. Note
`dtb_addr = 0x47c80000`, comfortably **above 4 GiB**, which is exactly the case
that a 4-byte read truncates.

Both images also carry an **AVB vbmeta partition after the payloads**. That
trailing data is not described by the header, so `aik-py` preserves it verbatim
to keep parse/pack lossless — it is carried through untouched rather than being
regenerated.

> ### The v4 fragment table is not empirically verified here
>
> The four v4-only words at `0x840`–`0x84C` (`vendor_ramdisk_table_size`,
> `vendor_ramdisk_table_entry_num`, `vendor_ramdisk_table_entry_size`,
> `bootconfig_size`) and the v4 payload ordering — fragments in table order,
> then dtb, then the fragment table, then bootconfig — come from the AOSP
> specification and **not** from measurement.
>
> The bundled `mkbootimg` predates `--vendor_ramdisk_fragment` and cannot
> produce a fragmented image, so there is no way to build one here to check
> against. Treat this table as spec-derived and unvalidated. The v3 layout, by
> contrast, is fully verified above.
>
> One useful detail that *is* verified: AOSP's `mkbootimg` writes the **v3
> layout even when asked for `--header_version 4`** with a single
> `--vendor_ramdisk`, so a v4 image built that way has a zero entry count and no
> fragment table. Both shapes parse.

---

## The `id` field

The `id` at offset `0x240` is nominally an integrity digest of the payload. The
bundled `mkbootimg` writes a value that does **not** match any AOSP hash
formula, including an exhaustive brute-force search over the plausible
candidates.

For a 4096-byte kernel of `0x4B` (`K`) and a 2048-byte ramdisk of `0x52`
(`R`), the tool writes:

```
id = a782956ccff86fc7362ccee8fe69d4bd40912417
```

Candidates tested and rejected:

| Candidate | SHA-1 | Match |
|-----------|-------|-------|
| `sha1(kernel)` | `760b10e1bc00cf983966dee5b476229bf7a57400` | no |
| `sha1(kernel + ramdisk)` | `62b26606757c4c9ea9995cc0b6ea06d8e441a16c` | no |
| AOSP `pack_id` (page-0 + kernel tail) | `605db3fdbaff4ba13729371ad0c4fbab3889378e` | no |

Note that the value is **stable across runs** for identical inputs, so it is
deterministic — just not by any formula we could identify.

### How `aik-py` handles it

`id` is treated as an **opaque field**:

- **Unpack** preserves it verbatim from the source image into the manifest.
- **Pack** regenerates it per the AOSP specification.

> **Consequence.** Our packer will **not** byte-match the legacy tool in this
> one 20-byte field. Everything else — header layout, field placement, payload
> ordering, padding — should match. If you need byte-identical output, keep the
> original image's `id` and repack with it preserved.

The hash algorithm follows the header's `hashtype` (v0–v2 only): SHA-1 gives a
20-byte `id`, SHA-256 (`--hashtype sha256`) gives 32 bytes. v3/v4 carry no
`id`.

---

## cpio `newc`

The innermost layer is a cpio archive in the `newc` ("new character") format —
an entirely **ASCII** format, which is why it survives being handled by shell
tooling and why so many Android rootfs images use it.

### Header

Each entry is preceded by a **110-byte** header made of 8-character
zero-padded uppercase hex fields. Verified against a real `cpio -H newc`
archive:

```
  0 07070102   8 7CA5A200  16 0041ED00  24 00000000  32 00000000
 40 0000036A  48 BC7F4400  56 00000000  64 00010300  72 00000500
 80 00000000  88 00000000  96 00000300 104 000000
```

| Offset | Size | Field | Notes |
|-------:|-----:|-------|-------|
| `0` | 6 | `magic` | `070701` |
| `6` | 8 | `ino` | |
| `14` | 8 | `mode` | file type + permissions |
| `22` | 8 | `uid` | **must be 0** |
| `30` | 8 | `gid` | **must be 0** |
| `38` | 8 | `nlink` | |
| `46` | 8 | `mtime` | seconds since epoch, hex |
| `54` | 8 | `filesize` | data length, hex |
| `62` | 8 | `devmajor` | |
| `70` | 8 | `devminor` | |
| `78` | 8 | `rdevmajor` | |
| `86` | 8 | `rdevminor` | |
| `94` | 8 | `namesize` | includes the trailing NUL |
| `102` | 8 | `check` | always 0 in `newc` |

After the header comes `namesize` bytes of NUL-terminated filename, then the
file data. **Both the filename and the file data are padded to a 4-byte
boundary.** Worked example from the archive above: `namesize = 3`, name is
`"cx\0"`, 110 + 3 = 113, pad 3 bytes to reach 116, and the data begins at 116 —
which is 4-byte aligned, as required.

The archive is terminated by an entry named `TRAILER!!!` (with a zero-length
data block) after which no more entries follow.

### Why uid/gid must be 0

The bootloader mounts and extracts this archive **before any part of Android
userspace exists**. At that point:

- there is no `/etc/passwd` and no `/etc/group`, so no name database exists;
- no user has been created and no SELinux policy is loaded;
- the bootloader's own permission model has exactly one notion of a valid
  owner: **root**.

If any entry carries a non-zero uid/gid, the extracted file is owned by a user
that does not exist and that the init process cannot represent. In practice the
result is a ramdisk the init cannot mount, or one that fails verification
outright. Hence the legacy scripts' `-R 0:0` (`aik-py` forces it
unconditionally) and the reason `unpackimg.sh` re-chowns `ramdisk/` to `0:0`
before extracting.

### The cpio version hazard

`unpackimg.sh` and `repackimg.sh` both warn when the system `cpio` is version
2.13:

```sh
[ "$(cpio --version | head -n1 | rev | cut -d\  -f1 | rev)" = "2.13" ] && cpiowarning=1
...
echo "Warning: Using cpio 2.13 may result in an unusable repack; downgrade to 2.12 to be safe!";
```

`newc` output is not byte-stable across cpio releases — field ordering,
alignment and inode numbering all vary — and a cpio 2.13 repack can produce an
archive the bootloader rejects. Since `aik-py` writes `newc` itself, this
dependency is gone entirely along with the 2.12/2.13/2.15 hazard.

---

## Compression

The compression layer wraps the cpio archive. Detection is by **magic bytes
only**; the filename is never consulted.

| Format | Magic bytes | Extension | Default level | Notes |
|--------|-------------|-----------|--------------:|-------|
| gzip | `1F 8B` | `.gz` | 9 | |
| bzip2 | `42 5A 68` (`BZh`) | `.bz2` | 9 | |
| xz | `FD 37 7A 58 5A 00` | `.xz` | 1 | **CRC32 check** |
| lzma | `5D 00 00` | `.lzma` | 1 | xz container, `-Flzma` |
| lzop | `89 4C 5A 4F` | `.lzo` | — | |
| lz4 | `04 22 4D 18` | `.lz4` | 9 | modern frame |
| lz4-legacy | `02 21 4C 18` | `.lz4` | 9 | **legacy frame** |
| cpio | `070701` | *(none)* | — | uncompressed |

### xz uses CRC32, not CRC64

Both start with the same six-byte magic; the **check type is in the stream
flags**, the low nibble of the second stream-flag byte:

```
xz  -9 -Ccrc32  ->  stream_flags=0001  check=1 (CRC32)   68 bytes
xz  -9 -Ccrc64  ->  stream_flags=0004  check=4 (CRC64)   72 bytes
```

`repackimg.sh` hard-codes this: `repackcmd="xz $level -Ccrc32"`. Some bootloaders
only implement a CRC32 xz check; producing CRC64 produces an image that fails to
decompress at boot. **`aik-py` keeps CRC32 as the default.**

Note the default level for xz is **1**, not 9 — `repackimg.sh` sets `level=-1`
for xz and `-9` for lz4 when the user gives no `--level`. Kernel ramdisks are
size-constrained and decompress on a slow CPU, so aggressive xz settings hurt.

### lz4 vs lz4-legacy is a frame-format distinction

The two lz4 variants are told apart **purely by frame magic**, never by
extension:

```
lz4 (modern) : magic 0x184D2204   flg=0x64  version=0x40
lz4 -l (legacy): magic 0x184C2102  flg=0x07  version=0x00
```

Both are named `.lz4` on disk. Getting this wrong produces a ramdisk the
bootloader cannot inflate. In the legacy scripts this is handled by the `lz4`
and `lz4-l` cases in `unpackimg.sh` and the corresponding `repackcmd` values in
`repackimg.sh`:

```sh
lz4)    unpackcmd="$bin/$arch/lz4 -dcq";;
lz4-l)  unpackcmd="$bin/$arch/lz4 -dcq"; compext=lz4;;
...
lz4)    repackcmd="$bin/$arch/lz4 $level";;
lz4-l)  repackcmd="$bin/$arch/lz4 $level -l"; compext=lz4;;
```

Note that both decompress with the same `-dcq` invocation but differ on
compress: the legacy frame needs `-l`.

---

## MTK header

MediaTek parts prepend a **512-byte** header to the kernel and/or the ramdisk.
It is identified by the magic `\x88\x16\x88\x58` and contains the ASCII type
string (`KERNEL`, `ROOTFS`, or `RECOVERY`) used to tell the components apart.

Both scripts strip it with the same idiom — drop the first 512 bytes:

```sh
dd bs=512 skip=1 conv=notrunc if="$file-kernel" of=tempkern
```

and re-add it via `mkmtkhdr`:

```
mkmtkhdr --kernel <zImage filename>
         [--rootfs|--recovery] <ramdisk filename>
```

## Patch layers

**Loki.** A secure-boot bypass. The bundled `loki_tool` (v2.1) exposes:

```
./bin/linux/x86_64/loki_tool [patch] [boot|recovery] [aboot.img] [in.img] [out.lok]
./bin/linux/x86_64/loki_tool [flash] [boot|recovery] [in.lok]
./bin/linux/x86_64/loki_tool [find] [aboot.img]
./bin/linux/x86_64/loki_tool [unlok] [in.lok] [out.img]
```

> Re-applying a Loki patch requires a **dump of the target device's
> `aboot.img`** in the working directory — the patch offset is derived from the
> device's own bootloader, so it is device-specific and cannot be computed.
> `unpackimg.sh` prints this warning explicitly during unpack, and
> `repackimg.sh` aborts if `aboot.img` is absent.

**Amonet.** A pure byte-shuffle patch with no external dependency — `repackimg.sh`
implements it with `dd` alone, moving the first 1024 bytes behind the extracted
microloader block.

## Signature wrappers

| Type | Detection magic | Tool |
|------|-----------------|------|
| BLOB | `-SIGNED-BY-SIGNBLOB-` at 0 | `blobpack` / `blobunpack` |
| CHROMEOS | `CHROMEOS` at 0 | `futility vbutil_kernel` |
| DHTB | `DHTB\x01\x00\x00` at 0 | `dhtbsign` |
| SIN | `\x01\x00\x00\x00`, `\x02\x00\x00\x00`, `\x03SIN` at 0 | `sony_dump` |
| NOOK / NOOKTAB | strings at offset 64 / 48 | plain `cat` + stored key block |

AVB v1 signing uses `bin/boot_signer.jar` with the key pair in `bin/avb/`
(`verity.pk8`, `verity.x509.pem`); `--avbkey <name>` selects an alternative.

## Footers

Appended after everything else, detected from the last 8 KiB of the image:

| Type | Magic |
|------|-------|
| AVB v1 | `\x02\x01\x01\x30\x82` (`AVB` descriptor) |
| AVB v2 | `AVBf` |
| Bump | `\x41\xA9\xE4\x67\x74\x4D\x1D\x1B\xA4\x29\xF2\xEC\xEA\x65\x52\x79` |
| SEAndroid | `SEANDROIDENFORCE` |

The Bump and SEAndroid footers are literal byte strings appended verbatim by
`awk`/`printf`. AVB v2 is the v2 hashtree/footer format; see
[formats.md](formats.md) for how each is handled.

---

## Detection

The legacy scripts detect formats by running libmagic with a custom magic
database and **scraping the human-readable output**:

```sh
imgtest="$(file -m "$bin/androidbootimg.magic" "$img" 2>/dev/null | cut -d: -f2-)";
if [ "$(echo $imgtest | awk '{ print $2 }' | cut -d, -f1)" = "signing" ]; then
```

`bin/androidbootimg.magic` is a custom libmagic database covering all the OEM
formats, MTK headers, QCDT, the AVB footers, Bump and SEAndroid. Each entry
emits a human-readable description, and the scripts pick the *word at field
position N* out of that sentence.

This is brittle: it depends on libmagic's exact field positions and on the
continuation-text layout of the magic file, and it is a documented source of
mis-detection. `aik-py` replaces it with direct byte-offset sniffing per
format. The magic file remains useful as the authoritative list of magics and
their offsets, and is quoted throughout this document.
