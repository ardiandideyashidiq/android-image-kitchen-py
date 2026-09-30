# Migrating from the Bash toolkit

`aik-py` replaces `unpackimg.sh`, `repackimg.sh` and `cleanup.sh` with a single
Python CLI. This page maps the old commands and flags onto the new ones, and
lists the bugs that were deliberately fixed along the way.

The legacy scripts are still in the repository and still work. Nothing is
removed — but new work should target `aik-py`.

---

## Command mapping

| Bash | Python |
|------|--------|
| `./unpackimg.sh <image>` | `aik-py unpack <image>` |
| `./repackimg.sh` | `aik-py repack` |
| `./cleanup.sh` | `aik-py clean` |

The default working directory is the current directory, and `unpack` creates
`split_img/` and `ramdisk/` exactly as the Bash did. See
[README.md](../README.md) for the full command reference.

---

## Flag mapping

### `unpackimg.sh` → `aik-py unpack`

The Bash usage line is:

```
usage: unpackimg.sh [--local] [--nosudo] <file>
```

| Flag | Equivalent | Notes |
|------|------------|-------|
| `--local` | *(default)* | **See the bug below.** In the Bash, `--local` and `--nosudo` were parsed in two *separate* `case` statements, so `--nosudo --local` silently dropped `--local`. In `aik-py`, `unpack` always operates on the current directory, which is what `--local` meant. |
| `--nosudo` | *(default)* | The Bash re-entered `sudo` before `chown 0:0 ramdisk` and `cpio`. `aik-py` forces uid/gid 0 without needing root. |
| `--sudo` | — | Accepted and ignored by the Bash. Not needed; `aik-py` never calls `sudo`. |
| `<file>` | `aik-py unpack <file>` | Required. |

### `repackimg.sh` → `aik-py repack`

The Bash usage line is:

```
usage: repackimg.sh [--local] [--original] [--origsize] [--level <0-9>] [--avbkey <name>] [--forceelf]
```

| Flag | Equivalent | Notes |
|------|------------|-------|
| `--local` | *(default)* | Same two-`case` parsing bug as `unpackimg.sh`. Current directory is the default in `aik-py`. |
| `--original` | `aik-py repack --original` | Reuse the extracted `ramdisk.cpio*` untouched instead of rebuilding from `ramdisk/`. |
| `--origsize` | `aik-py repack --origsize` | Pad the output with `truncate -s <origsize>` back to the original file size. The Bash stored the source size in `split_img/<image>-origsize`; `aik-py` records it in the manifest. |
| `--level <0-9>` | `aik-py repack --level <n>` | Compression level. The Bash accepted digits `0`–`9` and passed them as `-<n>`. Note the **non-numeric-silently-swallowed bug below**. |
| `--avbkey <name>` | `aik-py repack --avbkey <name>` | AVB signing key stem. The Bash searched `$2`, `$cur/$2`, `$aik/$2` for `<name>.pk8` + `<name>.x509.*`. Default is `bin/avb/verity`. |
| `--forceelf` | `aik-py repack --forceelf` | For Sony ELF: repack as ELF even without an RPM component present. |

### `cleanup.sh` → `aik-py clean`

The Bash usage line is:

```
usage: cleanup.sh [--local] [--quiet]
```

| Flag | Equivalent | Notes |
|------|------------|-------|
| `--local` | *(default)* | |
| `--quiet` | `aik-py clean --quiet` | Suppress "Working directory cleaned." |
| — | `aik-py clean` | Removes `ramdisk/`, `split_img/` and `*new.*`. |

---

## Working-directory state: `split_img/<image>-<field>` → `manifest.json`

The Bash scripts recorded unpack state as a pile of one-word files in
`split_img/`, named `<original-image-name>-<field>`. For an image called
`boot.img`, unpack produced files such as:

```
split_img/boot.img-imgtype          # "AOSP", "AOSP_VNDR", "ELF", ...
split_img/boot.img-origsize         # original file size in bytes
split_img/boot.img-ramdiskcomp      # detected compression, e.g. "lz4-l"
split_img/boot.img-cmdline          # kernel command line
split_img/boot.img-board            # board name
split_img/boot.img-header_version   # 0..4
split_img/boot.img-sigtype          # AVBv1 / BLOB / CHROMEOS / DHTB / ...
split_img/boot.img-lokitype         # Loki patch type
split_img/boot.img-mtktype          # MTK component type
split_img/boot.img-ramdisk.cpio.gz  # the extracted ramdisk archive
split_img/boot.img-kernel           # the kernel
```

Repack then re-read that pile with globs — `ls *-kernel`, `cat *-imgtype`,
`cat split_img/*-*ramdiskcomp` — to rebuild the command line. The field name
*was* the schema; there was no schema document and no validation.

`aik-py` replaces this with a single typed **`split_img/manifest.json`**. The
information content is the same, but it is explicit, validated, and typed:

```json
{
  "image": "boot.img",
  "format": "aosp",
  "header_version": 4,
  "page_size": 4096,
  "original_size": 67108864,
  "kernel": "boot.kernel",
  "ramdisk": {
    "compression": "lz4-legacy",
    "archive": "boot.ramdisk.cpio.lz4",
    "level": 9
  },
  "cmdline": "console=ttyHSL0,115200,n8 androidboot.hardware=qcom",
  "board": "qcom",
  "layers": {
    "signature": null,
    "patch": null,
    "mtk": null,
    "footer": null
  }
}
```

Benefits:

- **Typed.** Compression is `lz4-legacy`, not the string `lz4-l`; the format is
  `aosp`, not `AOSP`. No string parsing.
- **Self-describing.** The manifest is one file to read instead of a directory
  listing with implicit globs.
- **Validated.** A manifest missing a field the packer needs is an error at
  unpack time, not a silently empty variable at repack time.
- **No filename collisions.** The Bash derived field names from the original
  image name, so two images in one directory clobbered each other's state.

> The manifest is the single source of truth for what was peeled. Repack reads
> it to decide which layers to re-apply and in what order. See
> [architecture.md](architecture.md#the-layer-stack).

---

## Bugs deliberately fixed

These are all real defects in the Bash toolkit, verified by reading the scripts.
They are listed because they are behaviours you may have **worked around** —
if you learned to always pass `--local` first, or to never trust an AVB v2
repack, that workaround is now unnecessary.

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | **AVB v2 detected but never re-signed.** `unpackimg.sh` matches `AVB*` in the tail test but only writes `-sigtype` for `*v1`. `repackimg.sh` gates all signing on that file existing. | **An AVB v2 image was repacked silently unsigned — a potential brick on verified-boot-enforcing devices.** | AVB v2 is detected and handled explicitly. |
| 2 | `abort()` printed `"Error!"` **without exiting**. | Every call site needed a manual `exit 1` after `abort`. | `abort` raises; no manual exit needed. |
| 3 | `--help` exited with status **1**. | `--help` reported failure in scripts and CI. | `--help` exits 0. |
| 4 | `--nosudo --local` silently dropped `--local`. | Flags parsed in two separate `case` statements, so only one of the two could take effect. | Single argument parser; order-independent. |
| 5 | `--level` with a non-numeric value was silently swallowed. | A typo like `--level high` fell back to the default level with no warning. | Invalid values are a hard error. |
| 6 | Pipeline exit status was the last command's. | `$?` reflected the tail of `find \| cpio \| compressor`, so decompressor failures were hidden behind a successful `cpio`. | Each stage's status is checked. |
| 7 | A `dd` used `skip=512 iflag=skip_bytes` with `bs=4096`. | `skip_bytes` makes `skip` a byte count, so this skipped 512 **bytes** — but the surrounding `bs=4096` idiom reads as if it skipped 512 *blocks*. Genuinely confusing, and a trap for anyone editing the line. | Offsets are explicit byte counts, not block counts. |
| 8 | Format detection scraped `file`'s human-readable output with `awk '{print $N}'`. | Relied on libmagic continuation-field positions; a libmagic version or magic-file change silently shifts the fields and mis-detects formats. | Direct byte-offset sniffing; no scraping. |

### Notes on the two most severe

**#1, the AVB v2 gap.** The detection code writes the signature type only for
v1:

```sh
case $tailtype in
  AVB*)
    case $tailtype in
      *v1) echo $tailtype > "$file-sigtype"; ...
    esac;
  ;;
```

and packing is gated on that file:

```sh
if [ -f split_img/*-sigtype ]; then
  ... sign ...
fi
```

An AVB v2 image therefore produces no `-sigtype`, the signing block is skipped,
and the output is unsigned with no warning printed. If you have repacked AVB v2
images with the Bash toolkit, treat those outputs as suspect.

**#7, the `dd` byte/block trap.** In `unpackimg.sh`:

```sh
DHTB) dd bs=4096 skip=512 iflag=skip_bytes conv=notrunc if="$img" of="$file" ;;
```

`iflag=skip_bytes` means `skip=512` is 512 **bytes**, not 512 × 4096-byte
blocks. The `bs=4096` is irrelevant to the skip arithmetic but strongly
suggests block semantics. The two nearby `skip_bytes` uses (`skip=8` for the
Rockchip KRNL ramdisk, and a computed `filesize - 8192` for the AVB footer
scan) are genuinely byte offsets and are correct — but they read as blocks.

---

## What is unchanged

The external contract is deliberately preserved so existing muscle memory and
any surrounding scripts keep working:

- **The working directory layout is the same.** `split_img/` and `ramdisk/`,
  with extracted components alongside the manifest.
- **The output filename is the same.** `image-new.img` when the source was
  unsigned, `unsigned-new.img` when a signature was stripped and re-applied.
- **The same external tools are used for shimmed formats**, from the same
  `bin/` directory, with the same arguments.
- **The compression defaults are the same** — including xz at CRC32 and the
  level-1 default. Images produced by `aik-py` should otherwise be
  byte-compatible with the Bash toolkit's output, with the caveat that the
  `id` field is regenerated per AOSP spec and may not match what the bundled
  `mkbootimg` wrote (see
  [architecture.md](architecture.md#the-id-field)).

## What is different

- **No `sudo`.** `aik-py` never escalates. It forces uid/gid 0 when writing cpio
  rather than requiring root, which is also why `--nosudo`/`--sudo` are gone.
- **No `cpio` subprocess.** The cpio `newc` layer is written in pure Python, so
  the 2.12/2.13/2.15 output-stability hazard is gone.
- **No libmagic scraping.** Formats are detected by byte offset.
- **Loki still needs your `aboot.img`.** This is a genuine device dependency,
  not a legacy artifact — the patch offset comes from your own bootloader.
