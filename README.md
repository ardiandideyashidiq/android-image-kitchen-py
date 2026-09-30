# Android Image Kitchen — Python

A Python library and CLI for unpacking and repacking Android `boot.img` and
`recovery.img` files.

This is a native reimplementation of the original AIK Bash toolkit. Where the
Bash scripts scraped `file(1)` output and shelled out to prebuilt binaries,
`aik-py` parses the formats directly — across every AOSP boot header version,
`vendor_boot`, the MTK header, the Amonet patch layer, and the full cpio and
compression stack.

> **Status: the CLI does not exist yet.** `uv run aik-py` currently fails with
> `Failed to spawn: aik-py`. Everything below is the **target** interface, not a
> description of working code. Until it lands, use the legacy scripts — see
> [Legacy usage](#what-is-still-legacy).
>
> Some formats also still shell out to the prebuilt tools in `bin/` even once
> the CLI exists; see [docs/formats.md](docs/formats.md) for an honest
> per-format breakdown.

## Documentation

| Document | What it covers |
|----------|----------------|
| [docs/architecture.md](docs/architecture.md) | Header layouts, the layer stack, cpio `newc`, compression — the reverse-engineered detail |
| [docs/formats.md](docs/formats.md) | Which formats are native and which still need `bin/` |
| [docs/migration.md](docs/migration.md) | Moving from `unpackimg.sh` / `repackimg.sh` / `cleanup.sh`, and the bugs that were fixed |

## Installation

This is a [`uv`](https://docs.astral.sh/uv/) project.

```sh
uv sync
```

That creates the environment and installs the project plus its dev
dependencies. Everything below runs through `uv run`, so no separate activation
step is needed.

## Quick start

```sh
# 1. Unpack
uv run aik-py unpack boot.img

# 2. Edit whatever you need
$EDITOR ramdisk/default.prop

# 3. Repack
uv run aik-py repack

# 4. Flash image-new.img (or unsigned-new.img, see below)
```

### The output layout

`unpack` produces three things in the current directory:

```
split_img/          # the extracted image components
ramdisk/            # the unpacked ramdisk, ready to edit
manifest.json       # what was found, as typed JSON
```

- **`ramdisk/`** is the working directory. Edit files here in place; this is
  the directory that gets rebuilt on `repack`.
- **`split_img/`** holds the kernel, the compressed ramdisk archive, and any
  other image components pulled out of the container.
- **`manifest.json`** records the format, header version, compression, load
  addresses, and **every layer that was peeled off the image** — signature
  wrapper, patch layer, MTK header, footer.

The manifest is what makes `repack` symmetric with `unpack`: `repack` reads it
and re-applies exactly the layers `unpack` found, in reverse order. You do not
need to remember or specify which of these you are dealing with.

> **Note.** `manifest.json` is the typed replacement for the old
> `split_img/<image>-<field>` filename convention, where a field's *name* was
> the only schema — `split_img/boot.img-imgtype`, `boot.img-ramdiskcomp`, and
> so on. The manifest holds the same information with explicit types
> (`"format": "aosp"`, `"compression": "lz4-legacy"`) and without the filename
> collisions that came from naming fields after the source image. See
> [docs/migration.md](docs/migration.md#working-directory-state-split_imgimage-field--manifestjson).

### Output filenames

| Filename | When |
|----------|------|
| `image-new.img` | The final output — flashed to the device. Unsigned source, or the original signature was re-applied |
| `unsigned-new.img` | Intermediate, written only when the source had a signature wrapper that was stripped. Signing reads this and writes `image-new.img` |

The Bash logic is `if [ -f split_img/*-sigtype ]; then outname=unsigned-new.img;
else outname=image-new.img`. So the *existence* of a stripped signature is what
selects the intermediate name — both cases end with `image-new.img` present.

## Commands

### `aik-py unpack <image>`

Splits an image into `split_img/` and extracts the ramdisk into `ramdisk/`.

Detects and peels, in order: signature wrapper (AVB, BLOB, CHROMEOS, DHTB, NOOK,
NOOKTAB, SIN) → patch layer (Loki, Amonet) → image format → MTK header →
compression → cpio. Each peel is recorded in the manifest.

Refuses to overwrite an existing `split_img/` or `ramdisk/` — run
`aik-py clean` first, or clean them up by hand.

### `aik-py repack`

Rebuilds the ramdisk from `ramdisk/` and reassembles the image using the
manifest.

| Flag | Effect |
|------|--------|
| `--original` | Reuse the already-extracted ramdisk archive instead of rebuilding from `ramdisk/` |
| `--origsize` | Pad the output back to the original file size |
| `--level <0-9>` | Compression level |
| `--avbkey <name>` | AVB signing key (default `verity`) |
| `--forceelf` | Sony ELF: repack as ELF even with no RPM component present |

With no `--level`, the default depends on the codec — **xz uses level 1** and
lz4 uses level 9, matching the original toolkit. Kernel ramdisks are
size-constrained and decompress on a slow CPU, so aggressive xz settings hurt.

### `aik-py clean`

Removes `split_img/`, `ramdisk/` and any `*new.*` outputs.

| Flag | Effect |
|------|--------|
| `--quiet` | Do not print the confirmation line |

## What is still legacy

`unpackimg.sh`, `repackimg.sh` and `cleanup.sh` are **still present and still
work**. They are now considered legacy — prefer the Python CLI.

Two reasons you might still reach for them: they are the reference
implementation for format behaviour, and a handful of formats (Sony ELF, DENX
U-Boot, Rockchip KRNL, Intel OSIP, PXA, and all the tool-backed signature
wrappers) still shell out to the same prebuilt binaries either way.

```sh
./unpackimg.sh boot.img
./repackimg.sh
./cleanup.sh
```

## Known divergences from the Bash toolkit

Two behaviours are worth knowing about up front:

- **The `id` field.** The bundled AIK `mkbootimg` writes a hash at offset
  `0x240` that matches no AOSP formula we could reproduce. `aik-py` treats it as
  an opaque field — preserved on unpack, regenerated per AOSP spec on pack — so
  **our output may differ from the legacy tool in that one 20-byte field**
  (32 with SHA-256). Everything else should match. Details and the evidence are
  in [docs/architecture.md](docs/architecture.md#the-id-field).
- **v3/v4 keep the `ANDROID!` magic.** The AOSP spec says v3/v4 drop it and
  start with `header_version` at offset 0. The bundled `mkbootimg` does the
  opposite, and so does `aik-py` — version dispatch is by the u32 at `0x28`. If
  you are writing your own tooling, read
  [docs/architecture.md](docs/architecture.md#header-versions-3-and-4) first.

## Layout

```
src/aik_py/         the library
docs/               architecture, format matrix, migration guide
bin/                legacy prebuilt tools (still used by shimmed formats)
unpackimg.sh        legacy — still works
repackimg.sh        legacy — still works
cleanup.sh          legacy — still works
```

## License

The original AIK toolkit is by osm0sis @ xda-developers. See the repository
for details.
