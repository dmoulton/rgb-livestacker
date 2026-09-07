# RGB Livestacker

A standalone, cross-platform live-stacking tool for a **mono camera cycling
through RGB filters**. It watches a capture folder, routes each sub to a
per-filter running stack, aligns every frame to a common reference, combines
the channels into a live colour preview, and serves that preview to a browser
with **linked** black-point / midtones / gamma / contrast controls — adjustments
on the *combined* image, not per channel.

See [`docs` design notes](../../Pictures/astro/nina/docs/rgb-livestacker-design-notes.md)
for the reasoning behind the design.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Replay an existing session (no rig needed — great for testing):

```bash
python -m livestacker --replay "/path/to/old/session" --interval 1.5
```

Watch a live capture folder:

```bash
python -m livestacker --watch "/path/to/capture/folder"
```

Then open <http://localhost:8080/>.

### Options

| flag | meaning |
|---|---|
| `--watch DIR` / `--replay DIR` | live folder watch, or replay an existing one |
| `--interval N` | replay pacing, seconds per frame (default 2) |
| `--limit N` | replay only the first N frames |
| `--host 0.0.0.0` | expose to the LAN (default: localhost only) |
| `--web-port` / `--ws-port` | ports (default 8080 / 8081) |
| `--dark PATH` | master dark: a FITS file, or a folder of dark subs to median-combine |
| `--flat PATH` | master flat(s): a FITS file, or a folder of raw flats (grouped by `FILTER`, one master per filter) |
| `--no-hot-pixel` | disable per-frame hot/cold pixel correction (on by default) |
| `--hot-pixel-sigma N` | hot-pixel detection threshold, robust sigma (default 8) |
| `--no-background` | disable gradient/background subtraction |
| `--no-auto-wb` | disable automatic per-channel white balance |
| `--background-box N` | background mesh box size in px (default 128) |
| `--preview-px N` | longest preview edge sent to the browser (default 1600) |
| `-v` | verbose logging |

## How it works

1. **Watch / replay** the folder; wait for each FITS file to finish writing.
2. **Calibrate**, per light frame:
   - master-dark subtraction (`--dark`), validated against the light's
     exposure / gain / binning / temperature, with warnings on a mismatch;
   - flat-field division (`--flat`), one master flat per `FILTER`, applied to
     the matching channel (warns + skips on a shape mismatch);
   - a per-frame hot/cold pixel cosmetic correction — needs no calibration
     frames, and is what stops hot pixels from smearing into diagonal "walking
     noise" as the mount dithers.
3. **Route** by the `FILTER` header keyword into a per-channel accumulator.
4. **Register** every frame — any filter — to the first frame received, with
   `astroalign` (triangle star matching, no plate solve). This is what keeps
   the independently-growing channel stacks from drifting apart.
5. **Stack**: per-pixel running mean, tracking per-pixel coverage so
   partially-overlapping (dithered / flipped) frames still contribute. The
   combined result is cropped to the region every channel covers, so the
   ragged dither border never appears.
6. **Background**: a `photutils` `Background2D` surface per channel —
   sigma-clipped per mesh cell, with bright sources and the frame border
   masked out, so the nebula does not bias its own background. Gated on a
   minimum sub count, cached and refreshed every few frames.
7. **White balance**: one scalar per channel matching bright-star levels.
8. **Combine** to linear RGB.
9. **Linked stretch**: one black/midtones/gamma/contrast curve through all
   three channels, applied live in the browser without re-stacking.

## Status / roadmap

Working. Not yet done: rolling sigma-clip rejection, star-flux-ratio white
balance, LRGB (luminance) support, meridian-flip robustness, PyInstaller
packaging.
