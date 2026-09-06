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
| `--no-hot-pixel` | disable per-frame hot/cold pixel correction (on by default) |
| `--hot-pixel-sigma N` | hot-pixel detection threshold, robust sigma (default 8) |
| `--no-background` | disable gradient/background subtraction |
| `--no-auto-wb` | disable automatic per-channel white balance |
| `--background-sigma N` | background blur radius in px (default 60) |
| `--preview-px N` | longest preview edge sent to the browser (default 1600) |
| `-v` | verbose logging |

## How it works

1. **Watch / replay** the folder; wait for each FITS file to finish writing.
2. **Calibrate**: optional master-dark subtraction (`--dark`, validated against
   the light's exposure / gain / binning / temperature, with warnings on a
   mismatch), then a per-frame hot/cold pixel cosmetic correction — the latter
   needs no calibration frames and is what stops hot pixels from smearing into
   diagonal "walking noise" as the mount dithers.
3. **Route** by the `FILTER` header keyword into a per-channel accumulator.
4. **Register** every frame — any filter — to the first frame received, with
   `astroalign` (triangle star matching, no plate solve). This is what keeps
   the independently-growing channel stacks from drifting apart.
5. **Stack**: per-pixel running mean, tracking per-pixel coverage so
   partially-overlapping (dithered / flipped) frames still contribute.
6. **Background**: subtract a large-radius Gaussian blur per channel (gated on
   a minimum sub count).
7. **White balance**: one scalar per channel matching bright-star levels.
8. **Combine** to linear RGB.
9. **Linked stretch**: one black/midtones/gamma/contrast curve through all
   three channels, applied live in the browser without re-stacking.

## Status / roadmap

Working first pass. Not yet done: rolling sigma-clip rejection, amp-glow /
column-defect handling (a matched master dark is the real fix), `photutils`
`Background2D`, star-flux-ratio white balance, LRGB (luminance) support,
meridian-flip robustness, PyInstaller packaging.
