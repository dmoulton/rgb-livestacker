"""Standalone RGB live-stacker for a mono camera cycling through filters.

Watches a capture folder (or replays an existing one), routes each sub to a
per-filter running stack, aligns every frame to a common reference, combines
the channels into a live colour preview, and serves that preview to a browser
with linked black-point / midtones / gamma / contrast controls.
"""

__version__ = "0.1.0"
