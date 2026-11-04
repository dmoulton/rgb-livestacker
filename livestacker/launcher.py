"""A small cross-platform launcher window (Tkinter, no extra dependencies).

Pick the capture folder and options, press Start, and it runs
`python -m livestacker …` as a child process with its log shown in the window.
Choices are remembered between runs in `launcher-settings.json` next to the repo.

Opened by running the module with no arguments (`python -m livestacker`).
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

SETTINGS = Path(__file__).resolve().parent.parent / "launcher-settings.json"

DEFAULTS = {
    "mode": "watch",
    "object_name": "",
    "object_auto": False,      # the name was filled in from the folder, not typed
    "interval": 2.0,
    "catch_up": True,
    "bias": "",
    "dark": "",
    "flat": "",
    "crop_top": 0,
    "crop_bottom": 0,
    "crop_left": 0,
    "crop_right": 0,
    "hot_pixel": True,
    "sigma_clip": True,
    "sigma_clip_kappa": 3.0,
    "background": True,
    "auto_wb": True,
    "narrowband": True,
    "preview_px": 1600,
    "share_lan": False,
    "web_port": 8080,
    "ws_port": 8081,
    "open_browser": True,
    "max_viewers": 16,
    "osc_debayer": "auto",
    "source_open": True,       # whether the Source section is expanded
    "calibration_open": True,  # ... and the Calibration section
}


VIEWERS_MARKER = "@@viewers "     # keep in step with __main__.VIEWERS_MARKER
LIMIT_MARKER = "@@max_viewers "   # ... and __main__.LIMIT_MARKER


def parse_viewers(line: str) -> int | None:
    """The browser count from a '@@viewers N' line the server prints, else None."""
    if not line.startswith(VIEWERS_MARKER):
        return None
    try:
        return max(0, int(line[len(VIEWERS_MARKER):].strip()))
    except ValueError:
        return None


def _covers(outer: str, inner: str) -> bool:
    """Is `inner` the same folder as `outer`, or inside it? (A watch already searches
    subfolders, so listing both would only stack everything twice.)"""
    try:
        a, b = Path(outer).resolve(), Path(inner).resolve()
    except OSError:
        return False
    return a == b or a in b.parents


def add_folder(folders: list[str], new: str) -> tuple[list[str], str]:
    """The folder list with `new` added, and a note for the user when that changed
    more than the obvious: skipped because a listed folder already covers it, or
    replacing listed folders that lie inside it."""
    for f in folders:
        if _covers(f, new):
            return list(folders), f"Already covered by {f}."
    inside = [f for f in folders if _covers(new, f)]
    kept = [f for f in folders if f not in inside]
    note = (f"Replaced {len(inside)} folder{'s' if len(inside) != 1 else ''} "
            "that lie inside it." if inside else "")
    return kept + [new], note


def split_existing(folders: list[str]) -> tuple[list[str], list[str]]:
    """(folders that exist, folders that don't)."""
    ok = [f for f in folders if Path(f).is_dir()]
    return ok, [f for f in folders if f not in ok]


class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RGB Livestacker")
        root.minsize(640, 560)

        self._viewers_n: int | None = None
        self.proc: subprocess.Popen | None = None
        self.log_q: "queue.Queue[str]" = queue.Queue()
        # not in self.v: that is what gets saved to disk, and a password should
        # not be written there — it is asked for again at each launch
        self.password = tk.StringVar()
        self.v = {k: tk.StringVar(value=str(v)) if not isinstance(v, bool)
                  else tk.BooleanVar(value=v) for k, v in DEFAULTS.items()}
        self.folders: list[str] = []             # the capture folders (see _load)
        self._section_rows: dict[str, tuple[int, int]] = {}   # collapsible: key -> grid rows
        self._hidden: dict[str, list] = {}                    # key -> widgets while collapsed
        self._section_ui: dict[str, tuple] = {}               # key -> (title, arrow label, summary label)
        self._load()

        self._build()
        self.root.after(100, self._drain_log)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        if sys.platform == "darwin":
            # Cmd-Q (and Quit in the app menu) skip the window-close handler
            root.createcommand("tk::mac::Quit", self._on_close)
        atexit.register(self._stop)       # any other normal exit

    # -- layout --------------------------------------------------------

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)
        r = 0

        self._section_header(frm, r, "Source", "source"); r += 1
        src_first = r

        mode = ttk.Frame(frm)
        mode.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Radiobutton(mode, text="Watch a live folder", value="watch",
                        variable=self.v["mode"], command=self._sync).pack(side="left")
        ttk.Radiobutton(mode, text="Replay an old session", value="replay",
                        variable=self.v["mode"], command=self._sync).pack(side="left", padx=12)
        r += 1

        ttk.Label(frm, text="Folders").grid(row=r, column=0, sticky="nw", **pad)
        box = ttk.Frame(frm)
        box.grid(row=r, column=1, sticky="ew", **pad)
        box.columnconfigure(0, weight=1)
        self.folder_list = tk.Listbox(box, height=3, exportselection=False,
                                      activestyle="none")
        self.folder_list.grid(row=0, column=0, sticky="ew")
        fsb = ttk.Scrollbar(box, command=self.folder_list.yview)
        fsb.grid(row=0, column=1, sticky="ns")
        self.folder_list["yscrollcommand"] = fsb.set
        buttons = ttk.Frame(frm)
        buttons.grid(row=r, column=2, sticky="nw", **pad)
        ttk.Button(buttons, text="Add…", width=9, command=self._add_folder_dialog).pack()
        ttk.Button(buttons, text="Remove", width=9, command=self._remove_folder).pack(pady=2)
        r += 1
        ttk.Label(frm, text="Subs from every folder are stacked in the order they were "
                            "captured.", foreground="#888").grid(
            row=r, column=1, columnspan=2, sticky="w", padx=8); r += 1
        self._refresh_folders()

        ttk.Label(frm, text="Object name").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.v["object_name"]).grid(
            row=r, column=1, sticky="ew", **pad)
        self._filling = False
        self.v["object_name"].trace_add("write", self._object_edited)
        ttk.Label(frm, text="from the FITS header; edit to override",
                  foreground="#888").grid(
            row=r, column=2, sticky="w", **pad); r += 1

        self.interval_row = ttk.Frame(frm)
        self.interval_row.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Label(self.interval_row, text="Replay speed").pack(side="left")
        ttk.Spinbox(self.interval_row, from_=0.1, to=30, increment=0.1, width=6,
                    textvariable=self.v["interval"]).pack(side="left", padx=6)
        ttk.Label(self.interval_row, text="seconds per frame").pack(side="left")
        r += 1

        self.catchup_row = ttk.Frame(frm)
        self.catchup_row.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Checkbutton(self.catchup_row, variable=self.v["catch_up"],
                        text="Also stack subs already in the folder "
                             "(covers a late start / restart)").pack(side="left")
        r += 1
        self._section_rows["source"] = (src_first, r)

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=8); r += 1
        self._section_header(frm, r, "Calibration (optional)", "calibration"); r += 1
        cal_first = r

        cal_rows = (
            ("bias", "Master bias", "a folder of bias subs, or one master FITS — also subtracted from the dark and flats"),
            ("dark", "Master dark", "a folder of dark subs (any filters), or one master FITS"),
            ("flat", "Master flat", "a folder of flat subs for every filter — sorted by FILTER, one master per channel"),
        )
        for key, label, hint in cal_rows:
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(frm, textvariable=self.v[key]).grid(row=r, column=1, sticky="ew", **pad)
            btns = ttk.Frame(frm)
            btns.grid(row=r, column=2, **pad)
            ttk.Button(btns, text="Folder…", width=8,
                       command=lambda k=key: self._pick_dir(k)).pack(side="left")
            ttk.Button(btns, text="File…", width=6,
                       command=lambda k=key: self._pick_file(k)).pack(side="left", padx=2)
            ttk.Button(btns, text="✕", width=2,
                       command=lambda k=key: self.v[k].set("")).pack(side="left")
            r += 1
            ttk.Label(frm, text=hint, foreground="#888").grid(
                row=r, column=1, columnspan=2, sticky="w", padx=8); r += 1

        ttk.Label(frm, text="Crop edges").grid(row=r, column=0, sticky="w", **pad)
        crop = ttk.Frame(frm)
        crop.grid(row=r, column=1, columnspan=2, sticky="w", **pad); r += 1
        for key, lbl in (("crop_top", "top"), ("crop_bottom", "bottom"),
                         ("crop_left", "left"), ("crop_right", "right")):
            ttk.Label(crop, text=lbl).pack(side="left", padx=(0, 2))
            ttk.Spinbox(crop, from_=0, to=500, width=4,
                        textvariable=self.v[key]).pack(side="left", padx=(0, 10))
        ttk.Label(frm, text="px trimmed off every raw frame (a bad sensor edge)",
                  foreground="#888").grid(row=r, column=1, columnspan=2,
                                          sticky="w", padx=8); r += 1
        self._section_rows["calibration"] = (cal_first, r)

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=8); r += 1

        opts = ttk.Frame(frm)
        opts.grid(row=r, column=0, columnspan=3, sticky="w", **pad); r += 1
        for key, label in (("hot_pixel", "Hot-pixel fix"), ("sigma_clip", "Sigma-clip"),
                           ("background", "Background removal"), ("auto_wb", "Auto white balance"),
                           ("narrowband", "Narrowband")):
            ttk.Checkbutton(opts, text=label, variable=self.v[key]).pack(side="left", padx=(0, 12))

        osc = ttk.Frame(frm)
        osc.grid(row=r, column=0, columnspan=3, sticky="w", **pad); r += 1
        ttk.Label(osc, text="Colour camera (OSC) resolution").pack(side="left")
        ttk.Combobox(osc, textvariable=self.v["osc_debayer"], width=11, state="readonly",
                     values=("auto", "full", "half")).pack(side="left", padx=8)
        ttk.Label(osc, text="auto = full up to 12 MP; half is faster and uses ¼ the memory",
                  foreground="#888").pack(side="left")

        pw = ttk.Frame(frm)
        pw.grid(row=r, column=0, columnspan=3, sticky="ew", **pad); r += 1
        ttk.Label(pw, text="Password").pack(side="left")
        ttk.Entry(pw, textvariable=self.password, width=24).pack(
            side="left", padx=8)
        ttk.Label(pw, text="optional: browsers must enter it. Not saved.",
                  foreground="#888").pack(side="left")

        net = ttk.Frame(frm)
        net.grid(row=r, column=0, columnspan=3, sticky="w", **pad); r += 1
        ttk.Checkbutton(net, text="Open browser on start",
                        variable=self.v["open_browser"]).pack(side="left")
        ttk.Checkbutton(net, text="Share on local network",
                        variable=self.v["share_lan"]).pack(side="left", padx=16)
        ttk.Label(net, text="Max browsers").pack(side="left")
        ttk.Spinbox(net, from_=1, to=500, width=5,
                    textvariable=self.v["max_viewers"]).pack(side="left", padx=6)
        ttk.Label(net, text="(changes apply at once)", foreground="#888").pack(side="left")
        self._limit_job = None
        self.v["max_viewers"].trace_add("write", self._limit_edited)

        # -- run controls ------------------------------------------
        run = ttk.Frame(frm)
        run.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 4)); r += 1
        self.start_btn = ttk.Button(run, text="Start", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(run, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.status = ttk.Label(run, text="idle", foreground="#888")
        self.status.pack(side="left", padx=12)
        self.viewers = ttk.Label(run, text="")          # live browser count
        self.viewers.pack(side="right", padx=4)

        frm.rowconfigure(r, weight=1)
        self.log = tk.Text(frm, height=12, wrap="none", state="disabled",
                           background="#111418", foreground="#cdd3dd",
                           insertbackground="#cdd3dd")
        self.log.grid(row=r, column=0, columnspan=3, sticky="nsew", padx=8, pady=(4, 0))
        sb = ttk.Scrollbar(frm, command=self.log.yview)
        sb.grid(row=r, column=3, sticky="ns")
        self.log["yscrollcommand"] = sb.set

        self._sync()
        # open wide enough that the path boxes have room (they stretch with the window),
        # and tall enough for everything *expanded*: a section left collapsed from last
        # time then gives its space to the log, rather than shrinking the window
        self.root.update_idletasks()
        self.root.geometry(f"{max(self.root.winfo_reqwidth(), 960)}x"
                           f"{self.root.winfo_reqheight()}")
        for key in self._section_rows:
            if not self.v[f"{key}_open"].get():
                self._set_open(key, False)

    # -- collapsible sections -----------------------------------------

    def _section_header(self, frm: ttk.Frame, row: int, title: str, key: str) -> None:
        """A clickable heading: '▾ Source' (open) / '▸ Source  summary' (collapsed)."""
        hdr = ttk.Frame(frm)
        hdr.grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 4))
        label = ttk.Label(hdr, font=("", 12, "bold"), cursor="hand2")
        label.pack(side="left")
        summary = ttk.Label(hdr, foreground="#888", cursor="hand2")
        summary.pack(side="left", padx=10)
        self._section_ui[key] = (title, label, summary)
        for w in (hdr, label, summary):
            w.bind("<Button-1>", lambda _e, k=key: self._toggle_section(k))
        label.configure(text=f"▾ {title}")

    def _toggle_section(self, key: str) -> None:
        self._set_open(key, not self.v[f"{key}_open"].get())

    def _set_open(self, key: str, opened: bool) -> None:
        """Show or hide a section's rows. The window keeps its size, so the log row
        (the only one that stretches) takes up the freed space, and gives it back."""
        self.v[f"{key}_open"].set(opened)
        if opened:
            for w in self._hidden.pop(key, []):          # (a hidden widget is no longer
                w.grid()                                  #  in the grid's list: keep our own)
        elif key not in self._hidden:
            first, last = self._section_rows[key]
            frm = self._section_ui[key][1].master.master
            hide = [w for w in frm.grid_slaves() if first <= int(w.grid_info()["row"]) < last]
            for w in hide:
                w.grid_remove()
            self._hidden[key] = hide
        title, label, summary = self._section_ui[key]
        label.configure(text=f"{'▾' if opened else '▸'} {title}")
        summary.configure(text="" if opened else self._summary(key))

    def _summary(self, key: str) -> str:
        """What a collapsed section would have shown, in a few words."""
        if key == "source":
            n = len(self.folders)
            what = "replaying" if self.v["mode"].get() == "replay" else "watching"
            text = f"{what} {n} folder{'s' if n != 1 else ''}" if n else "no folder chosen"
            name = self.v["object_name"].get().strip()
            return f"{text}  ·  {name}" if name else text
        parts = [label for k, label in (("bias", "bias"), ("dark", "dark"), ("flat", "flat"))
                 if self.v[k].get().strip()]
        def trimmed(edge: str) -> bool:
            try:
                return float(self.v[f"crop_{edge}"].get() or 0) > 0
            except ValueError:
                return False
        if any(trimmed(e) for e in ("top", "bottom", "left", "right")):
            parts.append("edge crop")
        return "  ·  ".join(parts) if parts else "none"

    # -- helpers ------------------------------------------------------

    def _sync(self) -> None:
        replay = self.v["mode"].get() == "replay"
        for child in self.interval_row.winfo_children():
            child.configure(state="normal" if replay else "disabled")
        for child in self.catchup_row.winfo_children():
            child.configure(state="disabled" if replay else "normal")

    def _pick_dir(self, key: str) -> None:
        d = filedialog.askdirectory(title="Choose folder",
                                    initialdir=self.v[key].get() or None)
        if d:
            self.v[key].set(d)

    def _refresh_folders(self) -> None:
        self.folder_list.delete(0, "end")
        for f in self.folders:
            self.folder_list.insert("end", f)

    def _add_folder_dialog(self) -> None:
        d = filedialog.askdirectory(
            title="Add a capture folder",
            initialdir=(self.folders[-1] if self.folders else None))
        if not d:
            return
        self.folders, note = add_folder(self.folders, d)
        self._refresh_folders()
        if note:
            messagebox.showinfo("Livestacker", note)
        self._autofill_object()

    def _remove_folder(self) -> None:
        for i in sorted(self.folder_list.curselection(), reverse=True):
            del self.folders[i]
        self._refresh_folders()

    def _object_edited(self, *_):
        if not self._filling:                  # the user typed: keep their text
            self.v["object_auto"].set(False)

    def _autofill_object(self) -> None:
        """Fill the object name from the folder's FITS headers — unless the user
        has typed their own."""
        if self.v["object_name"].get().strip() and not self.v["object_auto"].get():
            return
        from .fitsframe import guess_object     # (pulls in astropy: only when needed)
        guess = ""
        for folder in split_existing(self.folders)[0]:       # the first folder that has one
            try:
                guess = guess_object(folder)
            except Exception:
                continue
            if guess:
                break
        if guess:
            self._filling = True
            try:
                self.v["object_name"].set(guess)
                self.v["object_auto"].set(True)
            finally:
                self._filling = False

    def _pick_file(self, key: str) -> None:
        f = filedialog.askopenfilename(title="Choose master FITS",
                                       filetypes=[("FITS", "*.fits *.fit *.fts"),
                                                  ("All files", "*.*")])
        if f:
            self.v[key].set(f)

    def _append(self, text: str) -> None:
        self.log["state"] = "normal"
        self.log.insert("end", text)
        self.log.see("end")
        self.log["state"] = "disabled"

    def _drain_log(self) -> None:
        try:
            while True:
                line = self.log_q.get_nowait()
                n = parse_viewers(line)
                if n is None:
                    self._append(line)
                else:
                    self._show_viewers(n)
        except queue.Empty:
            pass
        if self.proc and self.proc.poll() is not None:
            self._finished(self.proc.returncode)
        self.root.after(100, self._drain_log)

    def _limit(self) -> int | None:
        """The 'Max browsers' box as a number, or None while it holds nonsense."""
        try:
            n = int(float(self.v["max_viewers"].get()))
        except (ValueError, tk.TclError):
            return None
        return n if 1 <= n <= 500 else None

    def _limit_edited(self, *_) -> None:
        # (the box fires on every keystroke: wait until typing pauses)
        if self._limit_job is not None:
            self.root.after_cancel(self._limit_job)
        self._limit_job = self.root.after(400, self._push_limit)
        self._show_viewers(self._viewers_n)

    def _push_limit(self) -> None:
        """Tell the running server the new limit; it takes effect straight away."""
        self._limit_job = None
        n = self._limit()
        if n is None or not self.proc or self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write(f"{LIMIT_MARKER}{n}\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def _show_viewers(self, n: int | None) -> None:
        """n browsers connected right now; None = the stacker isn't running."""
        self._viewers_n = n
        limit = self._limit()
        of = f" of {limit}" if limit else ""
        if n is None:
            self.viewers.configure(text="")
        elif n == 0:
            self.viewers.configure(text=f"0{of} browsers connected", foreground="#888")
        else:
            full = limit is not None and n >= limit
            self.viewers.configure(
                text=f"● {n}{of} browser{'s' if (n != 1 or of) else ''} connected"
                     + (" (full)" if full else ""),
                foreground="#c80" if full else "#3a8")

    def _same_target(self, folders: list[str]) -> bool:
        """Everything is aligned to the first frame, so a folder of a different target
        would only be rejected: say so before starting. True = go ahead."""
        from .fitsframe import guess_object
        names = {}
        for f in folders:
            try:
                name = guess_object(f)
            except Exception:
                name = ""
            if name:
                names[f] = name
        if len(set(names.values())) < 2:
            return True
        listing = "\n".join(f"{n}  —  {f}" for f, n in names.items())
        return messagebox.askyesno(
            "Livestacker", "These folders look like different targets:\n\n" + listing
            + "\n\nEverything is aligned to the very first frame, so the other "
              "target's subs would be rejected. Start anyway?")

    # -- run / stop -------------------------------------------------

    def _command(self) -> list[str] | None:
        folders, missing = split_existing(self.folders)
        if not folders:
            messagebox.showerror("Livestacker", "Add at least one folder that exists.")
            return None
        if missing and not messagebox.askokcancel(
                "Livestacker", "These folders can't be found and will be skipped:\n\n"
                + "\n".join(missing) + "\n\nStart with the others?"):
            return None
        if len(folders) > 1 and not self._same_target(folders):
            return None
        cmd = [sys.executable, "-m", "livestacker", "--exit-with-launcher", "-v"]
        for folder in folders:
            cmd += [f"--{self.v['mode'].get()}", folder]
        if self.v["mode"].get() == "replay":
            cmd += ["--interval", self.v["interval"].get()]
        elif not self.v["catch_up"].get():
            cmd.append("--no-catch-up")
        osc = {"full": "bilinear", "half": "superpixel"}.get(self.v["osc_debayer"].get(), "auto")
        cmd += ["--osc-debayer", osc]
        limit = self._limit()
        if limit is None:
            messagebox.showerror("Livestacker", "Max browsers must be a number from 1 to 500.")
            return None
        cmd += ["--max-viewers", str(limit)]
        name = self.v["object_name"].get().strip()
        if name:
            cmd += ["--object-name", name]
        for key, flag in (("bias", "--bias"), ("dark", "--dark"), ("flat", "--flat")):
            val = self.v[key].get().strip()
            if val:
                cmd += [flag, val]
        if not self.v["hot_pixel"].get():
            cmd.append("--no-hot-pixel")
        if not self.v["sigma_clip"].get():
            cmd.append("--no-sigma-clip")
        else:
            cmd += ["--sigma-clip-kappa", self.v["sigma_clip_kappa"].get()]
        if not self.v["background"].get():
            cmd.append("--no-background")
        if not self.v["auto_wb"].get():
            cmd.append("--no-auto-wb")
        if not self.v["narrowband"].get():
            cmd.append("--no-narrowband")
        cmd += ["--preview-px", self.v["preview_px"].get()]
        for edge in ("top", "bottom", "left", "right"):
            val = str(self.v[f"crop_{edge}"].get()).strip()
            if val and val not in ("0", "0.0"):
                cmd += [f"--crop-{edge}", val]
        cmd += ["--host", "0.0.0.0" if self.v["share_lan"].get() else "127.0.0.1"]
        cmd += ["--web-port", self.v["web_port"].get(),
                "--ws-port", self.v["ws_port"].get()]
        return cmd

    def _start(self) -> None:
        cmd = self._command()
        if not cmd:
            return
        self._save()
        self.log["state"] = "normal"
        self.log.delete("1.0", "end")
        self.log["state"] = "disabled"
        self._append("$ " + " ".join(cmd) + "\n\n")

        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        # handed over in the environment, so it never appears in the command line
        env = os.environ.copy()
        env.pop("LIVESTACK_PASSWORD", None)
        if self.password.get():
            env["LIVESTACK_PASSWORD"] = self.password.get()
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,      # held open: its closing tells the
                # server we are gone (see --exit-with-launcher)
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=flags, env=env,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
        except OSError as exc:
            messagebox.showerror("Livestacker", f"Could not start:\n{exc}")
            return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

        self.start_btn["state"] = "disabled"
        self.stop_btn["state"] = "normal"
        self.status.configure(text="running", foreground="#3a8")
        self._show_viewers(0)
        if self.v["open_browser"].get():
            host = "localhost"
            url = f"http://{host}:{self.v['web_port'].get()}/"
            self.root.after(1500, lambda: webbrowser.open(url))

    def _reader(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log_q.put(line)

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _finished(self, code: int) -> None:
        self.proc = None
        self._show_viewers(None)
        self.start_btn["state"] = "normal"
        self.stop_btn["state"] = "disabled"
        ok = code == 0
        self.status.configure(text="stopped" if ok else f"exited ({code})",
                              foreground="#888" if ok else "#c55")

    def _on_close(self) -> None:
        self._stop()
        self._save()
        self.root.destroy()

    # -- settings --------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(SETTINGS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for k, var in self.v.items():
            if k in data:
                var.set(data[k])
        folders = data.get("folders")
        if not isinstance(folders, list):                    # settings from before there were several
            folders = [data["folder"]] if data.get("folder") else []
        self.folders = [f for f in folders if isinstance(f, str) and f.strip()]

    def _save(self) -> None:
        data = {}
        for k, var in self.v.items():
            val = var.get()
            if isinstance(DEFAULTS[k], bool):
                data[k] = bool(val)
            elif isinstance(DEFAULTS[k], int):
                try:
                    data[k] = int(float(val))
                except ValueError:
                    data[k] = DEFAULTS[k]
            elif isinstance(DEFAULTS[k], float):
                try:
                    data[k] = float(val)
                except ValueError:
                    data[k] = DEFAULTS[k]
            else:
                data[k] = val
        data["folders"] = list(self.folders)
        data["folder"] = self.folders[0] if self.folders else ""     # (for an older launcher)
        try:
            SETTINGS.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass


def run() -> int:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"error: could not open a window ({exc}).", file=sys.stderr)
        print("Run with --watch / --replay for the headless mode, or install Tk:",
              file=sys.stderr)
        print("  macOS (Homebrew): brew install python-tk", file=sys.stderr)
        print("  Debian/Ubuntu:    sudo apt install python3-tk", file=sys.stderr)
        return 1
    Launcher(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
