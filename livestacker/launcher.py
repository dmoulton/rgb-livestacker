"""A small cross-platform launcher window (Tkinter, no extra dependencies).

Pick the capture folder and options, press Start, and it runs
`python -m livestacker …` as a child process with its log shown in the window.
Choices are remembered between runs in `launcher-settings.json` next to the repo.

Opened by running the module with no arguments (`python -m livestacker`).
"""

from __future__ import annotations

import json
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
    "folder": "",
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
    "preview_px": 1600,
    "share_lan": False,
    "web_port": 8080,
    "ws_port": 8081,
    "open_browser": True,
}


class Launcher:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("RGB Livestacker")
        root.minsize(640, 560)

        self.proc: subprocess.Popen | None = None
        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.v = {k: tk.StringVar(value=str(v)) if not isinstance(v, bool)
                  else tk.BooleanVar(value=v) for k, v in DEFAULTS.items()}
        self._load()

        self._build()
        self.root.after(100, self._drain_log)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout --------------------------------------------------------

    def _build(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)
        r = 0

        ttk.Label(frm, text="Source", font=("", 12, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(0, 4)); r += 1

        mode = ttk.Frame(frm)
        mode.grid(row=r, column=0, columnspan=3, sticky="w", **pad)
        ttk.Radiobutton(mode, text="Watch a live folder", value="watch",
                        variable=self.v["mode"], command=self._sync).pack(side="left")
        ttk.Radiobutton(mode, text="Replay an old session", value="replay",
                        variable=self.v["mode"], command=self._sync).pack(side="left", padx=12)
        r += 1

        ttk.Label(frm, text="Folder").grid(row=r, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.v["folder"]).grid(row=r, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse…", command=lambda: self._pick_dir("folder")).grid(
            row=r, column=2, **pad); r += 1

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

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=8); r += 1
        ttk.Label(frm, text="Calibration (optional)", font=("", 12, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(0, 4)); r += 1

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

        ttk.Separator(frm, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=8); r += 1

        opts = ttk.Frame(frm)
        opts.grid(row=r, column=0, columnspan=3, sticky="w", **pad); r += 1
        for key, label in (("hot_pixel", "Hot-pixel fix"), ("sigma_clip", "Sigma-clip"),
                           ("background", "Background removal"), ("auto_wb", "Auto white balance")):
            ttk.Checkbutton(opts, text=label, variable=self.v[key]).pack(side="left", padx=(0, 12))

        net = ttk.Frame(frm)
        net.grid(row=r, column=0, columnspan=3, sticky="w", **pad); r += 1
        ttk.Checkbutton(net, text="Open browser on start",
                        variable=self.v["open_browser"]).pack(side="left")
        ttk.Checkbutton(net, text="Share on local network",
                        variable=self.v["share_lan"]).pack(side="left", padx=16)

        # -- run controls ------------------------------------------
        run = ttk.Frame(frm)
        run.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 4)); r += 1
        self.start_btn = ttk.Button(run, text="Start", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(run, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        self.status = ttk.Label(run, text="idle", foreground="#888")
        self.status.pack(side="left", padx=12)

        frm.rowconfigure(r, weight=1)
        self.log = tk.Text(frm, height=12, wrap="none", state="disabled",
                           background="#111418", foreground="#cdd3dd",
                           insertbackground="#cdd3dd")
        self.log.grid(row=r, column=0, columnspan=3, sticky="nsew", padx=8, pady=(4, 0))
        sb = ttk.Scrollbar(frm, command=self.log.yview)
        sb.grid(row=r, column=3, sticky="ns")
        self.log["yscrollcommand"] = sb.set

        self._sync()

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
                self._append(self.log_q.get_nowait())
        except queue.Empty:
            pass
        if self.proc and self.proc.poll() is not None:
            self._finished(self.proc.returncode)
        self.root.after(100, self._drain_log)

    # -- run / stop -------------------------------------------------

    def _command(self) -> list[str] | None:
        folder = self.v["folder"].get().strip()
        if not folder or not Path(folder).is_dir():
            messagebox.showerror("Livestacker", "Pick a folder that exists.")
            return None
        cmd = [sys.executable, "-m", "livestacker", "-v"]
        cmd += [f"--{self.v['mode'].get()}", folder]
        if self.v["mode"].get() == "replay":
            cmd += ["--interval", self.v["interval"].get()]
        elif not self.v["catch_up"].get():
            cmd.append("--no-catch-up")
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
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=flags,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
        except OSError as exc:
            messagebox.showerror("Livestacker", f"Could not start:\n{exc}")
            return
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

        self.start_btn["state"] = "disabled"
        self.stop_btn["state"] = "normal"
        self.status.configure(text="running", foreground="#3a8")
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
