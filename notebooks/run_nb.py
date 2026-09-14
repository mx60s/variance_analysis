#!/usr/bin/env python
"""
Execute a project notebook locally (no Colab, no Helvetica, headless plotting).

Usage:
  .venv/bin/python notebooks/run_nb.py notebooks/bio_neurons.ipynb [--only 3,4,7-11] [--skip 13] [--stop-on-error]

Cells are executed sequentially in one namespace, like "Run all". Colab-only cells
(drive mount, pip installs) are skipped, shell/magic lines are stripped,
`Path().resolve()` is pinned to the project root, plt.show()/plotly show() are no-ops,
and plotly write_image failures (kaleido/chrome) are logged rather than fatal.
"""
import argparse, json, os, sys, time, traceback, re

ROOT = "/Users/moo/variance_analysis"

def parse_ranges(s, n):
    if not s:
        return None
    out = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out

def prepare(src):
    if "%pip" in src or "eval_js" in src:
        return None
    drop = ("google.colab", "drive.mount", "os.chdir(", "subprocess", "kaleido", "importlib", "PackageNotFoundError")
    lines = []
    for ln in src.splitlines():
        s = ln.lstrip()
        if s.startswith("!") or s.startswith("%") or any(d in ln for d in drop):
            continue
        lines.append(ln)
    src = "\n".join(lines)
    src = src.replace("Path().resolve()", f"Path('{ROOT}')")
    src = src.replace("'/content/drive/MyDrive/ve/variance_analysis'", f"'{ROOT}'")
    src = src.replace("plt.rcParams['font.family'] = 'Helvetica'", "plt.rcParams['font.family'] = 'DejaVu Sans'")
    src = re.sub(r"device\s*=\s*'cuda'", "device = 'mps'", src)
    src = src.replace("fm._load_fontmanager(try_read_cache=False)", "pass")
    return src

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None
    try:
        import plotly.graph_objects as go
        go.Figure.show = lambda self, *a, **k: None
        _wi = go.Figure.write_image
        def safe_write_image(self, *a, **k):
            try:
                return _wi(self, *a, **k)
            except Exception as e:  # kaleido / chrome missing
                print(f"[run_nb] write_image skipped: {type(e).__name__}: {str(e)[:120]}")
        go.Figure.write_image = safe_write_image
    except Exception:
        pass

    nb = json.load(open(args.notebook))
    cells = nb["cells"]
    only = parse_ranges(args.only, len(cells))
    skip = parse_ranges(args.skip, len(cells)) or set()
    g = {"__name__": "__main__", "os": os}
    n_err = 0
    t_all = time.time()
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if only is not None and i not in only:
            continue
        if i in skip:
            print(f"[run_nb] cell {i}: skipped (--skip)")
            continue
        src = prepare("".join(c["source"]))
        if src is None or not src.strip():
            print(f"[run_nb] cell {i}: skipped (colab/empty)")
            continue
        print(f"\n[run_nb] ===== cell {i} =====", flush=True)
        t0 = time.time()
        try:
            exec(compile(src, f"<cell {i}>", "exec"), g)
            print(f"[run_nb] cell {i} ok ({time.time()-t0:.1f}s)", flush=True)
        except Exception:
            n_err += 1
            print(f"[run_nb] cell {i} ERROR ({time.time()-t0:.1f}s):", flush=True)
            traceback.print_exc()
            if args.stop_on_error:
                break
    print(f"\n[run_nb] done: {n_err} cell errors, {time.time()-t_all:.0f}s total")

if __name__ == "__main__":
    main()
