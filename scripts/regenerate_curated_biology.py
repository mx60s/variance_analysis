"""Consume arriving curated subsets, rerun exactly the 30 legacy sessions."""

import os, time, argparse, json
from pathlib import Path
from regenerate_biology import run, sessions


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--root", type=Path, required=True)
    args = a.parse_args()
    pending = [(d, r) for kind, d, r in sessions() if kind == "dusz"]
    root = args.root
    while pending:
        ready = [
            (d, r)
            for d, r in pending
            if (root / "curated_inputs" / f"{r}.json").exists()
        ]
        for d, r in ready:
            run(
                "dusz",
                d,
                r,
                root / "curated_aligned",
                curated_dir=root / "curated_inputs",
            )
            run(
                "dusz",
                d,
                r,
                root / "curated_native_coarse",
                no_decoding=True,
                native=True,
                hdmd_bins=36,
                curated_dir=root / "curated_inputs",
            )
            pending.remove((d, r))
            print("CURATED remaining", len(pending), flush=True)
        if pending:
            time.sleep(10)
    print("CURATED COMPLETE", flush=True)


if __name__ == "__main__":
    main()
