"""Save completed inference scores from a live container before its timeout."""

import argparse
import base64
import io
import subprocess
import zipfile
from pathlib import Path

REMOTE = """
from pathlib import Path
import zipfile,io,base64,time
root=Path('/project/results/review_20261116/revalidation/models')
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 for meta in root.glob('*/summary.json'):
  d=meta.parent
  if not (d/'ev_native.npz').exists():continue
  for p in [meta,d/'ev.npz',d/'ev_native.npz']:z.write(p,str(p.relative_to(root)))
encoded=base64.b64encode(buf.getvalue()).decode()
for i in range(0,len(encoded),4096):
 print(encoded[i:i+4096],flush=True);time.sleep(.01)
time.sleep(1)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modal", required=True)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = subprocess.run(
        [
            args.modal,
            "container",
            "exec",
            "--no-pty",
            args.container,
            "--",
            "python",
            "-c",
            REMOTE,
        ],
        capture_output=True,
        check=False,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    data = base64.b64decode(b"".join(result.stdout.split()), validate=True)
    print("Received archive bytes", len(data), "header", data[:4].hex())
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        print("Backed up", len(z.namelist()) // 3, "completed model/layers")
        if z.testzip():
            raise ValueError("Archive integrity failed")
    args.output.write_bytes(data)
