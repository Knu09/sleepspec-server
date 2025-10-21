import os
from pathlib import Path


OUTDIR = Path(os.getenv("OUTDIR") or "/tmp/sleepspec")
OUTDIR.mkdir(exist_ok=True)
