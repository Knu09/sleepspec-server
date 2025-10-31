import os
from pathlib import Path


MAX_WORKERS = os.getenv("MAX_WORKERS") or 2
OUTDIR = Path(os.getenv("OUTDIR") or "/tmp/sleepspec")
OUTDIR.mkdir(exist_ok=True)
