import os
from pathlib import Path


MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
OUTDIR = Path(os.getenv("OUTDIR") or "/tmp/sleepspec")
OUTDIR.mkdir(exist_ok=True)
