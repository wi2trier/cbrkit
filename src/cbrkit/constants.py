from os import getenv
from pathlib import Path

CACHE_DIR = Path(getenv("CBRKIT_CACHE_DIR", Path.home() / ".cache" / "cbrkit"))
