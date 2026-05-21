import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PILOT = ROOT / "workdirs" / "PILOT"

for path in (SRC, PILOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
