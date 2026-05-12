import os
import tempfile
from pathlib import Path


_RUNTIME_ROOT = Path(tempfile.gettempdir()) / "nf2pino-test-runtime"
_SUNPY_DIR = _RUNTIME_ROOT / "sunpy"
_MPL_DIR = _RUNTIME_ROOT / "matplotlib"
_SUNPY_DIR.mkdir(parents=True, exist_ok=True)
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("SUNPY_CONFIGDIR", str(_SUNPY_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
