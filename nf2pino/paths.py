"""Path helpers for using the sibling NF2 and RTMAG repositories in-place."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NF2_ROOT = REPO_ROOT / "NF2"
RTMAG_ROOT = REPO_ROOT / "rtmag"


def ensure_repo_paths() -> None:
    """Make the in-tree `nf2` and `rtmag` packages importable."""
    for path in (NF2_ROOT, RTMAG_ROOT, REPO_ROOT):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def configure_runtime_env(base_dir: str | Path) -> dict[str, str]:
    """Set writable runtime directories for optional desktop/science deps.

    SunPy and Matplotlib both want writable user config/cache directories. When
    running inside sandboxed or ephemeral environments, those defaults may not
    exist or may not be writable. This helper only sets env vars that are
    currently unset and keeps all runtime scratch data under ``base_dir``.
    """
    base_dir = Path(base_dir)
    runtime_dir = base_dir / "runtime_env"
    sunpy_dir = runtime_dir / "sunpy"
    mpl_dir = runtime_dir / "matplotlib"
    sunpy_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)

    resolved = {}
    if not os.environ.get("SUNPY_CONFIGDIR"):
        os.environ["SUNPY_CONFIGDIR"] = str(sunpy_dir)
    resolved["SUNPY_CONFIGDIR"] = os.environ["SUNPY_CONFIGDIR"]

    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    resolved["MPLCONFIGDIR"] = os.environ["MPLCONFIGDIR"]
    return resolved
