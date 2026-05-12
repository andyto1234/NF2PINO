"""Small wrappers around RTMAG/PINO inference."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from .bridge import load_pino_field
from .paths import ensure_repo_paths


def load_vector_magnetogram(
    *,
    magnetogram: str | Path | None = None,
    bp: str | Path | None = None,
    bt: str | Path | None = None,
    br: str | Path | None = None,
) -> np.ndarray:
    """Load a vector magnetogram as `[nx, ny, 3]` with `Bx, By, Bz`.

    `.npy` files may contain the array directly. `.npz` files may use either
    `hmi_data` or `b`; if `b` is 3D, the bottom slice is used. FITS input is
    supported through separate `bp`, `bt`, and `br` component paths.
    """
    if bp and bt and br:
        from astropy.io import fits

        p_data = fits.getdata(bp).astype(np.float32)
        t_data = fits.getdata(bt).astype(np.float32)
        r_data = fits.getdata(br).astype(np.float32)
        return np.stack([p_data, -t_data, r_data], axis=-1).astype(np.float32)

    if magnetogram is None:
        raise ValueError(
            "Provide a magnetogram path, bp/bt/br FITS paths, or pass hmi_data from "
            "get_sharp_map() (see rtmag/main.ipynb)."
        )
    magnetogram = Path(magnetogram)

    if magnetogram.suffix == ".npy":
        return np.load(magnetogram).astype(np.float32)

    if magnetogram.suffix == ".npz":
        with np.load(magnetogram, allow_pickle=True) as data:
            if "hmi_data" in data:
                return np.asarray(data["hmi_data"], dtype=np.float32)
            if "b" in data:
                b = np.asarray(data["b"], dtype=np.float32)
                return b[:, :, 0, :] if b.ndim == 4 else b
        raise KeyError(f"{magnetogram} must contain `hmi_data` or `b`")

    raise ValueError("Single-file FITS vector magnetograms are ambiguous; pass --bp --bt --br instead")


def resolve_rtmag_latest_extrapolation(rtmag_root: str | Path) -> Path | None:
    """Return `rtmag_extrapolation.npz` from the case pointed to by main.ipynb.

    main.ipynb writes ``examples/analysis_exports/latest_case.txt`` with the case
    directory; the field lives under ``extrapolation/rtmag_extrapolation.npz``.
    Falls back to the legacy inclination export if the pointer is missing.
    """
    rtmag_root = Path(rtmag_root)
    latest = rtmag_root / "examples" / "analysis_exports" / "latest_case.txt"
    if latest.is_file():
        case_dir = Path(latest.read_text().strip())
        candidate = case_dir / "extrapolation" / "rtmag_extrapolation.npz"
        if candidate.is_file():
            return candidate
    legacy = rtmag_root / "examples" / "extrapolations" / "rtmag_extrapolation_for_inclination.npz"
    if legacy.is_file():
        return legacy
    return None


def run_pino_extrapolation(
    *,
    model_path: str | Path,
    output_path: str | Path,
    magnetogram: str | Path | None = None,
    hmi_data: np.ndarray | None = None,
    bp: str | Path | None = None,
    bt: str | Path | None = None,
    br: str | Path | None = None,
    nx: int = 512,
    ny: int = 256,
    device: str | torch.device | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run RTMAG/PINO inference and save an NF2 bridge-ready `.npz` export."""
    ensure_repo_paths()
    from rtmag.process.paper.hmi_to_input import get_input
    from rtmag.process.paper.load import MyModel

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        field = load_pino_field(output_path)
        return {"path": str(output_path), "cached": True, "shape": field.shape}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hmi_data is not None:
        hmi_arr = np.asarray(hmi_data, dtype=np.float32)
    else:
        if magnetogram is None and not (bp and bt and br):
            raise ValueError(
                "Provide hmi_data from get_sharp_map(), a magnetogram file path, or bp/bt/br FITS paths."
            )
        hmi_arr = load_vector_magnetogram(magnetogram=magnetogram, bp=bp, bt=bt, br=br)
    model_input, x, y, z, dx, dy, dz = get_input(hmi_arr, nx=nx, ny=ny)
    device = _resolve_device(device)
    start = perf_counter()
    model = MyModel(model_path, device=device)
    b = model.get_pred_from_numpy(model_input)
    runtime = perf_counter() - start

    np.savez_compressed(
        output_path,
        b=b.astype(np.float32),
        x_coords=np.asarray(x, dtype=np.float32),
        y_coords=np.asarray(y, dtype=np.float32),
        z_coords=np.asarray(z, dtype=np.float32),
        dx_Mm=np.float32(dx),
        dy_Mm=np.float32(dy),
        dz_Mm=np.float32(dz),
        component_order=np.asarray(["Bx", "By", "Bz"]),
        coordinate_units=np.asarray("Mm"),
        field_units=np.asarray("G"),
    )
    return {"path": str(output_path), "cached": False, "runtime_seconds": runtime, "shape": b.shape}


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
