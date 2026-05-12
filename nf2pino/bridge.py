"""Bridge utilities between RTMAG/PINO cubes and NF2 initialization samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EXPECTED_COMPONENTS = ("Bx", "By", "Bz")


def _linear_map_to_range(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Affine map ``values`` from its min–max to ``[lo, hi]`` (matches :func:`nf2pino.compare.evaluate_nf2_on_pino_axes`)."""
    arr = np.asarray(values, dtype=np.float64)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.full_like(arr, 0.5 * (lo + hi), dtype=np.float64)
    s = (arr - mn) / (mx - mn)
    return lo + s * (hi - lo)


@dataclass(frozen=True)
class PinoField:
    """A PINO magnetic-field cube on a Cartesian model grid."""

    b: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    z_coords: np.ndarray
    field_units: str = "G"
    coordinate_units: str = "Mm"
    source_path: str | None = None

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.b.shape

    @property
    def spacing_Mm(self) -> tuple[float, float, float]:
        return tuple(float(_spacing(a)) for a in (self.x_coords, self.y_coords, self.z_coords))


def load_pino_field(path: str | Path, field_key: str = "b") -> PinoField:
    """Load and validate a PINO field export.

    RTMAG's `main.ipynb` writes `rtmag_extrapolation.npz` with key `b` and
    shape `[nx, ny, nz, 3]` in Gauss. This loader also accepts plain `.npy`
    cubes for tests and lightweight experiments.
    """
    path = Path(path)
    if path.suffix == ".npy":
        b = np.load(path).astype(np.float32)
        x, y, z = _default_coords_for_field(b)
        return _validate_pino_field(PinoField(b, x, y, z, source_path=str(path)))

    with np.load(path, allow_pickle=True) as data:
        if field_key not in data:
            available = ", ".join(sorted(data.files))
            raise KeyError(f"PINO field key '{field_key}' not found in {path}. Available keys: {available}")
        b = np.asarray(data[field_key], dtype=np.float32)

        component_order = tuple(str(c) for c in np.asarray(data.get("component_order", EXPECTED_COMPONENTS)).tolist())
        if component_order != EXPECTED_COMPONENTS:
            raise ValueError(f"Expected component order {EXPECTED_COMPONENTS}, got {component_order}")

        x, y, z = _coords_from_export(data, b)
        field_units = _scalar_str(data.get("field_units", "G"))
        coordinate_units = _scalar_str(data.get("coordinate_units", "Mm"))

    return _validate_pino_field(
        PinoField(
            b=b,
            x_coords=x,
            y_coords=y,
            z_coords=z,
            field_units=field_units,
            coordinate_units=coordinate_units,
            source_path=str(path),
        )
    )


def prepare_nfourier_init_samples(
    field: PinoField | str | Path,
    *,
    G_per_dB: float = 2500.0,
    Mm_per_ds: float = 0.36 * 320,
    max_samples: int | None = 200_000,
    stride: int | Iterable[int] | None = None,
    target_shape: tuple[int, int, int] | None = None,
    z_range_Mm: tuple[float, float] | None = None,
    seed: int = 0,
    align_domain: bool = True,
    nf2_xy_range_ds: np.ndarray | None = None,
    nf2_z_top_mm: float | None = None,
) -> dict[str, np.ndarray]:
    """Convert a PINO cube into NF2-normalized coordinate/field samples.

    NF2 Cartesian loaders scale coordinates by `Mm_per_ds` and fields by
    `G_per_dB`. The returned arrays are ready for direct supervised fitting of
    the existing NF2 neural field.

    When ``nf2_xy_range_ds`` (shape ``(2, 2)``, x/y bounds in **ds**) and
    ``nf2_z_top_mm`` are passed and ``align_domain`` is true, each PINO axis in
    Mm is affinely mapped to the NF2 FITS horizontal box and ``[0, z_top]`` **before**
    dividing by ``Mm_per_ds``. This matches evaluation via
    :func:`nf2pino.compare.evaluate_nf2_on_pino_axes` and avoids fitting PINO
    samples at the wrong ds coordinates (which yields kG fields collapsing to near‑zero **B** inside the NF2 volume).
    """
    field = load_pino_field(field) if isinstance(field, (str, Path)) else field
    field = _validate_pino_field(field)

    b, x, y, z = field.b, field.x_coords, field.y_coords, field.z_coords
    if z_range_Mm is not None:
        z_mask = (z >= z_range_Mm[0]) & (z <= z_range_Mm[1])
        if not np.any(z_mask):
            raise ValueError(f"No PINO z slices fall inside z_range_Mm={z_range_Mm}")
        b = b[:, :, z_mask]
        z = z[z_mask]

    if stride is not None:
        sx, sy, sz = _stride_tuple(stride)
        b = b[::sx, ::sy, ::sz]
        x, y, z = x[::sx], y[::sy], z[::sz]

    if target_shape is not None and tuple(target_shape) != b.shape[:3]:
        b, x, y, z = _resample_field(b, x, y, z, target_shape)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if (
        align_domain
        and nf2_xy_range_ds is not None
        and nf2_z_top_mm is not None
    ):
        cr = np.asarray(nf2_xy_range_ds, dtype=np.float64)
        if cr.shape != (2, 2):
            raise ValueError(f"nf2_xy_range_ds must have shape (2, 2), got {cr.shape}")
        mmds = float(Mm_per_ds)
        x_lo, x_hi = float(cr[0, 0]) * mmds, float(cr[0, 1]) * mmds
        y_lo, y_hi = float(cr[1, 0]) * mmds, float(cr[1, 1]) * mmds
        z_lo, z_hi = 0.0, float(nf2_z_top_mm)
        x = _linear_map_to_range(x, x_lo, x_hi)
        y = _linear_map_to_range(y, y_lo, y_hi)
        z = _linear_map_to_range(z, z_lo, z_hi)

    coords = _mesh_coords(x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)).reshape(-1, 3)
    values = b.reshape(-1, 3)
    finite_mask = np.isfinite(coords).all(axis=1) & np.isfinite(values).all(axis=1)
    coords = coords[finite_mask]
    values = values[finite_mask]

    if max_samples is not None and coords.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(coords.shape[0], size=int(max_samples), replace=False)
        coords = coords[idx]
        values = values[idx]

    return {
        "coords": (coords / float(Mm_per_ds)).astype(np.float32),
        "b_true": (values / float(G_per_dB)).astype(np.float32),
        "coords_Mm": coords.astype(np.float32),
        "b_true_G": values.astype(np.float32),
        "grid_shape": np.asarray(b.shape[:3], dtype=np.int32),
        "G_per_dB": np.asarray(float(G_per_dB), dtype=np.float32),
        "Mm_per_ds": np.asarray(float(Mm_per_ds), dtype=np.float32),
    }


def prepare_nf2_init_samples(*args, **kwargs) -> dict[str, np.ndarray]:
    """Alias with the public NF2-oriented name."""
    return prepare_nfourier_init_samples(*args, **kwargs)


def write_npy_slices_for_nf2(field: PinoField | str | Path, output_path: str | Path) -> Path:
    """Write a PINO cube as an NF2 `NumpyDataModule`-compatible `.npy` file."""
    field = load_pino_field(field) if isinstance(field, (str, Path)) else field
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, field.b.astype(np.float32))
    return output_path


def _validate_pino_field(field: PinoField) -> PinoField:
    b = np.asarray(field.b, dtype=np.float32)
    if b.ndim != 4 or b.shape[-1] != 3:
        raise ValueError(f"Expected PINO field shape [nx, ny, nz, 3], got {b.shape}")
    if not np.isfinite(b).all():
        raise ValueError("PINO field contains NaN or infinite values")
    if field.field_units.lower() not in {"g", "gauss"}:
        raise ValueError(f"Expected PINO field units in Gauss, got {field.field_units!r}")
    if field.coordinate_units.lower() not in {"mm", "megameter", "megameters"}:
        raise ValueError(f"Expected PINO coordinate units in Mm, got {field.coordinate_units!r}")

    coords = []
    for axis_name, arr, expected in zip("xyz", (field.x_coords, field.y_coords, field.z_coords), b.shape[:3]):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != expected:
            raise ValueError(f"{axis_name}_coords length {arr.shape} does not match field axis {expected}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{axis_name}_coords contains NaN or infinite values")
        coords.append(arr)
    return PinoField(b, coords[0], coords[1], coords[2], field.field_units, field.coordinate_units, field.source_path)


def _coords_from_export(data: np.lib.npyio.NpzFile, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all(k in data for k in ("x_coords", "y_coords", "z_coords")):
        return (
            np.asarray(data["x_coords"], dtype=np.float32),
            np.asarray(data["y_coords"], dtype=np.float32),
            np.asarray(data["z_coords"], dtype=np.float32),
        )

    dx = float(np.asarray(data.get("dx_Mm", data.get("dx", 1.0))))
    dy = float(np.asarray(data.get("dy_Mm", data.get("dy", dx))))
    dz = float(np.asarray(data.get("dz_Mm", data.get("dz", dy))))
    return (
        np.arange(b.shape[0], dtype=np.float32) * dx,
        np.arange(b.shape[1], dtype=np.float32) * dy,
        np.arange(b.shape[2], dtype=np.float32) * dz,
    )


def _default_coords_for_field(b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if b.ndim < 3:
        raise ValueError(f"Expected at least 3 dimensions for field, got {b.shape}")
    return tuple(np.arange(n, dtype=np.float32) for n in b.shape[:3])


def _mesh_coords(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).astype(np.float32)


def _resample_field(
    b: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    target_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from skimage.transform import resize
    except ImportError as exc:
        raise ImportError("Install scikit-image to use target_shape resampling") from exc

    target_shape = tuple(int(v) for v in target_shape)
    resized = resize(
        b,
        (*target_shape, 3),
        order=1,
        mode="edge",
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    new_coords = tuple(np.linspace(arr[0], arr[-1], n, dtype=np.float32) for arr, n in zip((x, y, z), target_shape))
    return resized, new_coords[0], new_coords[1], new_coords[2]


def _stride_tuple(stride: int | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    stride = tuple(int(s) for s in stride)
    if len(stride) != 3 or any(s < 1 for s in stride):
        raise ValueError(f"stride must be a positive int or 3-tuple, got {stride}")
    return stride


def _spacing(arr: np.ndarray) -> float:
    if arr.shape[0] < 2:
        return 1.0
    return float(np.mean(np.diff(arr)))


def _scalar_str(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(arr.tolist())
