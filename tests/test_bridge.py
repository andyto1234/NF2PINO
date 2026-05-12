from pathlib import Path

import numpy as np
import pytest

from nf2pino.bridge import load_pino_field, prepare_nf2_init_samples


def test_pino_npz_loads_and_normalizes_samples(tmp_path: Path):
    field = np.arange(4 * 3 * 2 * 3, dtype=np.float32).reshape(4, 3, 2, 3)
    path = tmp_path / "rtmag_extrapolation.npz"
    np.savez_compressed(
        path,
        b=field,
        x_coords=np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32),
        y_coords=np.array([0.0, 3.0, 6.0], dtype=np.float32),
        z_coords=np.array([0.0, 5.0], dtype=np.float32),
        component_order=np.array(["Bx", "By", "Bz"]),
        coordinate_units=np.array("Mm"),
        field_units=np.array("G"),
    )

    loaded = load_pino_field(path)
    assert loaded.shape == (4, 3, 2, 3)
    assert loaded.spacing_Mm == (2.0, 3.0, 5.0)

    samples = prepare_nf2_init_samples(loaded, G_per_dB=10, Mm_per_ds=2, max_samples=None)
    assert samples["coords"].shape == (24, 3)
    assert samples["b_true"].shape == (24, 3)
    assert np.allclose(samples["coords"][1], [0.0, 0.0, 2.5])
    assert np.allclose(samples["b_true"][0], field.reshape(-1, 3)[0] / 10)


def test_prepare_samples_nf2_domain_alignment_restricts_ds_coords(tmp_path: Path):
    field = np.arange(4 * 3 * 2 * 3, dtype=np.float32).reshape(4, 3, 2, 3)
    path = tmp_path / "rtmag_extrapolation.npz"
    np.savez_compressed(
        path,
        b=field,
        x_coords=np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float32),
        y_coords=np.array([0.0, 50.0, 100.0], dtype=np.float32),
        z_coords=np.array([0.0, 80.0], dtype=np.float32),
        component_order=np.array(["Bx", "By", "Bz"]),
        coordinate_units=np.array("Mm"),
        field_units=np.array("G"),
    )
    loaded = load_pino_field(path)
    cr = np.array([[0.0, 0.75], [0.0, 0.46]], dtype=np.float64)
    samples = prepare_nf2_init_samples(
        loaded,
        G_per_dB=10.0,
        Mm_per_ds=100.0,
        max_samples=None,
        align_domain=True,
        nf2_xy_range_ds=cr,
        nf2_z_top_mm=40.0,
    )
    c = samples["coords"]
    assert c[:, 0].min() >= -1e-5 and c[:, 0].max() <= 0.75 + 1e-4
    assert c[:, 1].min() >= -1e-5 and c[:, 1].max() <= 0.46 + 1e-4
    assert c[:, 2].min() >= -1e-5 and c[:, 2].max() <= 40.0 / 100.0 + 1e-4


def test_target_shape_resize_suppresses_peak_b_amplitude(tmp_path: Path):
    """skimage.resize + anti_aliasing averages strong pixels — bad for PINO warm-start peaks."""
    try:
        from skimage.transform import resize  # noqa: F401
    except ImportError:
        pytest.skip("scikit-image required")

    nx, ny, nz = 16, 16, 8
    b = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    b[8, 4, 3, 0] = 2400.0
    x = np.arange(nx, dtype=np.float32) * 10.0
    y = np.arange(ny, dtype=np.float32) * 10.0
    z = np.arange(nz, dtype=np.float32) * 10.0
    path = tmp_path / "peak.npz"
    np.savez_compressed(
        path,
        b=b,
        x_coords=x,
        y_coords=y,
        z_coords=z,
        component_order=np.array(["Bx", "By", "Bz"]),
        coordinate_units=np.array("Mm"),
        field_units=np.array("G"),
    )
    loaded = load_pino_field(path)
    full = prepare_nf2_init_samples(loaded, G_per_dB=2500.0, Mm_per_ds=100.0, max_samples=None)
    res = prepare_nf2_init_samples(loaded, G_per_dB=2500.0, Mm_per_ds=100.0, max_samples=None, target_shape=(8, 8, 4))
    assert float(np.max(np.abs(res["b_true_G"]))) < float(np.max(np.abs(full["b_true_G"]))) * 0.5
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        b=np.zeros((2, 2, 2, 3), dtype=np.float32),
        component_order=np.array(["By", "Bx", "Bz"]),
    )

    with pytest.raises(ValueError, match="component order"):
        load_pino_field(path)
