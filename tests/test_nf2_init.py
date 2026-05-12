from pathlib import Path

import numpy as np
import pytest
import torch

import nf2pino.nf2_init as nf2_init
from nf2pino.nf2_init import inject_pino_init_config, prefit_nf2_from_pino
from nf2pino.paths import ensure_repo_paths


def _write_tiny_field(path: Path) -> Path:
    x = np.linspace(0, 3, 4, dtype=np.float32)
    y = np.linspace(0, 2, 3, dtype=np.float32)
    z = np.linspace(0, 1, 2, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    b = np.stack([xx + 1, yy - 1, 0.5 * zz], axis=-1).astype(np.float32)
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
    return path


def test_prefit_raises_when_cartesian_missing_fits_bounds(tmp_path: Path):
    ensure_repo_paths()
    field_path = _write_tiny_field(tmp_path / "pino.npz")
    with pytest.raises(ValueError, match="coord_range"):
        prefit_nf2_from_pino(
            field_path,
            tmp_path / "pino_init.pt",
            model_kwargs={"type": "b", "dim": 8, "n_layers": 1, "activation": "tanh"},
            data_config={
                "type": "cartesian",
                "coord_range": [],
                "max_height": 40.0,
                "G_per_dB": 1.0,
                "Mm_per_ds": 1.0,
            },
            init_config={"steps": 1, "batch_size": 8, "device": "cpu", "max_samples": None, "align_domain": True},
        )


def test_prefit_summary_reports_kilogauss_scale_targets(tmp_path: Path):
    """Stride subsampling keeps strong-B pixels; sine MLP should reach a fraction of max |b_true|."""
    ensure_repo_paths()
    nx, ny, nz = 14, 12, 10
    x = np.linspace(0.0, 280.0, nx, dtype=np.float32)
    y = np.linspace(0.0, 200.0, ny, dtype=np.float32)
    z = np.linspace(0.0, 180.0, nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    b = np.stack(
        [
            1900.0 * np.sin(xx / 55.0),
            600.0 * np.cos(yy / 40.0),
            400.0 * np.sin(zz / 45.0),
        ],
        axis=-1,
    ).astype(np.float32)
    path = tmp_path / "solarish.npz"
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
    dc = {
        "type": "cartesian",
        "coord_range": [np.array([[0.0, 0.72], [0.0, 0.44]], dtype=np.float64)],
        "max_height": 45.0,
        "Mm_per_ds": 110.0,
        "G_per_dB": 2500.0,
    }
    meta = tmp_path / "meta.pt"
    summary = prefit_nf2_from_pino(
        path,
        meta,
        model_kwargs={"type": "b", "dim": 96, "n_layers": 3, "activation": "sine"},
        data_config=dc,
        init_config={
            "steps": 6000,
            "batch_size": 2048,
            "max_samples": 12000,
            "stride": (2, 2, 2),
            "device": "cpu",
            "lr": 3e-3,
            "log_every": 2000,
        },
    )
    assert summary["b_true_max_abs_dB"] > 0.75
    assert summary["prefit_sample_max_abs_pred_dB"] > 0.55


def test_prefit_saves_nf2_meta_checkpoint(tmp_path: Path):
    ensure_repo_paths()
    from nf2.train.model import BModel

    field_path = _write_tiny_field(tmp_path / "pino.npz")
    meta_path = tmp_path / "pino_init.pt"
    summary = prefit_nf2_from_pino(
        field_path,
        meta_path,
        model_kwargs={"type": "b", "dim": 8, "n_layers": 1, "activation": "tanh"},
        data_config={"G_per_dB": 1.0, "Mm_per_ds": 1.0},
        init_config={"steps": 5, "batch_size": 8, "max_samples": None, "device": "cpu", "lr": 1e-3},
    )

    checkpoint = torch.load(meta_path, map_location="cpu")
    model = BModel(dim=8, n_layers=1, activation="tanh")
    model.load_state_dict(checkpoint["m"])

    assert meta_path.exists()
    assert summary["samples"] == 24
    assert np.isfinite(summary["final_mse"])


def test_inject_pino_init_preserves_physics_losses(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_prefit(pino_field, output_path, *, model_kwargs, data_config, init_config, device):
        captured["pino_field"] = pino_field
        Path(output_path).write_bytes(b"checkpoint")
        return {"meta_path": str(output_path)}

    monkeypatch.setattr(nf2_init, "prefit_nf2_from_pino", fake_prefit)
    training = {
        "init_mode": "pino",
        "init_from_pino": "pino.npz",
        "pino_init": {"steps": 1},
        "loss_config": [
            {"type": "boundary", "lambda": 1.0, "ds_id": ["boundary_01", "potential"]},
            {"type": "force_free", "lambda": 0.1},
            {"type": "divergence", "lambda": 0.1},
        ],
    }

    result = inject_pino_init_config(
        training,
        model_kwargs={"type": "b", "dim": 8},
        data_config={"G_per_dB": 2500.0, "Mm_per_ds": 115.2},
        base_path=tmp_path,
    )

    loss_types = [item["type"] for item in result["loss_config"]]
    assert result["meta_path"].endswith("pino_init.pt")
    assert captured["pino_field"] == "pino.npz"
    assert loss_types == ["boundary", "force_free", "divergence"]
