import json
from pathlib import Path

import numpy as np

from nf2pino.paths import ensure_repo_paths
from nf2pino.workflow import run_hybrid_workflow


def _write_pino_field(path: Path) -> Path:
    x = np.linspace(0.0, 30.0, 4, dtype=np.float32)
    y = np.linspace(0.0, 20.0, 3, dtype=np.float32)
    z = np.linspace(0.0, 10.0, 2, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    b = np.stack([xx + 5.0, yy - 2.0, zz + 1.0], axis=-1).astype(np.float32)
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


def test_workflow_rewrites_paths_and_writes_histories(tmp_path: Path, monkeypatch):
    ensure_repo_paths()
    import nf2.extrapolate as nf2_extrapolate
    import nf2pino.workflow as workflow

    pino_path = _write_pino_field(tmp_path / "pino.npz")
    output_dir = tmp_path / "workflow"
    stale_run_dir = output_dir / "nf2_pino_init"
    stale_work_dir = output_dir / "work" / "nf2_pino_init"
    stale_run_dir.mkdir(parents=True, exist_ok=True)
    stale_work_dir.mkdir(parents=True, exist_ok=True)
    (stale_run_dir / "last.ckpt").write_text("stale")
    (stale_work_dir / "stale.txt").write_text("stale")

    calls = []

    def fake_run(*, base_path, data, work_directory=None, callbacks=None, logging=None, model=None, training=None, config=None):
        run_dir = Path(base_path)
        work_dir = Path(work_directory)
        calls.append(
            {
                "base_path": run_dir,
                "work_directory": work_dir,
                "has_last_ckpt": (run_dir / "last.ckpt").exists(),
                "callbacks": callbacks or [],
                "training": training or {},
            }
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        history = [{"epoch": 0, "global_step": 10, "total_loss": 1.0}]
        for callback in callbacks or []:
            callback.output_path.parent.mkdir(parents=True, exist_ok=True)
            callback.output_path.write_text(json.dumps(history))
        if training and training.get("init_mode") == "pino":
            (run_dir / "pino_init.pt.json").write_text(json.dumps({"history": [{"step": 1, "mse": 0.1}]}))

    monkeypatch.setattr(nf2_extrapolate, "run", fake_run)
    monkeypatch.setattr(
        workflow,
        "_summarize_nf2_result",
        lambda **kwargs: {
            "result_path": str(kwargs["run_dir"] / "extrapolation_result.nf2"),
            "metrics": {"finite": True},
            "comparison_to_pino": {"delta_l2_rel": 0.5},
            "bottom_boundary_mismatch": {"mean_abs": 1.0, "rmse": 2.0, "max_abs": 3.0},
        },
    )

    benchmark = run_hybrid_workflow(
        nf2_config={
            "data": {"type": "fits", "slices": [{"fits_path": {"Bp": "bp", "Bt": "bt", "Br": "br"}}]},
            "model": {"type": "b", "dim": 8, "n_layers": 1},
            "training": {"epochs": 1, "loss_config": [{"type": "boundary", "lambda": 1.0, "ds_id": ["boundary_01", "potential"]}]},
        },
        output_dir=output_dir,
        pino_field=pino_path,
        run_scratch=True,
        resume_nf2=False,
    )

    assert len(calls) == 2
    scratch_call, hybrid_call = calls
    assert scratch_call["base_path"] == output_dir / "nf2_scratch"
    assert scratch_call["work_directory"] == output_dir / "work" / "nf2_scratch"
    assert hybrid_call["base_path"] == output_dir / "nf2_pino_init"
    assert hybrid_call["work_directory"] == output_dir / "work" / "nf2_pino_init"
    assert not hybrid_call["has_last_ckpt"]
    assert benchmark["nf2_scratch"]["history"][0]["total_loss"] == 1.0
    assert benchmark["nf2_pino_init"]["history"][0]["total_loss"] == 1.0
    assert benchmark["scratch"]["history"][0]["total_loss"] == 1.0
    assert benchmark["hybrid"]["history"][0]["total_loss"] == 1.0
    assert benchmark["nf2_pino_init"]["prefit_summary"]["history"][0]["mse"] == 0.1
    assert benchmark["nf2_pino_init"]["bottom_boundary_mismatch"]["mean_abs"] == 1.0
    assert benchmark["benchmark_path"].endswith("benchmark_summary.json")


def test_workflow_resume_keeps_last_checkpoint(tmp_path: Path, monkeypatch):
    ensure_repo_paths()
    import nf2.extrapolate as nf2_extrapolate
    import nf2pino.workflow as workflow

    pino_path = _write_pino_field(tmp_path / "pino.npz")
    output_dir = tmp_path / "workflow"
    run_dir = output_dir / "nf2_pino_init"
    work_dir = output_dir / "work" / "nf2_pino_init"
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "last.ckpt").write_text("resume-me")

    captured = {}

    def fake_run(*, base_path, data, work_directory=None, callbacks=None, logging=None, model=None, training=None, config=None):
        captured["has_last_ckpt"] = (Path(base_path) / "last.ckpt").exists()
        Path(base_path).mkdir(parents=True, exist_ok=True)
        Path(work_directory).mkdir(parents=True, exist_ok=True)
        for callback in callbacks or []:
            callback.output_path.parent.mkdir(parents=True, exist_ok=True)
            callback.output_path.write_text("[]")
        if training and training.get("init_mode") == "pino":
            (Path(base_path) / "pino_init.pt.json").write_text(json.dumps({}))

    monkeypatch.setattr(nf2_extrapolate, "run", fake_run)
    monkeypatch.setattr(workflow, "_summarize_nf2_result", lambda **kwargs: {"result_path": "dummy"})

    run_hybrid_workflow(
        nf2_config={
            "data": {"type": "fits", "slices": [{"fits_path": {"Bp": "bp", "Bt": "bt", "Br": "br"}}]},
            "model": {"type": "b", "dim": 8, "n_layers": 1},
            "training": {"epochs": 1, "loss_config": [{"type": "boundary", "lambda": 1.0, "ds_id": ["boundary_01", "potential"]}]},
        },
        output_dir=output_dir,
        pino_field=pino_path,
        resume_nf2=True,
    )

    assert captured["has_last_ckpt"] is True


def test_notebook_uses_workflow_single_source_of_truth():
    notebook = Path("notebooks/pino_nf2_hybrid.ipynb").read_text()
    assert "run_hybrid_workflow(" in notebook
    assert "prefit_nf2_from_pino(" not in notebook
