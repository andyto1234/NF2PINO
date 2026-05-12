"""Warm-start NF2 models from PINO magnetic-field cubes."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from .bridge import PinoField, prepare_nf2_init_samples
from .paths import ensure_repo_paths


def prefit_nf2_from_pino(
    pino_field: PinoField | str | Path,
    output_path: str | Path,
    *,
    model_kwargs: dict[str, Any] | None = None,
    data_config: dict[str, Any] | None = None,
    init_config: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Fit an NF2 model to a PINO cube and save a meta checkpoint.

    The saved checkpoint intentionally contains only `{'m': state_dict}` so it
    can be consumed by `NF2Module(..., meta_path=...)` without changing NF2's
    training loop or physics losses.
    """
    ensure_repo_paths()
    from nf2.train.model import BModel, VectorPotentialModel

    model_kwargs = dict(model_kwargs or {"type": "b", "dim": 256})
    init_config = dict(init_config or {})
    data_config = dict(data_config or {})

    device = _resolve_device(device or init_config.get("device"))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    align_domain = bool(init_config.get("align_domain", True))
    nf2_xy_range_ds: np.ndarray | None = None
    nf2_z_top_mm: float | None = None
    if align_domain and data_config.get("type") == "cartesian":
        cr_list = data_config.get("coord_range") or []
        if cr_list:
            nf2_xy_range_ds = np.asarray(cr_list[0], dtype=np.float64)
        mh = data_config.get("max_height")
        if mh is not None:
            nf2_z_top_mm = float(mh)
        if nf2_xy_range_ds is None or nf2_z_top_mm is None:
            raise ValueError(
                "PINO prefit with align_domain=True and a cartesian NF2 data config needs "
                "non-empty data['coord_range'] (FITS footprint in ds) and data['max_height'] (Mm). "
                "Ensure NF2 runs with a FITS data module so inject_pino_init_config receives "
                "data_module.config from FITSDataModule."
            )

    if init_config.get("target_shape"):
        warnings.warn(
            "pino_init.target_shape uses skimage.resize (smoothing) and typically slashes peak |B| "
            "in supervised samples; use stride-based subsampling instead unless you intend to smooth.",
            UserWarning,
            stacklevel=2,
        )

    samples = prepare_nf2_init_samples(
        pino_field,
        G_per_dB=float(data_config.get("G_per_dB", init_config.get("G_per_dB", 2500.0))),
        Mm_per_ds=float(data_config.get("Mm_per_ds", init_config.get("Mm_per_ds", 0.36 * 320))),
        max_samples=init_config.get("max_samples", 200_000),
        stride=init_config.get("stride"),
        target_shape=_tuple_or_none(init_config.get("target_shape")),
        z_range_Mm=_tuple_or_none(init_config.get("z_range_Mm")),
        seed=int(init_config.get("seed", 0)),
        align_domain=align_domain,
        nf2_xy_range_ds=nf2_xy_range_ds,
        nf2_z_top_mm=nf2_z_top_mm,
    )

    model_type = model_kwargs.pop("type", "b")
    if model_type == "b":
        model = BModel(**model_kwargs)
    elif model_type == "vector_potential":
        model = VectorPotentialModel(**model_kwargs)
    else:
        raise ValueError("PINO warm-start currently supports NF2 model types 'b' and 'vector_potential'")
    model.to(device)

    coords = torch.as_tensor(samples["coords"], dtype=torch.float32)
    b_true = torch.as_tensor(samples["b_true"], dtype=torch.float32)
    sample_weights, weight_summary = _build_sample_weights(coords, b_true, init_config)

    steps = int(init_config.get("steps", init_config.get("max_steps", 1000)))
    batch_size = int(init_config.get("batch_size", min(8192, len(coords))))
    lr = float(init_config.get("lr", 5e-4))
    weight_decay = float(init_config.get("weight_decay", 0.0))
    log_every = int(init_config.get("log_every", max(1, steps // 10)))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(int(init_config.get("seed", 0)))
    history: list[dict[str, float]] = []
    start = perf_counter()
    final_loss = np.nan

    model.train()
    for step in range(steps):
        idx = torch.randint(0, len(coords), (batch_size,), generator=generator)
        batch_coords = coords[idx].to(device)
        batch_b = b_true[idx].to(device)
        batch_weights = sample_weights[idx].to(device)
        if model_type == "vector_potential":
            batch_coords.requires_grad_(True)

        out = model(batch_coords, compute_jacobian=False)
        point_mse = torch.mean((out["b"] - batch_b) ** 2, dim=-1)
        loss = (point_mse * batch_weights).sum() / batch_weights.sum().clamp_min(1e-12)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        if step == 0 or (step + 1) % log_every == 0 or step == steps - 1:
            history.append({"step": float(step + 1), "mse": final_loss})

    bmax_dB = float(np.max(np.abs(samples["b_true"])))

    model.eval()
    with torch.no_grad():
        sub = torch.randint(0, len(coords), (min(4096, len(coords)),), generator=generator)
        pred_mx = float(
            torch.abs(model(coords[sub].to(device), compute_jacobian=False)["b"]).max().cpu()
        )

    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"m": cpu_state}, output_path)

    summary = {
        "meta_path": str(output_path),
        "model_type": model_type,
        "model_kwargs": {"type": model_type, **model_kwargs},
        "samples": int(len(coords)),
        "grid_shape": samples["grid_shape"].astype(int).tolist(),
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "final_mse": final_loss,
        "b_true_max_abs_dB": bmax_dB,
        "prefit_sample_max_abs_pred_dB": pred_mx,
        **weight_summary,
        "runtime_seconds": perf_counter() - start,
        "history": history,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def inject_pino_init_config(
    training: dict[str, Any],
    *,
    model_kwargs: dict[str, Any],
    data_config: dict[str, Any],
    base_path: str | Path,
) -> dict[str, Any]:
    """Resolve optional PINO init keys in an NF2 training config."""
    training = dict(training)
    init_mode = training.pop("init_mode", None)
    init_from_pino = training.pop("init_from_pino", None)
    init_config = training.pop("pino_init", {})

    if init_mode not in (None, "pino"):
        raise ValueError(f"Unknown training.init_mode={init_mode!r}")
    if init_mode != "pino" and init_from_pino is None:
        return training
    if init_from_pino is None:
        raise ValueError("training.init_mode='pino' requires training.init_from_pino")

    init_config = dict(init_config or {})
    meta_path = Path(init_config.get("meta_path", Path(base_path) / "pino_init.pt"))
    summary = prefit_nf2_from_pino(
        init_from_pino,
        meta_path,
        model_kwargs=model_kwargs,
        data_config=data_config,
        init_config=init_config,
        device=init_config.get("device"),
    )
    training["meta_path"] = summary["meta_path"]
    return training


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _tuple_or_none(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return tuple(value)


def _build_sample_weights(
    coords: torch.Tensor,
    b_true: torch.Tensor,
    init_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    strong_field_percentile = float(init_config.get("strong_field_percentile", 90.0))
    strong_field_weight = float(init_config.get("strong_field_weight", 5.0))
    bottom_weight = float(init_config.get("bottom_weight", 2.0))

    weights = torch.ones(len(coords), dtype=torch.float32)
    strength = torch.linalg.norm(b_true, dim=-1)
    threshold = torch.quantile(strength, strong_field_percentile / 100.0)
    strong_mask = strength >= threshold
    if strong_field_weight != 1.0:
        weights[strong_mask] *= strong_field_weight

    bottom_z = coords[:, 2].min()
    bottom_mask = torch.isclose(coords[:, 2], bottom_z, atol=1e-7, rtol=0.0)
    if bottom_weight != 1.0:
        weights[bottom_mask] *= bottom_weight

    summary = {
        "strong_field_percentile": strong_field_percentile,
        "strong_field_weight": strong_field_weight,
        "bottom_weight": bottom_weight,
        "strong_field_threshold_dB": float(threshold.detach().cpu()),
        "strong_field_fraction": float(strong_mask.float().mean().detach().cpu()),
        "bottom_fraction": float(bottom_mask.float().mean().detach().cpu()),
    }
    return weights, summary
