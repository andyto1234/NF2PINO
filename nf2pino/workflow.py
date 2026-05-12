"""End-to-end orchestration for PINO-initialized NF2 refinement."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

from .bridge import load_pino_field
from .history import MetricsHistoryCallback
from .metrics import field_quality_metrics, write_metrics
from .paths import configure_runtime_env, ensure_repo_paths
from .pino_runner import run_pino_extrapolation


def _inject_fits_paths_into_config(
    config: dict[str, Any],
    bp: str | Path | None,
    bt: str | Path | None,
    br: str | Path | None,
) -> dict[str, Any]:
    """Replace {%Bp}, {%Bt}, {%Br} placeholders inside a loaded NF2 config dict."""
    if not all([bp, bt, br]):
        return config
    repl = {
        "Bp": str(Path(bp).expanduser().resolve()),
        "Bt": str(Path(bt).expanduser().resolve()),
        "Br": str(Path(br).expanduser().resolve()),
    }
    return _deep_replace_placeholders(config, repl)


def _deep_replace_placeholders(obj: Any, repl: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_replace_placeholders(v, repl) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_replace_placeholders(item, repl) for item in obj]
    if isinstance(obj, str):
        s = obj
        for key, value in repl.items():
            # NB: "{%s}" % key is wrong — % starts printf-style conversion and yields "{Bp}", not "{%Bp}".
            s = s.replace("{%" + key + "}", value)
        return s
    return obj


def _config_has_unresolved_placeholders(obj: Any) -> bool:
    if isinstance(obj, dict):
        return any(_config_has_unresolved_placeholders(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_config_has_unresolved_placeholders(item) for item in obj)
    if isinstance(obj, str):
        return "{%" in obj
    return False


def _prepare_nf2_run_config(
    nf2_config: dict[str, Any],
    *,
    output_dir: Path,
    run_name: str,
) -> tuple[dict[str, Any], Path, Path]:
    config = copy.deepcopy(nf2_config)
    run_dir = output_dir / run_name
    work_dir = output_dir / "work" / run_name
    config["base_path"] = str(run_dir)
    config["work_directory"] = str(work_dir)
    config.setdefault("logging", {}).setdefault("mode", "disabled")
    return config, run_dir, work_dir


def _clear_workflow_run_dirs(run_dir: Path, work_dir: Path) -> None:
    for path in (run_dir, work_dir):
        if path.exists():
            shutil.rmtree(path)


def _load_json_if_exists(path: str | Path) -> dict[str, Any] | list[Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _summarize_nf2_result(
    *,
    run_dir: Path,
    pino_field,
    device: str,
) -> dict[str, Any]:
    result_path = run_dir / "extrapolation_result.nf2"
    summary: dict[str, Any] = {"result_path": str(result_path)}
    if not result_path.is_file():
        summary["analysis_error"] = f"missing_result:{result_path}"
        return summary

    try:
        from .compare import compare_b_on_same_grid, evaluate_nf2_on_pino_axes

        b_nf2, _ = evaluate_nf2_on_pino_axes(
            result_path,
            x_m=pino_field.x_coords,
            y_m=pino_field.y_coords,
            z_m=pino_field.z_coords,
            device=_resolve_analysis_device(device),
            progress=False,
            compute_jacobian=False,
            align_domain=True,
        )
        comparison = compare_b_on_same_grid(pino_field.b, b_nf2, pino_field.spacing_Mm)
        bottom_delta = b_nf2[:, :, 0, :] - pino_field.b[:, :, 0, :]
        summary.update(
            {
                "metrics": comparison["metrics_nf2"],
                "comparison_to_pino": comparison,
                "bottom_boundary_mismatch": {
                    "mean_abs": float(abs(bottom_delta).mean()),
                    "rmse": float((bottom_delta**2).mean() ** 0.5),
                    "max_abs": float(abs(bottom_delta).max()),
                },
            }
        )
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps / heavy artifacts
        summary["analysis_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def _resolve_analysis_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def run_hybrid_workflow(
    *,
    nf2_config: dict[str, Any],
    output_dir: str | Path,
    pino_field: str | Path | None = None,
    pino_model: str | Path | None = None,
    magnetogram: str | Path | None = None,
    bp: str | Path | None = None,
    bt: str | Path | None = None,
    br: str | Path | None = None,
    nf2_fits_bp: str | Path | None = None,
    nf2_fits_bt: str | Path | None = None,
    nf2_fits_br: str | Path | None = None,
    sharp_jsoc_time: str | None = None,
    sharp_jsoc_harpnum: int | None = None,
    sharp_jsoc_overwrite_fits: bool = False,
    nx: int = 512,
    ny: int = 256,
    device: str = "auto",
    run_scratch: bool = False,
    prefit_steps: int | None = None,
    resume_nf2: bool = False,
    force_pino: bool = False,
) -> dict[str, Any]:
    """Run PINO, optional scratch NF2, and PINO-initialized NF2 refinement."""
    output_dir = Path(output_dir)
    configure_runtime_env(output_dir / "cache")
    ensure_repo_paths()
    from nf2.extrapolate import run as run_nf2

    fits_bp = nf2_fits_bp if nf2_fits_bp is not None else bp
    fits_bt = nf2_fits_bt if nf2_fits_bt is not None else bt
    fits_br = nf2_fits_br if nf2_fits_br is not None else br

    pino_hmi_data = None
    if sharp_jsoc_time is not None and sharp_jsoc_harpnum is not None:
        need_nf2_fits = not all([fits_bp, fits_bt, fits_br])
        need_pino_bottom = (
            pino_field is None
            and pino_model is not None
            and magnetogram is None
            and not (bp and bt and br)
        )
        if need_nf2_fits or need_pino_bottom:
            from rtmag.process.download.dl_map import fetch_sharp_cea_bundle

            sharp_cache = output_dir / "cache" / "sharp_jsoc_fits"
            _, sharp_hmi_data, sharp_paths, _ = fetch_sharp_cea_bundle(
                sharp_jsoc_time, sharp_jsoc_harpnum, sharp_cache, overwrite=sharp_jsoc_overwrite_fits
            )
            if need_nf2_fits:
                fits_bp = fits_bp or sharp_paths["Bp"]
                fits_bt = fits_bt or sharp_paths["Bt"]
                fits_br = fits_br or sharp_paths["Br"]
            if need_pino_bottom:
                pino_hmi_data = sharp_hmi_data
    nf2_config = copy.deepcopy(nf2_config)
    nf2_config = _inject_fits_paths_into_config(nf2_config, fits_bp, fits_bt, fits_br)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark: dict[str, Any] = {
        "output_dir": str(output_dir),
        "resume_nf2": resume_nf2,
        "force_pino": force_pino,
    }

    if pino_field is None:
        if pino_model is None:
            raise ValueError("Provide either pino_field or pino_model plus magnetogram/component inputs")
        pino_field = output_dir / "cache" / "pino_extrapolation.npz"
        benchmark["pino"] = run_pino_extrapolation(
            model_path=pino_model,
            output_path=pino_field,
            magnetogram=magnetogram,
            hmi_data=pino_hmi_data,
            bp=bp,
            bt=bt,
            br=br,
            nx=nx,
            ny=ny,
            device=device,
            overwrite=force_pino,
        )
    else:
        benchmark["pino"] = {"path": str(pino_field), "cached": True}

    field = load_pino_field(pino_field)
    benchmark["pino"]["metrics"] = field_quality_metrics(field.b, field.spacing_Mm)

    if _config_has_unresolved_placeholders(nf2_config):
        raise ValueError(
            "NF2 config still contains {%...} placeholders (e.g. {%Bp}). "
            "Pass nf2_fits_bp/nf2_fits_bt/nf2_fits_br (or bp/bt/br when using FITS for PINO), "
            "or pass sharp_jsoc_time and sharp_jsoc_harpnum to download SHARP CEA FITS into the output cache."
        )

    if run_scratch:
        scratch_config, scratch_run_dir, scratch_work_dir = _prepare_nf2_run_config(
            nf2_config, output_dir=output_dir, run_name="nf2_scratch"
        )
        if not resume_nf2:
            _clear_workflow_run_dirs(scratch_run_dir, scratch_work_dir)
        scratch_history_path = scratch_run_dir / "validation_history.json"
        scratch_history_callback = MetricsHistoryCallback(scratch_history_path)
        scratch_start = perf_counter()
        run_nf2(**scratch_config, callbacks=[scratch_history_callback])
        scratch_section = {
            "runtime_seconds": perf_counter() - scratch_start,
            "run_dir": str(scratch_run_dir),
            "work_directory": str(scratch_work_dir),
            "history_path": str(scratch_history_path),
            "history": _load_json_if_exists(scratch_history_path),
        }
        scratch_section.update(_summarize_nf2_result(run_dir=scratch_run_dir, pino_field=field, device=device))
        benchmark["nf2_scratch"] = scratch_section
        benchmark["scratch"] = scratch_section

    hybrid_config, hybrid_run_dir, hybrid_work_dir = _prepare_nf2_run_config(
        nf2_config, output_dir=output_dir, run_name="nf2_pino_init"
    )
    if not resume_nf2:
        _clear_workflow_run_dirs(hybrid_run_dir, hybrid_work_dir)
    training = hybrid_config.setdefault("training", {})
    training["init_mode"] = "pino"
    training["init_from_pino"] = str(pino_field)
    pino_init = training.setdefault("pino_init", {})
    pino_init.setdefault("device", device)
    if prefit_steps is not None:
        pino_init["steps"] = int(prefit_steps)

    hybrid_history_path = hybrid_run_dir / "validation_history.json"
    hybrid_history_callback = MetricsHistoryCallback(hybrid_history_path)
    hybrid_start = perf_counter()
    run_nf2(**hybrid_config, callbacks=[hybrid_history_callback])
    hybrid_section = {
        "runtime_seconds": perf_counter() - hybrid_start,
        "run_dir": str(hybrid_run_dir),
        "work_directory": str(hybrid_work_dir),
        "history_path": str(hybrid_history_path),
        "history": _load_json_if_exists(hybrid_history_path),
    }
    prefit_summary_path = hybrid_run_dir / "pino_init.pt.json"
    if prefit_summary_path.is_file():
        hybrid_section["prefit_summary_path"] = str(prefit_summary_path)
        hybrid_section["prefit_summary"] = _load_json_if_exists(prefit_summary_path)
    hybrid_section.update(_summarize_nf2_result(run_dir=hybrid_run_dir, pino_field=field, device=device))
    benchmark["nf2_pino_init"] = hybrid_section
    benchmark["hybrid"] = hybrid_section

    benchmark_path = write_metrics(output_dir / "benchmark_summary.json", benchmark)
    benchmark["benchmark_path"] = str(benchmark_path)
    return benchmark


def load_json_or_yaml_config(
    path: str | Path,
    replacements: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load an NF2 config from YAML or JSON.

    ``replacements`` maps keys to values for NF2-style tokens ``{%key}`` in the
    file (same as ``nf2-extrapolate --config x --Bp /path ...``).
    """
    path = Path(path)
    text = path.read_text()
    if replacements:
        for key, value in replacements.items():
            text = text.replace("{%" + key + "}", str(value))
    if path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)
