# PINO to NF2 Hybrid NLFFF Workflow

## Overview

This workflow keeps RTMAG/PINO and NF2 in their existing roles:

1. RTMAG/PINO extrapolates a 3D magnetic field cube from a vector magnetogram.
2. `nf2pino` converts that cube into NF2-normalized samples.
3. NF2 is warm-started by fitting its existing neural field to those samples and saving an NF2-compatible `meta_path`.
4. Standard NF2 refinement then continues with the normal boundary, force-free, and divergence losses.

The PINO cube is an initialization only. It is not used as a persistent target during NF2 refinement.

## Execution Model

The notebook and CLI now use the same single source of truth: `run_hybrid_workflow()`.

- No separate manual prefit path is used in the notebook.
- Workflow-owned runs always live under `output_dir`:
  - `output_dir/nf2_pino_init`
  - `output_dir/nf2_scratch`
  - `output_dir/work/...`
- By default, NF2 starts fresh.
- Use `--resume-nf2` only when you intentionally want to continue workflow-owned `last.ckpt` files.
- PINO caches are reused by default.
- Use `--force-pino` to rebuild `output_dir/cache/pino_extrapolation.npz`.

## Runtime Environment

Recommended environment: `fast_NLFFF`.

The workflow, CLI, notebook, and verification script all set safe defaults for:

- `SUNPY_CONFIGDIR`
- `MPLCONFIGDIR`

These directories are created under the workflow cache/output tree when unset, so SunPy and Matplotlib work in sandboxed or nonstandard environments.

When the NF2 config does not hard-code an accelerator, the trainer now auto-selects CUDA when available, otherwise MPS on Apple Silicon, otherwise CPU.

## Files

- `nf2pino/bridge.py`
  Loads PINO exports, validates component order and units, and prepares NF2 samples.
- `nf2pino/nf2_init.py`
  Fits NF2's neural field to PINO samples and writes `pino_init.pt`.
- `nf2pino/history.py`
  Captures convergence snapshots at validation boundaries.
- `nf2pino/workflow.py`
  Owns path rewriting, run control, orchestration, and benchmark output.
- `run_hybrid.py`
  CLI entry point.
- `notebooks/pino_nf2_hybrid.ipynb`
  Researcher-facing workflow notebook built on the same execution path as the CLI.

## Config Profiles

Two NF2 profiles are provided:

- `config/pino_nf2_lightweight.yaml`
  Fast laptop-oriented profile for smoke tests and quick iteration. It keeps the NF2 model small and the refinement short.
- `config/pino_nf2_balanced.yaml`
  Research-oriented profile with a larger NF2 model, denser PINO warm-start sampling, and a stronger boundary schedule.

For first-frame convergence studies, prefer the balanced profile. Use the lightweight profile when you mainly want a quick end-to-end check.

## CLI Usage

Use an existing PINO export:

```bash
python run_hybrid.py \
  --nf2-config config/pino_nf2_balanced.yaml \
  --pino-field runs/pino_nf2_notebook/cache/pino_extrapolation.npz \
  --output-dir runs/hybrid_test
```

Run PINO first from a checkpoint and vector FITS components:

```bash
python run_hybrid.py \
  --nf2-config config/pino_nf2_balanced.yaml \
  --pino-model rtmag/best_model.pt \
  --bp path/to/Bp.fits \
  --bt path/to/Bt.fits \
  --br path/to/Br.fits \
  --output-dir runs/hybrid_test
```

Run a scratch baseline too:

```bash
python run_hybrid.py \
  --nf2-config config/pino_nf2_balanced.yaml \
  --pino-field runs/pino_nf2_notebook/cache/pino_extrapolation.npz \
  --output-dir runs/hybrid_test \
  --run-scratch
```

Resume existing workflow-owned NF2 runs:

```bash
python run_hybrid.py \
  --nf2-config config/pino_nf2_balanced.yaml \
  --pino-field runs/pino_nf2_notebook/cache/pino_extrapolation.npz \
  --output-dir runs/hybrid_test \
  --resume-nf2
```

Rebuild the cached PINO field:

```bash
python run_hybrid.py \
  --nf2-config config/pino_nf2_balanced.yaml \
  --pino-model rtmag/best_model.pt \
  --bp path/to/Bp.fits \
  --bt path/to/Bt.fits \
  --br path/to/Br.fits \
  --output-dir runs/hybrid_test \
  --force-pino
```

## NF2 Hook

Hybrid initialization is still opt-in under `training`:

```yaml
training:
  init_mode: pino
  init_from_pino: path/to/rtmag_extrapolation.npz
  pino_init:
    steps: 500
    batch_size: 8192
    max_samples: 200000
    stride: [8, 8, 8]
    lr: 5.0e-4
    strong_field_percentile: 90
    strong_field_weight: 5.0
    bottom_weight: 2.0
```

When `init_mode` and `init_from_pino` are absent, NF2 behavior is unchanged.

Avoid very small `--max-prefit-steps` values for scientific runs. In local validation, the 100-step smoke-test override reproduced the workflow but substantially underused the PINO warm start, while the default `8000`-step prefit preserved much more field strength into NF2 refinement.

## Warm-Start Details

The prefit stage now uses a weighted supervised objective:

- base MSE on all sampled points
- extra weight on strong-field samples
- extra weight on the bottom boundary

This improves the usefulness of the PINO warm start without changing NF2's downstream physics optimization.

`target_shape` is still supported but discouraged. Prefer stride-based downsampling to avoid smoothing kilogauss-scale peaks.

## Benchmark Outputs

`benchmark_summary.json` now includes:

- PINO path and PINO field metrics
- scratch NF2 runtime, history path, and final comparison metrics when requested
- PINO-initialized NF2 runtime, prefit summary path, history path, and final comparison metrics
- bottom-boundary mismatch on the shared PINO-grid bottom slice

For convenience, the summary exposes both workflow-owned run keys (`nf2_scratch`, `nf2_pino_init`) and friendlier aliases (`scratch`, `hybrid`).

Each workflow-owned NF2 run writes:

- `validation_history.json`
- `extrapolation_result.nf2`
- `pino_init.pt.json` for the hybrid run

## Notebook Behavior

The notebook is designed for local, reproducible experimentation:

- it prefers cached local PINO and SHARP artifacts
- it defaults to the balanced NF2 profile, with an easy switch back to the lightweight profile
- it leaves `max_prefit_steps` unset by default so the selected config profile controls the warm start
- JSOC download is optional, not required for the main path
- it runs one hybrid workflow call
- it reads back saved histories and results for comparison
- optional AIA / field-line overlay cells fail gracefully when optional packages or network access are unavailable

## Scientific Caveats

- PINO can accelerate NF2 convergence, but it can also bias the early optimization trajectory.
- Final quality should be judged by NF2 losses and physical metrics, not by closeness to PINO alone.
- Common-grid comparisons use PINO axes aligned to the NF2 FITS volume, which is the least invasive way to compare the two models consistently in this workflow.

## Current Scope

- Phase 1 targets single-frame Cartesian SHARP-style workflows.
- Time-series warm-starting is deferred until the single-frame path is stable.
- Existing NF2 conversion tools remain the downstream path for VTK/FITS export and related analysis.
