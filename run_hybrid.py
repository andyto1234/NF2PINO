#!/usr/bin/env python
"""Run a PINO -> NF2 hybrid extrapolation workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nf2pino.paths import configure_runtime_env
from nf2pino.workflow import load_json_or_yaml_config, run_hybrid_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PINO initialization followed by NF2 refinement.")
    parser.add_argument("--nf2-config", required=True, help="NF2 YAML/JSON config for the refinement run.")
    parser.add_argument("--output-dir", default="runs/pino_nf2_hybrid", help="Directory for caches and outputs.")
    parser.add_argument("--pino-field", help="Existing RTMAG/PINO `rtmag_extrapolation.npz` or `.npy` cube.")
    parser.add_argument("--pino-model", help="PINO checkpoint path, e.g. rtmag best_model.pt.")
    parser.add_argument("--magnetogram", help="Vector magnetogram `.npz`/`.npy` input for PINO.")
    parser.add_argument("--bp", help="FITS Bp component path (PINO file mode and/or NF2 {%Bp} placeholder).")
    parser.add_argument("--bt", help="FITS Bt component path.")
    parser.add_argument("--br", help="FITS Br component path.")
    parser.add_argument(
        "--nf2-bp",
        help="NF2 boundary Bp FITS (defaults to --bp if placeholders are used).",
    )
    parser.add_argument("--nf2-bt", help="NF2 boundary Bt FITS (defaults to --bt).")
    parser.add_argument("--nf2-br", help="NF2 boundary Br FITS (defaults to --br).")
    parser.add_argument(
        "--sharp-d",
        help="SHARP observation time (same string as rtmag main.ipynb `d`). Downloads Bp/Bt/Br into --output-dir/cache/sharp_jsoc_fits when NF2 placeholders or PINO inputs are missing.",
    )
    parser.add_argument("--sharp-harpnum", type=int, help="SHARP HARP number (use with --sharp-d).")
    parser.add_argument(
        "--sharp-overwrite-fits",
        action="store_true",
        help="Re-download SHARP FITS even if the cache directory already has them.",
    )
    parser.add_argument("--resolution", default="512,256", help="PINO input resolution as `nx,ny`.")
    parser.add_argument(
        "--max-prefit-steps",
        type=int,
        help="Override PINO-to-NF2 prefit steps. Mainly useful for smoke tests; use the config default for science runs.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--run-scratch", action="store_true", help="Also run a random-initialized NF2 baseline.")
    parser.add_argument(
        "--resume-nf2",
        action="store_true",
        help="Resume existing workflow-owned NF2 run directories instead of starting fresh.",
    )
    parser.add_argument(
        "--force-pino",
        action="store_true",
        help="Recompute the cached PINO extrapolation even if --output-dir/cache already has one.",
    )
    args = parser.parse_args()

    configure_runtime_env(Path(args.output_dir) / "cache")
    nx, ny = _parse_resolution(args.resolution)
    text = Path(args.nf2_config).read_text()
    has_ph = "{%" in text
    has_manual_fits = (args.bp and args.bt and args.br) or (
        args.nf2_bp and args.nf2_bt and args.nf2_br
    )
    has_sharp = args.sharp_d is not None and args.sharp_harpnum is not None
    if has_ph and not has_manual_fits and not has_sharp:
        raise SystemExit(
            "NF2 config still has {%...} tokens. Pass --bp/--bt/--br or --nf2-bp/--nf2-bt/--nf2-br, "
            "or --sharp-d and --sharp-harpnum to cache SHARP vector FITS, or edit the YAML with absolute paths."
        )
    replacements = None
    if has_ph and has_manual_fits:
        fb, ft, fr = args.nf2_bp or args.bp, args.nf2_bt or args.bt, args.nf2_br or args.br
        replacements = {
            "Bp": str(Path(fb).expanduser().resolve()),
            "Bt": str(Path(ft).expanduser().resolve()),
            "Br": str(Path(fr).expanduser().resolve()),
        }
    if (args.sharp_d is None) ^ (args.sharp_harpnum is None):
        raise SystemExit("Provide both --sharp-d and --sharp-harpnum, or omit both.")
    config = load_json_or_yaml_config(args.nf2_config, replacements=replacements)
    benchmark = run_hybrid_workflow(
        nf2_config=config,
        output_dir=args.output_dir,
        pino_field=args.pino_field,
        pino_model=args.pino_model,
        magnetogram=args.magnetogram,
        bp=args.bp,
        bt=args.bt,
        br=args.br,
        nf2_fits_bp=args.nf2_bp or args.bp,
        nf2_fits_bt=args.nf2_bt or args.bt,
        nf2_fits_br=args.nf2_br or args.br,
        nx=nx,
        ny=ny,
        device=args.device,
        run_scratch=args.run_scratch,
        prefit_steps=args.max_prefit_steps,
        resume_nf2=args.resume_nf2,
        force_pino=args.force_pino,
        sharp_jsoc_time=args.sharp_d,
        sharp_jsoc_harpnum=args.sharp_harpnum,
        sharp_jsoc_overwrite_fits=args.sharp_overwrite_fits,
    )
    print(json.dumps(benchmark, indent=2))


def _parse_resolution(value: str) -> tuple[int, int]:
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--resolution must be formatted as nx,ny")
    return parts[0], parts[1]


if __name__ == "__main__":
    main()
