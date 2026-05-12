#!/usr/bin/env python3
"""Smoke-test hybrid NF2 on PINO axes: Mm spans, ds query box, max |B|, comparison metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pino",
        type=Path,
        default=REPO / "runs" / "pino_nf2_notebook" / "cache" / "pino_extrapolation.npz",
        help="PINO export NPZ",
    )
    parser.add_argument(
        "--nf2",
        type=Path,
        default=REPO / "runs" / "pino_nf2_notebook" / "nf2_pino_init" / "extrapolation_result.nf2",
        help="NF2 extrapolation_result.nf2",
    )
    parser.add_argument(
        "--no-align-domain",
        action="store_true",
        help="Pass align_domain=False (legacy: PINO Mm already matches NF2 FITS box).",
    )
    parser.add_argument("--device", default="cpu", help="torch device for NF2 eval")
    parser.add_argument("--progress", action="store_true", help="tqdm in NF2 loader")
    args = parser.parse_args()

    runtime_root = args.nf2.parent.parent / "cache"
    os.environ.setdefault("SUNPY_CONFIGDIR", str(runtime_root / "runtime_env" / "sunpy"))
    os.environ.setdefault("MPLCONFIGDIR", str(runtime_root / "runtime_env" / "matplotlib"))
    Path(os.environ["SUNPY_CONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    sys.path[:0] = [str(REPO), str(REPO / "NF2"), str(REPO / "rtmag")]

    from nf2pino.bridge import load_pino_field
    from nf2pino.compare import compare_b_on_same_grid, evaluate_nf2_on_pino_axes

    if not args.pino.is_file():
        print(f"Missing PINO file: {args.pino}", file=sys.stderr)
        return 2
    if not args.nf2.is_file():
        print(f"Missing NF2 file: {args.nf2}", file=sys.stderr)
        return 2

    field = load_pino_field(args.pino)
    xm = np.asarray(field.x_coords, dtype=np.float64)
    ym = np.asarray(field.y_coords, dtype=np.float64)
    zm = np.asarray(field.z_coords, dtype=np.float64)
    print("PINO Mm ranges:")
    print(f"  x: [{xm.min():.4f}, {xm.max():.4f}]  span={xm.max()-xm.min():.4f}")
    print(f"  y: [{ym.min():.4f}, {ym.max():.4f}]  span={ym.max()-ym.min():.4f}")
    print(f"  z: [{zm.min():.4f}, {zm.max():.4f}]  span={zm.max()-zm.min():.4f}")
    print(f"PINO max|B| = {float(np.max(np.abs(field.b))):.4f} G")

    align = not args.no_align_domain
    b_nf2, out_h = evaluate_nf2_on_pino_axes(
        args.nf2,
        x_m=xm,
        y_m=ym,
        z_m=zm,
        device=args.device,
        progress=args.progress,
        compute_jacobian=False,
        align_domain=align,
    )
    mmds = float(out_h.Mm_per_ds)
    xr, yr = out_h.coord_range[0], out_h.coord_range[1]
    print(f"\nNF2 FITS coord_range (ds): x {xr} y {yr}  max_height={out_h.max_height} Mm")
    print(f"Mm_per_ds = {mmds:.6f}")
    print(f"evaluate_nf2_on_pino_axes align_domain={align}")

    # Reproduce affine map for diagnostics
    if align:
        x_lo, x_hi = float(xr[0] * mmds), float(xr[1] * mmds)
        y_lo, y_hi = float(yr[0] * mmds), float(yr[1] * mmds)
        z_lo, z_hi = 0.0, float(out_h.max_height)

        def _map(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
            mn, mx = float(a.min()), float(a.max())
            if mx <= mn:
                return np.full_like(a, 0.5 * (lo + hi), dtype=np.float64)
            s = (a - mn) / (mx - mn)
            return lo + s * (hi - lo)

        xd = _map(xm, x_lo, x_hi) / mmds
        yd = _map(ym, y_lo, y_hi) / mmds
        zd = _map(zm, z_lo, z_hi) / mmds
    else:
        xd = xm / mmds
        yd = ym / mmds
        zd = zm / mmds
    print("Query coords in ds (after alignment if any):")
    print(
        f"  x_ds: [{xd.min():.4f}, {xd.max():.4f}]  y_ds: [{yd.min():.4f}, {yd.max():.4f}]  z_ds: [{zd.min():.4f}, {zd.max():.4f}]"
    )

    print(f"NF2 on PINO grid: max|B| = {float(np.max(np.abs(b_nf2))):.4f} G  mean|B| = {float(np.mean(np.abs(b_nf2))):.4f} G")

    ratio = float(np.max(np.abs(b_nf2))) / (float(np.max(np.abs(field.b))) + 1e-9)
    if ratio < 0.05:
        print(
            "\nNote: NF2 |B| is still much smaller than PINO. If you have not re-run the hybrid "
            "since enabling FITS-domain-aligned PINO prefit, delete nf2_pino_init (or bump base_path) "
            "and run training again so `prepare_nf2_init_samples` maps PINO Mm to the NF2 box.",
            file=sys.stderr,
        )

    cmp_ = compare_b_on_same_grid(field.b, b_nf2, field.spacing_Mm)
    print("\ncompare_b_on_same_grid:")
    print(json.dumps(cmp_, indent=2, default=float))

    # Heuristic: tiny max|B| with align=False usually means out-of-box queries
    if float(np.max(np.abs(b_nf2))) < 0.05 * float(np.max(np.abs(field.b))) and not align:
        print(
            "\nHint: NF2 |B| is much smaller than PINO. Try without --no-align-domain "
            "if PINO Mm span ≠ NF2 FITS span.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
