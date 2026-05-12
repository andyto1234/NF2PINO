import drms
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve

from astropy.time import Time
from astropy.io import fits
from sunpy.map import Map

JSOC_ROOT = "http://jsoc.stanford.edu"


def _sharp_cea_query(d, harpnum):
    """Run one DRMS query for SHARP CEA vector components + LOS magnetogram."""
    st = Time(d)

    yr = st.iso[:4]
    mo = st.iso[5:7]
    da = st.iso[8:10]
    hr = st.iso[11:13]
    mi = st.iso[14:16]

    c = drms.Client()
    hmi_query = f"hmi.sharp_cea_720s[{harpnum}][{yr}.{mo}.{da}_{hr}:{mi}:00_TAI]"
    return c.query(hmi_query, key=drms.JsocInfoConstants.all, seg="Bp, Bt, Br, magnetogram")


def _download_jsoc_segment(relative_url: str, out_dir: Path, overwrite: bool) -> Path:
    url = JSOC_ROOT + relative_url
    dest = out_dir / url.split("/")[-1]
    if not dest.exists() or overwrite:
        print(f"Downloading {dest.name} ...")
        urlretrieve(url, dest)
    return dest


def download_sharp_cea_vector_fits(d, harpnum, out_dir, overwrite=False):
    """Download SHARP CEA ``Bp``, ``Bt``, ``Br`` FITS files to disk (one JSOC query).

    Same series and cadence as :func:`get_sharp_map`. Use this when you need paths
    for NF2 or other file-based pipelines without building arrays here.

    Parameters
    ----------
    d : str
        Observation time string, same as ``get_sharp_map``.
    harpnum : int
        HARP number.
    out_dir : str or pathlib.Path
        Directory for the three FITS files (JSOC filenames preserved).
    overwrite : bool
        Re-download even if files already exist.

    Returns
    -------
    dict
        Keys ``"Bp"``, ``"Bt"``, ``"Br"`` mapping to ``Path`` objects, and ``"t_rec"``.

    Examples
    --------
    >>> paths = download_sharp_cea_vector_fits(d, harpnum, "sharp_cache")
    >>> nf2_yaml_placeholder_replacements = {k: str(paths[k]) for k in ("Bp", "Bt", "Br")}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hmi_keys, hmi_segments = _sharp_cea_query(d, harpnum)
    t_rec = str(hmi_keys.T_REC[0])
    print("T_REC: ", t_rec)

    paths = {"t_rec": t_rec}
    for comp in ("Bp", "Bt", "Br"):
        rel = getattr(hmi_segments, comp)[0]
        paths[comp] = _download_jsoc_segment(rel, out_dir, overwrite)
    return paths


def fetch_sharp_cea_bundle(d, harpnum, cache_dir, overwrite=False):
    """One JSOC query: cache ``Bp``/``Bt``/``Br`` + LOS magnetogram; build ``hmi_map`` / ``hmi_data``.

    Use this for hybrid PINO + NF2 workflows so NF2 can reuse the same FITS on disk
    without a separate ``nf2-download`` step. Arrays match :func:`get_sharp_map`.

    Parameters
    ----------
    d, harpnum
        Same as :func:`get_sharp_map`.
    cache_dir : str or pathlib.Path
        Directory to store the four FITS files (reused on subsequent runs if present).
    overwrite : bool
        Force re-download of cached FITS.

    Returns
    -------
    hmi_map : sunpy.map.Map
    hmi_data : ndarray
        Shape ``(nx, ny, 3)``, ``Bx = Bp``, ``By = -Bt``, ``Bz = Br``.
    fits_paths : dict
        ``{"Bp": Path, "Bt": Path, "Br": Path}`` for NF2 ``fits_path`` entries.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    hmi_keys, hmi_segments = _sharp_cea_query(d, harpnum)
    t_rec = str(hmi_keys.T_REC[0])
    print("T_REC: ", t_rec)
    mag_header = dict(hmi_keys.iloc[0])

    fits_paths = {}
    for comp in ("Bp", "Bt", "Br"):
        rel = getattr(hmi_segments, comp)[0]
        fits_paths[comp] = _download_jsoc_segment(rel, cache_dir, overwrite)

    mag_path = _download_jsoc_segment(hmi_segments.magnetogram[0], cache_dir, overwrite)
    with fits.open(mag_path) as mag_image:
        hmi_map = Map(mag_image[1].data, mag_header)

    with (
        fits.open(fits_paths["Bp"]) as hmi_Bp,
        fits.open(fits_paths["Bt"]) as hmi_Bt,
        fits.open(fits_paths["Br"]) as hmi_Br,
    ):
        hmi_data = np.stack([hmi_Bp[1].data, -hmi_Bt[1].data, hmi_Br[1].data]).T
    hmi_data = np.nan_to_num(hmi_data, nan=0.0).astype(np.float32)

    return hmi_map, hmi_data, fits_paths, t_rec


def get_sharp_map(d, harpnum):
    """Load HMI SHARP CEA vector field by streaming FITS from JSOC (nothing written to disk)."""
    hmi_keys, hmi_segments = _sharp_cea_query(d, harpnum)
    print("T_REC: ", hmi_keys.T_REC[0])

    hmi_Bp_url = JSOC_ROOT + hmi_segments.Bp[0]
    hmi_Bt_url = JSOC_ROOT + hmi_segments.Bt[0]
    hmi_Br_url = JSOC_ROOT + hmi_segments.Br[0]

    hmi_Br = fits.open(hmi_Br_url)
    hmi_Bt = fits.open(hmi_Bt_url)
    hmi_Bp = fits.open(hmi_Bp_url)

    mag_url = JSOC_ROOT + hmi_segments.magnetogram[0]
    mag_image = fits.open(mag_url)
    mag_header = dict(hmi_keys.iloc[0])
    hmi_map = Map(mag_image[1].data, mag_header)

    # Bx = Bp, By = -Bt, Bz = Br
    hmi_data = np.stack([hmi_Bp[1].data, -hmi_Bt[1].data, hmi_Br[1].data]).T
    hmi_data = np.nan_to_num(hmi_data, nan=0.0)
    hmi_data = hmi_data.astype(np.float32)

    return hmi_map, hmi_data


def get_aia_map(d, wavelength=171):
    st = Time(d)

    yr = st.iso[:4]
    mo = st.iso[5:7]
    da = st.iso[8:10]
    hr = st.iso[11:13]
    mi = st.iso[14:16]

    c = drms.Client()

    wavelength = str(wavelength)

    aia_query = f"aia.lev1_euv_12s[{yr}-{mo}-{da}T{hr}:00:00Z][{wavelength}]"
    aia_keys, aia_segments = c.query(aia_query, key=drms.JsocInfoConstants.all, seg="image")
    print("T_REC: ", aia_keys.T_REC[0])

    aia_url = JSOC_ROOT + aia_segments.image[0]
    aia_image = fits.open(aia_url)
    aia_header = dict(aia_keys.iloc[0])
    aia_image.verify("fix")
    aia_map = Map(aia_image[1].data, aia_header)

    return aia_map
