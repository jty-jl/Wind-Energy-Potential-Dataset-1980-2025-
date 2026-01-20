#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to apply the WDPA mask to 'sp' (Surface Pressure) NetCDF files.

This script loads a pre-generated boolean mask (.npy) where True indicates
a Protected Area (WDPA). It applies this mask to the NetCDF data, setting
values within protected areas to NaN.
- Input: NetCDF files (sp variable).
- Mask: Binary WDPA mask (.npy) generated in Step 4a.
- Output: NetCDF files with Protected Areas excluded.
"""

import numpy as np
import xarray as xr
from pathlib import Path

# ================= Configuration =================

# Variable Name
VAR_NAME = "sp"
TARGET_YEAR = "1980"

# Directories
# Input: Output from the previous EEZ filtering step
IN_DIR = Path("data") / VAR_NAME / "ocean" / "sp_EEZ"
# Output: Final WDPA filtered data
OUT_DIR = Path("data") / VAR_NAME / "ocean" / "sp_WDPA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Path to the boolean mask generated in Step 4a
MASK_NPY = Path("data") / "masks" / "wdpa_ocean_mask_bool_targetgrid.npy"

# Coordinate names in the NetCDF files
LON_NAME = "longitude"
LAT_NAME = "latitude"

# Compression settings for output NetCDF
ENC = {
    VAR_NAME: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


# ================= Helper Functions =================

def detect_time_dim(da: xr.DataArray):
    """Detects the name of the time dimension in the DataArray."""
    for cand in ("valid_time", "time", "step"):
        if cand in da.dims:
            return cand
    return None


def ensure_sorted_xy(ds: xr.Dataset) -> xr.Dataset:
    if not np.all(np.diff(ds[LON_NAME].values) > 0):
        ds = ds.sortby(LON_NAME)
    if not np.all(np.diff(ds[LAT_NAME].values) < 0):
        ds = ds.sortby(LAT_NAME, ascending=False)
    return ds


def open_xr_try(nc_path: Path) -> xr.Dataset:
    for eng in ("netcdf4", "h5netcdf"):
        try:
            return xr.open_dataset(nc_path, engine=eng, mask_and_scale=False)
        except Exception:
            continue
    raise RuntimeError(f"Failed to open {nc_path.name} with netcdf4 or h5netcdf.")


def list_nc_files(in_dir: Path):
    files = []
    for p in sorted(in_dir.glob("*.nc")):
        if p.name.startswith("._"):
            continue
        try:
            if p.stat().st_size <= 1024:
                continue
        except Exception:
            continue
        files.append(p)
    return files


def apply_mask_to_nc(nc_path: Path, out_path: Path, mask_bool: np.ndarray):
    """
    Applies the 2D WDPA mask to the NetCDF variable.
    Logic: True in mask (Protected Area) -> Set to NaN.
    """
    print(f"[PROC] Processing: {nc_path.name}")
    ds = open_xr_try(nc_path)
    try:
        if VAR_NAME not in ds:
            raise KeyError(f"Variable '{VAR_NAME}' not found in {nc_path.name}")
        if LON_NAME not in ds.coords or LAT_NAME not in ds.coords:
            raise KeyError(f"Missing coordinates {LON_NAME}/{LAT_NAME} in {nc_path.name}")

        # Ensure grid alignment with the static mask
        ds = ensure_sorted_xy(ds)
        da = ds[VAR_NAME]
        tdim = detect_time_dim(da)

        lons = ds[LON_NAME].values
        lats = ds[LAT_NAME].values

        # Validate dimensions
        if mask_bool.shape != (lats.size, lons.size):
            raise ValueError(f"Mask shape {mask_bool.shape} mismatch with data grid {(lats.size, lons.size)}")

        # Create DataArray for broadcasting
        mask2d = xr.DataArray(
            mask_bool,
            coords={LAT_NAME: lats, LON_NAME: lons},
            dims=(LAT_NAME, LON_NAME)
        )

        # Apply Mask:
        # mask2d is True for Protected Areas.
        # we exclude them by keeping data where mask is False (~mask2d).
        if tdim is None:
            masked = da.where(~mask2d)
        else:
            masked = da.where(~mask2d.broadcast_like(da))

        masked = masked.astype("float32")
        for k in ("_FillValue", "scale_factor", "add_offset", "missing_value"):
            if k in masked.attrs:
                del masked.attrs[k]

        ds_out = ds.copy()
        ds_out[VAR_NAME] = masked

        ds_out[VAR_NAME].attrs["note"] = "Excluded World Database on Protected Areas (WDPA)."

        ds_out.to_netcdf(out_path, encoding=ENC)
        print(f"[OK] Saved: {out_path.name}")
    finally:
        ds.close()


# ================= Main Execution =================

if __name__ == "__main__":
    # 1. Load Mask
    if not MASK_NPY.exists():
        raise FileNotFoundError(f"Mask file not found: {MASK_NPY}\nPlease run Step 4a to generate it.")

    mask = np.load(MASK_NPY)
    print(f"[MASK] Loaded WDPA mask. (Protected pixels: {mask.sum()}/{mask.size})")

    # 2. Find Files
    nc_list = list_nc_files(IN_DIR)
    if not nc_list:
        raise FileNotFoundError(f"No valid .nc files found in {IN_DIR}")
    print(f"[INFO] Found {len(nc_list)} files to process.")

    # 3. Batch Process
    for nc in nc_list:
        out_nc = OUT_DIR / nc.name.replace(".nc", "_wdpa.nc")
        apply_mask_to_nc(nc, out_nc, mask)
    print("Processing completed successfully.")