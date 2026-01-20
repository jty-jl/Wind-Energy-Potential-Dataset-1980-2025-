#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script applies the pre-computed boolean WDPA mask to the 'sp' data.
- Pixels marked as True in the mask are set to NaN.
- Pixels marked as False are retained.

- Input: 'sp' NetCDF files.
- Mask: Boolean .npy mask.
- Output: Filtered 'sp' NetCDF files.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Input Directory
IN_DIR = Path("data") / "sp" / "land" / "sp_MCD"

# Output Directory
OUT_DIR = Path("data") / "sp" / "land" / "sp_WDPA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Path to the Boolean Mask (True = Protected Area)
MASK_NPY = Path("data") / "masks" / "WDPA_Processed" / "wdpa_land_mask_bool.npy"

# Variable and Coordinate Names
VAR_NAME = "sp"
LON_NAME = "longitude"
LAT_NAME = "latitude"

# Compression Settings
ENC = {
    VAR_NAME: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


# ================= Helper Functions =================

def ensure_sorted_xy(ds: xr.Dataset) -> xr.Dataset:
    if not np.all(np.diff(ds[LON_NAME].values) > 0):
        ds = ds.sortby(LON_NAME)
    if not np.all(np.diff(ds[LAT_NAME].values) < 0):
        ds = ds.sortby(LAT_NAME, ascending=False)
    return ds


def open_xr_safe(nc_path: Path) -> xr.Dataset:
    for eng in ("netcdf4", "h5netcdf", None):
        try:
            return xr.open_dataset(nc_path, engine=eng, mask_and_scale=False)
        except Exception:
            continue
    raise RuntimeError(f"Failed to open file: {nc_path.name}")


def apply_mask_to_nc(nc_path: Path, out_path: Path, mask_bool: np.ndarray):
    print(f"[Processing] {nc_path.name}")
    ds = open_xr_safe(nc_path)
    try:
        if VAR_NAME not in ds:
            raise KeyError(f"Variable '{VAR_NAME}' missing in {nc_path.name}")

        ds = ensure_sorted_xy(ds)

        da = ds[VAR_NAME]
        lats = ds[LAT_NAME].values
        lons = ds[LON_NAME].values

        if mask_bool.shape != (lats.size, lons.size):
            raise ValueError(f"Mask shape {mask_bool.shape} mismatch with data grid {(lats.size, lons.size)}")

        final_mask = mask_bool
        if lats[0] < lats[-1]:
            print("  [INFO] Flipping mask vertically to match data orientation.")
            final_mask = np.flipud(final_mask)

        # Construct Mask DataArray
        mask_da = xr.DataArray(
            final_mask,
            coords={LAT_NAME: lats, LON_NAME: lons},
            dims=(LAT_NAME, LON_NAME)
        )

        # Apply Mask
        masked_da = da.where(~mask_da)

        # Update Attributes
        masked_da = masked_da.astype("float32")
        masked_da.attrs = ds[VAR_NAME].attrs
        masked_da.attrs["note"] = "Filtered by WDPA (Protected Areas excluded)."

        # Clean conflicting encoding attributes
        for k in ["_FillValue", "scale_factor", "add_offset", "missing_value"]:
            if k in masked_da.attrs: del masked_da.attrs[k]

        # Save
        ds_out = ds.copy()
        ds_out[VAR_NAME] = masked_da
        ds_out.to_netcdf(out_path, encoding=ENC)
        print(f"  -> Saved: {out_path.name}")

    finally:
        ds.close()


# ================= Main Execution =================

if __name__ == "__main__":

    # 1. Load Mask
    if not MASK_NPY.exists():
        raise FileNotFoundError(f"Mask file not found: {MASK_NPY}")

    print(f"[MASK] Loading WDPA mask from: {MASK_NPY.name}")
    wdpa_mask = np.load(MASK_NPY)
    coverage = (wdpa_mask.sum() / wdpa_mask.size) * 100
    print(f"[MASK] Protected Area Coverage: {coverage:.2f}%")

    # 2. List Input Files
    nc_files = sorted([p for p in IN_DIR.glob("*.nc") if not p.name.startswith("._")])
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {IN_DIR}")

    print(f"Found {len(nc_files)} files to process.")

    # 3. Batch Process
    for nc in nc_files:
        out_name = nc.name.replace(".nc", "_wdpa.nc")
        out_nc = OUT_DIR / out_name

        apply_mask_to_nc(nc, out_nc, wdpa_mask)

    print("Processing completed successfully.")