#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to apply a pre-calculated EEZ (Exclusive Economic Zone) mask to NetCDF files.
- Input: Ocean-only NetCDF files.
- Mask: A boolean numpy array (.npy) representing the EEZ coverage on the same grid.
- Output: NetCDF files with values outside the EEZ set to NaN.

Note: This script requires the 'eez_mask_bool_targetgrid.npy' file.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Target variable name
VAR_NAME = "sp"
TARGET_YEAR = "1980"

# Directories
IN_DIR = Path("data") / VAR_NAME / TARGET_YEAR / "ocean"
OUT_DIR = Path("data") / VAR_NAME / TARGET_YEAR / "ocean_eez"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_PATH = Path("data") / "masks" / "eez_mask_bool_targetgrid.npy"

# Coordinate names
LON_NAME = "longitude"
LAT_NAME = "latitude"

# Compression settings
ENC = {
    VAR_NAME: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


# =================================================

def list_nc_files(in_dir: Path):
    """Lists valid NetCDF files, skipping hidden or empty files."""
    files = []
    for p in sorted(in_dir.glob("*.nc")):
        if p.name.startswith("._"):
            continue
        if p.stat().st_size <= 1024:
            continue
        files.append(p)
    return files


def detect_grid_from_file(nc_path: Path):
    """
    Extracts grid coordinates from a sample file.
    Enforces a standard sort order: Longitude ascending, Latitude descending (North to South).
    """
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    try:
        if LON_NAME not in ds.coords or LAT_NAME not in ds.coords:
            raise KeyError(f"Missing coordinates {LON_NAME} or {LAT_NAME} in {nc_path.name}")

        # Enforce standard coordinate order to match the static mask
        if not np.all(np.diff(ds[LON_NAME].values) > 0):
            ds = ds.sortby(LON_NAME)
        if not np.all(np.diff(ds[LAT_NAME].values) < 0):
            ds = ds.sortby(LAT_NAME, ascending=False)

        return ds[LON_NAME].values, ds[LAT_NAME].values
    finally:
        ds.close()


def main():
    # 1. Load the pre-calculated mask
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Mask file not found at: {MASK_PATH}")

    print(f"Loading mask from {MASK_PATH}...")
    eez_mask = np.load(MASK_PATH)

    # 2. Locate input files
    nc_list = list_nc_files(IN_DIR)
    if not nc_list:
        raise FileNotFoundError(f"No valid NetCDF files found in {IN_DIR}")
    print(f"[{VAR_NAME}] Found {len(nc_list)} files to process.")

    # 3. Validate mask dimensions against the first data file
    lons, lats = detect_grid_from_file(nc_list[0])
    h_data, w_data = lats.size, lons.size

    if eez_mask.shape != (h_data, w_data):
        raise ValueError(
            f"Dimension mismatch! Mask shape {eez_mask.shape} does not match "
            f"data grid shape {(h_data, w_data)}.\n"
            f"Please ensure the mask was generated for the exact same grid."
        )

    # 4. Create an xarray DataArray for the mask
    eez_mask_da = xr.DataArray(
        eez_mask,
        coords={LAT_NAME: lats, LON_NAME: lons},
        dims=(LAT_NAME, LON_NAME)
    )

    # 5. Batch process files
    for nc in nc_list:
        ds = xr.open_dataset(nc, engine="netcdf4")
        try:
            if VAR_NAME not in ds.data_vars:
                print(f"[Warning] Variable '{VAR_NAME}' not found in {nc.name}, skipping.")
                continue

            if not np.all(np.diff(ds[LON_NAME].values) > 0):
                ds = ds.sortby(LON_NAME)
            if not np.all(np.diff(ds[LAT_NAME].values) < 0):
                ds = ds.sortby(LAT_NAME, ascending=False)

            # Apply Mask: Keep values where mask is True (Inside EEZ), else set to NaN
            da_filtered = ds[VAR_NAME].astype("float32").where(eez_mask_da)

            # Save processed data
            out_ds = ds.copy()
            out_ds[VAR_NAME] = da_filtered

            # Add processing note
            out_ds[VAR_NAME].attrs["note"] = "Filtered by Exclusive Economic Zone (EEZ) mask."

            out_nc = OUT_DIR / nc.name.replace(".nc", "_eez.nc")
            out_ds.to_netcdf(out_nc, encoding=ENC)

        except Exception as e:
            print(f"[Error] Failed to process {nc.name}: {e}")
        finally:
            ds.close()

    print("Processing completed successfully.")


if __name__ == "__main__":
    main()