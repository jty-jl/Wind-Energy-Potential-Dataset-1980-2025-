#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script removes areas covered by sea ice from the 'sp' (Surface Pressure) dataset.
1. Reads annual mean Sea Ice Concentration (SIC) data.
2. Aligns SIC data to the target ERA5 grid using nearest-neighbor interpolation.
3. Defines sea ice areas using a threshold (SIC > 0.7).
4. Masks out these areas (sets to NaN) in the 'sp' dataset.

- Input 1: 'sp' NetCDF files.
- Input 2: Sea Ice Concentration NetCDF (Annual Mean).
- Output: 'sp' NetCDF files with sea ice areas removed.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# 1. Paths
# Input directory: Output from Step 5b
IN_DIR = Path("data") / "sp" / "ocean" / "sp_GEBCO"

# Sea Ice Data Path
SEAICE_NC = Path("data") / "sea_ice" / "sea_ice_annual_mean_2024_regular025.nc"

# Output Directory
OUT_DIR = Path("data") / "sp" / "ocean" / "sp_noice"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Variable and Coordinate Names
VAR_NAME = "sp"
LAT_NAME = "latitude"
LON_NAME = "longitude"

ICE_VAR_NAME = "sea_ice_annual_mean"
ICE_LAT = "lat"
ICE_LON = "lon"

# 3. Compression Settings
ENC = {
    VAR_NAME: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


def to_lon360(lon):
    v = (lon % 360.0)
    v[v == 360.0] = 0.0
    return v


def build_ice_mask_on_target(ice_nc, lat_target, lon_target):
    ds = xr.open_dataset(ice_nc)
    try:
        if ICE_VAR_NAME not in ds.data_vars:
            raise KeyError(f"Variable '{ICE_VAR_NAME}' not found in {ice_nc.name}. "
                           f"Available: {list(ds.data_vars)}")

        sic = ds[ICE_VAR_NAME].astype("float32")

        vmax = float(sic.max(skipna=True).values)
        if vmax > 1.5:
            print("[INFO] Converting SIC from % to fraction (0-1)...")
            sic = sic / 100.0

        # Standardize Longitude to [0, 360) and sort
        lon360 = to_lon360(ds[ICE_LON].values.copy())
        sic = sic.assign_coords({ICE_LON: lon360}).sortby(ICE_LON)

        # Interpolate to target grid
        print("[Mask] Interpolating sea ice data to target grid...")
        sic_on_target = sic.interp(
            {ICE_LAT: lat_target, ICE_LON: lon_target},
            method="nearest"
        )

        # Define Mask: True where SIC > 0.7 (Ice Covered)
        mask = (sic_on_target > 0.7)

        # Rename coordinates to match target dataset conventions
        mask = mask.rename({ICE_LAT: LAT_NAME, ICE_LON: LON_NAME})
        mask.attrs.clear()
        return mask
    finally:
        ds.close()


def process_one(nc_path, ice_mask_on_target):
    print(f"[Processing] {nc_path.name}")
    ds = xr.open_dataset(nc_path)
    try:
        if VAR_NAME not in ds:
            raise KeyError(f"Variable '{VAR_NAME}' not found in {nc_path.name}")

        # Ensure coordinate sorting matches the mask
        if not np.all(np.diff(ds[LON_NAME].values) > 0):
            ds = ds.sortby(LON_NAME)
        if not np.all(np.diff(ds[LAT_NAME].values) < 0):
            ds = ds.sortby(LAT_NAME, ascending=False)

        da = ds[VAR_NAME].astype("float32")

        # Reindex mask to ensure exact alignment
        mask2d = ice_mask_on_target.reindex(
            {LAT_NAME: ds[LAT_NAME], LON_NAME: ds[LON_NAME]},
            method="nearest"
        )

        # Apply Mask
        da_noice = da.where(~mask2d)

        ds_out = ds.copy()
        ds_out[VAR_NAME] = da_noice

        # Update metadata
        ds_out[VAR_NAME].attrs["note"] = "Filtered by Sea Ice Concentration (> 0.7)."

        out_nc = OUT_DIR / nc_path.name.replace(".nc", "_noice.nc")
        ds_out.to_netcdf(out_nc, encoding=ENC)
        print(f"    -> Saved: {out_nc.name}")
    finally:
        ds.close()


# ================= Main Execution =================

if __name__ == "__main__":
    # 1. Retrieve Target Grid Definition from the first input file
    files = sorted(IN_DIR.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {IN_DIR}")

    with xr.open_dataset(files[0]) as sample:
        lat_target = sample[LAT_NAME].values
        lon_target = sample[LON_NAME].values
        print(f"[Grid] Target Lat: {lat_target[0]} to {lat_target[-1]} (n={lat_target.size})")
        print(f"[Grid] Target Lon: {lon_target[0]} to {lon_target[-1]} (n={lon_target.size})")

    # 2. Build Static Sea Ice Mask
    print("[Mask] Generating Sea Ice Mask (SIC > 0.7)...")
    if not SEAICE_NC.exists():
        raise FileNotFoundError(f"Sea Ice file not found: {SEAICE_NC}")

    ice_mask = build_ice_mask_on_target(SEAICE_NC, lat_target, lon_target)

    n_ice = int(ice_mask.sum().values)
    print(f"[Mask] Ice-covered pixels: {n_ice} / {ice_mask.size}")

    # 3. Batch Process 'sp' Files
    print(f"[Batch] Processing {len(files)} files...")
    for fp in files:
        process_one(fp, ice_mask)

    print("Processing completed successfully.")