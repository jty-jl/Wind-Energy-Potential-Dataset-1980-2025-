#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script filters land data based on MODIS MCD12C1 Land Cover types.
It retains only specific IGBP classes suitable for wind energy development.

Methodology:
1. Load MCD12C1 Land Cover data.
2. Create a boolean mask for allowed IGBP classes.
3. Regrid the mask to match the target grid using Nearest Neighbor interpolation.
4. Apply the mask to 'sp' files (set non-allowed areas to NaN).

- Input: Land NetCDF files (from Step 2-LSM).
- Mask Source: MODIS MCD12C1 (Year 2024).
- Output: Filtered 'sp' NetCDF files.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Input Directory
IN_DIR = Path("data") / "sp" / "land"

# Land Cover Data Path
LC_PATH = Path("data") / "masks" / "MCD12C1_land_2024.nc"

# Output Directory
OUT_DIR = Path("data") / "sp" / "land" / "sp_MCD"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable Names
VAR_NAME = "sp"
LC_VAR = "Majority_Land_Cover_Type_1"

# Allowed IGBP Classes
ALLOWED_IGBP = {6, 7, 8, 9, 10, 12, 14, 16}

# Compression Settings
ENC = {
    VAR_NAME: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


def load_land_cover_mask(lc_path, var_name, allowed_classes):
    if not lc_path.exists():
        raise FileNotFoundError(f"Land Cover file not found: {lc_path}")

    ds = xr.open_dataset(lc_path)

    rename_dict = {}
    if "lat" in ds.coords: rename_dict["lat"] = "latitude"
    if "lon" in ds.coords: rename_dict["lon"] = "longitude"
    ds = ds.rename(rename_dict)

    if var_name not in ds:
        raise KeyError(f"Variable '{var_name}' not found in {lc_path.name}. "
                       f"Available: {list(ds.data_vars)}")

    lc_data = ds[var_name].squeeze()

    # Create Boolean Mask
    mask = xr.zeros_like(lc_data, dtype=bool)
    for k in allowed_classes:
        mask = mask | (lc_data == k)

    return mask


def align_mask_to_template(mask, template_ds):
    t_lat = template_ds["latitude"]
    t_lon = template_ds["longitude"]

    # 1. Handle Longitude Wrapping
    if float(mask.longitude.min()) < 0 and float(t_lon.min()) >= 0:
        print("[INFO] Converting Mask Longitude from [-180, 180] to [0, 360)...")
        mask = mask.assign_coords(longitude=(mask.longitude % 360)).sortby("longitude")

    # 2. Interpolation
    print("[INFO] Regridding Land Cover mask to ERA5 grid...")

    # Cast to int for safe interpolation
    mask_int = mask.astype("int16")

    mask_aligned = mask_int.interp(
        latitude=t_lat,
        longitude=t_lon,
        method="nearest"
    ).astype(bool)

    return mask_aligned


# ================= Main Execution =================

if __name__ == "__main__":

    # 1. Prepare File List
    nc_files = sorted(IN_DIR.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {IN_DIR}")

    print(f"Found {len(nc_files)} files to process.")

    # 2. Create Aligned Mask
    print(f"Loading Land Cover Mask from {LC_PATH.name}...")
    raw_mask = load_land_cover_mask(LC_PATH, LC_VAR, ALLOWED_IGBP)

    # Load first file to use as grid template
    with xr.open_dataset(nc_files[0]) as template:
        if "lat" in template.coords: template = template.rename({"lat": "latitude"})
        if "lon" in template.coords: template = template.rename({"lon": "longitude"})

        # Align mask
        final_mask = align_mask_to_template(raw_mask, template)

        # Load into memory to optimize speed
        final_mask.load()

    # Calculate stats
    valid_pct = (final_mask.sum() / final_mask.size) * 100
    print(f"[MASK] Buildable Land Coverage: {valid_pct:.2f}% of grid cells.")

    # 3. Batch Process
    for f in nc_files:
        out_name = f.name.replace(".nc", "_mcd.nc")
        out_path = OUT_DIR / out_name

        print(f"[Processing] {f.name}")

        ds = xr.open_dataset(f)
        try:
            # Rename coords if necessary
            rename_map = {}
            if "lat" in ds.coords: rename_map["lat"] = "latitude"
            if "lon" in ds.coords: rename_map["lon"] = "longitude"
            ds = ds.rename(rename_map)

            if VAR_NAME not in ds:
                print(f"  [SKIP] Variable {VAR_NAME} missing.")
                continue

            da = ds[VAR_NAME]

            # Apply Mask
            da_masked = da.where(final_mask)

            # Metadata
            da_masked = da_masked.astype("float32")
            da_masked.attrs = ds[VAR_NAME].attrs
            da_masked.attrs[
                "note"] = f"Filtered by MODIS MCD12C1 Land Cover (Year 2024). Allowed IGBP: {sorted(list(ALLOWED_IGBP))}"

            # Save
            ds_out = ds.copy()
            ds_out[VAR_NAME] = da_masked

            # Clean encoding attributes
            if VAR_NAME in ENC:
                for k in ["_FillValue", "scale_factor", "add_offset"]:
                    if k in ds_out[VAR_NAME].attrs: del ds_out[VAR_NAME].attrs[k]

            ds_out.to_netcdf(out_path, encoding=ENC)
            print(f"  -> Saved: {out_path.name}")

        finally:
            ds.close()

    print("Processing completed successfully.")