#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply GEBCO Mask to Surface Pressure (sp) - Land.
This script applies the pre-computed GEBCO buildable mask (Slope/Elevation criteria)
to the 'sp' data.
- Pixels marked as 1 (True) in the mask are RETAINED.
- Pixels marked as 0 (False) are set to NaN.

- Input: 'sp' NetCDF files.
- Mask: Boolean GEBCO mask (land_buildable_mask.nc).
- Output: Filtered 'sp' NetCDF files.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Input Directory
IN_DIR = Path("data") / "sp" / "land" / "sp_WDPA"

# Output Directory
OUT_DIR = Path("data") / "sp" / "land" / "sp_GEBCO"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# GEBCO Mask Path
MASK_NC = Path("data") / "masks" / "land_buildable_mask.nc"

# Variable Names
VAR_NAME = "sp"
LAT_NAME = "latitude"
LON_NAME = "longitude"
MASK_VAR_NAME = "__xarray_dataarray_variable__"

# Output Encoding
ENC = {
    VAR_NAME: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


def ensure_sorted_coords(ds, lat_name, lon_name):
    if not np.all(np.diff(ds[lon_name].values) > 0):
        ds = ds.sortby(lon_name)
    if not np.all(np.diff(ds[lat_name].values) < 0):
        ds = ds.sortby(lat_name, ascending=False)
    return ds


# ================= Main Execution =================

if __name__ == "__main__":

    # 1. Load GEBCO Mask
    if not MASK_NC.exists():
        raise FileNotFoundError(f"Mask file not found: {MASK_NC}")

    print(f"[MASK] Loading GEBCO mask: {MASK_NC.name}")
    ds_m = xr.open_dataset(MASK_NC)

    try:
        if MASK_VAR_NAME not in ds_m:
            print(f"[WARN] Variable '{MASK_VAR_NAME}' not found. Trying auto-detection...")
            MASK_VAR_NAME = list(ds_m.data_vars)[0]
            print(f"       -> Detected variable: {MASK_VAR_NAME}")

        m_lat = next((c for c in [LAT_NAME, "lat"] if c in ds_m.coords), LAT_NAME)
        m_lon = next((c for c in [LON_NAME, "lon"] if c in ds_m.coords), LON_NAME)

        mask2d = ds_m[MASK_VAR_NAME]

        # Ensure Boolean
        mask2d = mask2d.fillna(0)
        mask2d = (mask2d > 0.5)

        if m_lat != LAT_NAME: mask2d = mask2d.rename({m_lat: LAT_NAME})
        if m_lon != LON_NAME: mask2d = mask2d.rename({m_lon: LON_NAME})

        # Sort mask for consistent alignment
        mask2d = ensure_sorted_coords(mask2d.to_dataset(name="m"), LAT_NAME, LON_NAME)["m"]

        valid_pixels = int(mask2d.sum())
        total_pixels = mask2d.size
        print(f"[MASK] Loaded. Buildable pixels: {valid_pixels} ({valid_pixels / total_pixels:.2%})")

    finally:
        ds_m.close()

    # 2. Process Files
    nc_files = sorted([f for f in IN_DIR.glob("*.nc") if not f.name.startswith("._")])
    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in {IN_DIR}")

    print(f"Found {len(nc_files)} files to process.")

    for nc in nc_files:
        print(f"[Processing] {nc.name}")
        ds = xr.open_dataset(nc)

        try:
            if VAR_NAME not in ds:
                raise KeyError(f"Variable '{VAR_NAME}' missing in {nc.name}")

            # Ensure coordinates are sorted
            ds = ensure_sorted_coords(ds, LAT_NAME, LON_NAME)

            da = ds[VAR_NAME]

            # Align mask to current file grid
            mask_aligned = mask2d.reindex(
                {LAT_NAME: ds[LAT_NAME], LON_NAME: ds[LON_NAME]},
                method="nearest"
            )

            # Apply Mask
            da_masked = da.where(mask_aligned)

            # Update Metadata
            da_masked = da_masked.astype("float32")
            da_masked.attrs = ds[VAR_NAME].attrs
            da_masked.attrs["note"] = "Filtered by GEBCO mask (Slope/Elev)."

            # Clean encoding attributes
            for k in ["_FillValue", "scale_factor", "add_offset"]:
                if k in da_masked.attrs: del da_masked.attrs[k]

            # Save
            ds_out = ds.copy()
            ds_out[VAR_NAME] = da_masked

            # Output filename logic
            out_nc = OUT_DIR / nc.name.replace(".nc", "_gebco.nc")
            ds_out.to_netcdf(out_nc, encoding=ENC)
            print(f"  -> Saved: {out_nc.name}")

        finally:
            ds.close()

    print("Processing completed successfully.")