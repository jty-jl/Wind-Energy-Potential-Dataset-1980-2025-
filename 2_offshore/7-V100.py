#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script calculates the 100m scalar wind speed (V100) from the zonal (u100)
and meridional (v100) wind components.
Formula: V100 = sqrt(u100^2 + v100^2)

- Input 1: u100 NetCDF files (Processed: No Ice).
- Input 2: v100 NetCDF files (Processed identically to u100).
- Output: V100 NetCDF files.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Input Directories
# Assuming directory structure: data/u100/ocean/... and data/v100/ocean/...
U_DIR = Path("data") / "u100" / "ocean" / "u100_GEBCO_ocean_noice"
V_DIR = Path("data") / "v100" / "ocean" / "v100_GEBCO_ocean_noice"

# Output Directory
OUT_DIR = Path("data") / "wind100" / "ocean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable Names
U_VAR = "u100"
V_VAR = "v100"
OUT_VAR = "V100"

# Candidate names for time dimension
TIME_CANDIDATES = ["valid_time", "time", "step"]

# Compression settings
ENC = {
    OUT_VAR: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


def detect_time_dim(ds):
    for t in TIME_CANDIDATES:
        if t in ds.dims:
            return t
    raise RuntimeError("Time dimension (time/valid_time/step) not found in dataset.")


# ================= Main Execution =================

if __name__ == "__main__":

    # 1. List files
    u_files = sorted(U_DIR.glob("*.nc"))
    v_files = sorted(V_DIR.glob("*.nc"))

    if not u_files:
        raise FileNotFoundError(f"No NetCDF files found in {U_DIR}")

    for uf in u_files:
        vf_name = uf.name.replace(U_VAR, V_VAR)
        vf = V_DIR / vf_name

        if not vf.exists():
            print(f"[WARN] Matching 'v' file not found for {uf.name}. Skipping.")
            continue

        print(f"[Processing] {uf.name} + {vf.name}")

        ds_u = xr.open_dataset(uf)
        ds_v = xr.open_dataset(vf)

        try:
            if U_VAR not in ds_u:
                raise KeyError(f"Variable '{U_VAR}' missing in {uf.name}")
            if V_VAR not in ds_v:
                raise KeyError(f"Variable '{V_VAR}' missing in {vf.name}")

            # Align coordinates
            ds_v = ds_v.reindex_like(ds_u, method=None)

            # Calculate Wind Speed
            # Formula: sqrt(u^2 + v^2)
            wind_speed = np.sqrt(ds_u[U_VAR] ** 2 + ds_v[V_VAR] ** 2)

            # Set Metadata
            wind_speed.name = OUT_VAR
            wind_speed = wind_speed.astype("float32")
            wind_speed.attrs["long_name"] = "100 metre wind speed"
            wind_speed.attrs["units"] = "m s-1"
            wind_speed.attrs["standard_name"] = "wind_speed"

            # Create Output Dataset
            ds_out = xr.Dataset(
                {OUT_VAR: wind_speed},
                coords=ds_u.coords
            )

            # Copy global attributes from U file, modify description
            ds_out.attrs = ds_u.attrs
            ds_out.attrs["description"] = "Derived 100m wind speed (sqrt(u^2+v^2))."

            # Construct output path
            out_file = OUT_DIR / uf.name.replace(U_VAR, OUT_VAR)

            # Save
            ds_out.to_netcdf(out_file, encoding=ENC)
            print(f"    -> Saved: {out_file.name}")

        finally:
            ds_u.close()
            ds_v.close()

    print("Processing completed successfully.")