#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script calculates the 100m scalar wind speed (V100) from the zonal (u100)
and meridional (v100) wind components.
Formula: V100 = sqrt(u100^2 + v100^2)

- Input 1: u100 NetCDF files (Processed: Slope/Elev filtered).
- Input 2: v100 NetCDF files (Processed identically to u100).
- Output: V100 NetCDF files.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Input Directories
U_DIR = Path("data") / "u100" / "land" / "u100_GEBCO"
V_DIR = Path("data") / "v100" / "land" / "v100_GEBCO"

# Output Directory
OUT_DIR = Path("data") / "wind100" / "land" / "wind100"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable Names
U_VAR = "u100"
V_VAR = "v100"
OUT_VAR = "V100"

# Compression Settings
ENC = {
    OUT_VAR: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


def open_ds_safe(path):
    for eng in ("netcdf4", "h5netcdf", None):
        try:
            return xr.open_dataset(path, engine=eng, mask_and_scale=False, chunks="auto")
        except Exception:
            continue
    raise RuntimeError(f"Failed to open file: {path.name}")


# ================= Main Execution =================

if __name__ == "__main__":

    # 1. List files
    u_files = sorted(U_DIR.glob("*.nc"))
    if not u_files:
        raise FileNotFoundError(f"No NetCDF files found in {U_DIR}")

    print(f"Found {len(u_files)} 'u' files. Starting processing...")

    for uf in u_files:
        # Construct corresponding 'v' filename
        vf_name = uf.name.replace(U_VAR, V_VAR)
        vf = V_DIR / vf_name

        if not vf.exists():
            print(f"[WARN] Matching 'v' file not found for {uf.name}. Skipping.")
            continue

        print(f"[Processing] {uf.name} + {vf.name}")

        ds_u = open_ds_safe(uf)
        ds_v = open_ds_safe(vf)

        try:
            if U_VAR not in ds_u:
                raise KeyError(f"Variable '{U_VAR}' missing in {uf.name}")
            if V_VAR not in ds_v:
                raise KeyError(f"Variable '{V_VAR}' missing in {vf.name}")

            # Align coordinates
            ds_v = ds_v.reindex_like(ds_u, method=None)

            # Calculation: V = sqrt(u^2 + v^2)
            u_data = ds_u[U_VAR].astype("float64")
            v_data = ds_v[V_VAR].astype("float64")

            wind_speed = np.sqrt(u_data ** 2 + v_data ** 2)

            # Metadata Update
            wind_speed = wind_speed.astype("float32")
            wind_speed.name = OUT_VAR
            wind_speed.attrs = {
                "long_name": "100 metre wind speed",
                "standard_name": "wind_speed",
                "units": "m s-1",
                "description": "Derived scalar wind speed from u and v components."
            }

            # Create Output Dataset
            ds_out = xr.Dataset(
                {OUT_VAR: wind_speed},
                coords=ds_u.coords
            )
            # Inherit global attributes
            ds_out.attrs = ds_u.attrs
            ds_out.attrs["comment"] = "Wind speed calculated after land suitability filtering."

            # Save
            out_file = OUT_DIR / uf.name.replace(U_VAR, OUT_VAR)
            ds_out.to_netcdf(out_file, encoding=ENC)
            print(f"    -> Saved: {out_file.name}")

        finally:
            ds_u.close()
            ds_v.close()

    print("Processing completed successfully.")