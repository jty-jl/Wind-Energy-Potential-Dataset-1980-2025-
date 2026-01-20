#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script calculates the hourly near-surface air density from sp
and t2m using the Ideal Gas Law for dry air.

Formula: rho = sp / (R_d * t2m)
where R_d = 287.05 J/(kg*K) is the specific gas constant for dry air.

- Input 1: 2m Temperature NetCDF files (Kelvin).
- Input 2: Surface Pressure NetCDF files (Pascal).
- Output: Air Density NetCDF files (kg m-3).
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Input Directories
T2M_DIR = Path("data") / "t2m" / "land" / "t2m_GEBCO_land"
SP_DIR  = Path("data") / "sp" / "land" / "sp_GEBCO_land"

# Output Directory
OUT_DIR = Path("data") / "rho" / "land"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable Names
VAR_T2M = "t2m"  # Input Temperature (K)
VAR_SP  = "sp"   # Input Pressure (Pa)
VAR_OUT = "rho"  # Output Density

# Constants
RD = 287.05  # Specific gas constant for dry air, J/(kg*K)

# Compression Settings
ENC = {
    VAR_OUT: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}

# ================= Main Execution =================

if __name__ == "__main__":

    # 1. List Input Files
    t2m_files = sorted(T2M_DIR.glob("*.nc"))
    if not t2m_files:
        raise FileNotFoundError(f"No NetCDF files found in {T2M_DIR}")

    print(f"Found {len(t2m_files)} files. Starting calculation...")

    for f_t2m in t2m_files:
        f_sp = SP_DIR / f_t2m.name.replace(VAR_T2M, VAR_SP)

        if not f_sp.exists():
            print(f"[WARN] Matching SP file not found: {f_sp.name}. Skipping.")
            continue

        print(f"[Processing] {f_t2m.name} + {f_sp.name}")

        ds_t = xr.open_dataset(f_t2m, chunks="auto")
        ds_p = xr.open_dataset(f_sp, chunks="auto")

        try:
            # 2. Validation
            if VAR_T2M not in ds_t:
                raise KeyError(f"Variable '{VAR_T2M}' missing in {f_t2m.name}")
            if VAR_SP not in ds_p:
                raise KeyError(f"Variable '{VAR_SP}' missing in {f_sp.name}")

            # 3. Alignment
            ds_p = ds_p.reindex_like(ds_t, method=None)

            # 4. Calculation
            # rho = P / (R * T)
            T = ds_t[VAR_T2M].astype("float64")
            P = ds_p[VAR_SP].astype("float64")
            rho = P / (RD * T)

            # 5. Metadata
            rho = rho.astype("float32")
            rho.name = VAR_OUT
            rho.attrs = {
                "long_name": "Near-surface air density",
                "standard_name": "air_density",
                "units": "kg m-3",
                "description": "Calculated using Ideal Gas Law (dry air approximation)."
            }

            # 6. Save Output
            ds_out = xr.Dataset({VAR_OUT: rho}, coords=ds_t.coords)
            ds_out.attrs = ds_t.attrs
            ds_out.attrs["comment"] = "Derived air density for land area."

            # Output filename
            out_name = f_t2m.name.replace(VAR_T2M, VAR_OUT)
            out_path = OUT_DIR / out_name

            ds_out.to_netcdf(out_path, encoding=ENC)
            print(f"    -> Saved: {out_path.name}")

        finally:
            ds_t.close()
            ds_p.close()

    print("Processing completed successfully.")