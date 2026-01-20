#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts hourly Wind Power (Pwind) into Capacity Factor (CF).
CF is a dimensionless ratio defined as:
    CF = P_wind / P_rated
The result is clipped to the range [0, 1].

- Input: Hourly Pwind NetCDF files (Watts).
- Output: Hourly CF NetCDF files (Dimensionless, 0-1).
- Naming Convention: CF_hourly_{YYYY}_{MM}_offshore.nc
"""

from pathlib import Path
import numpy as np
import xarray as xr
import re

# ================= Configuration =================

# Input Directory
P_DIR = Path("data") / "Pwind" / "ocean"

# Output Directory
OUT_DIR = Path("data") / "CF" / "ocean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable Names
P_VAR = "Pwind"
CF_VAR = "CF"

# Fallback Turbine Parameters
FALLBACK_PARAMS = {
    "Cp": 0.45,
    "D": 200.0,  # Rotor Diameter (m)
    "U_r": 11.0,  # Rated Wind Speed (m/s)
    "rho_ref": 1.225  # Reference Air Density (kg/m^3)
}

# Calculate Fallback Rated Power
_A = np.pi * (FALLBACK_PARAMS["D"] ** 2) / 4.0
P_RATED_FALLBACK = 0.5 * FALLBACK_PARAMS["rho_ref"] * _A * FALLBACK_PARAMS["Cp"] * (FALLBACK_PARAMS["U_r"] ** 3)

# Output Compression Settings
ENC = {
    CF_VAR: {
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


def extract_date_from_filename(filename):
    match = re.search(r"(\d{4})[_]?(\d{2})", filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


# ================= Main Execution =================

if __name__ == "__main__":

    # List input files
    p_files = sorted(P_DIR.glob("*.nc"))
    if not p_files:
        raise FileNotFoundError(f"No Pwind files found in {P_DIR}")

    print(f"Found {len(p_files)} files. Starting CF calculation...")

    for p_nc in p_files:
        print(f"[Processing] {p_nc.name}")

        ds = open_ds_safe(p_nc)
        try:
            if P_VAR not in ds:
                raise KeyError(f"Variable '{P_VAR}' missing in {p_nc.name}")

            # 1. Load Data
            P = ds[P_VAR].astype("float64")

            # Safety
            P = xr.where(P < 0, 0, P)

            # 2. Determine Rated Power
            # Priority: Variable Attribute -> Global Attribute -> Fallback Calculation
            p_rated = None

            # Check variable attribute
            if "P_rated_W" in ds[P_VAR].attrs:
                val = ds[P_VAR].attrs["P_rated_W"]
                if val is not None and val > 0:
                    p_rated = float(val)

            # Check global attribute
            if p_rated is None and "P_rated_W" in ds.attrs:
                val = ds.attrs["P_rated_W"]
                if val is not None and val > 0:
                    p_rated = float(val)

            # Use Fallback
            if p_rated is None:
                p_rated = float(P_RATED_FALLBACK)
                print(f"  [WARN] P_rated_W metadata missing. Using fallback: {p_rated / 1e6:.2f} MW")

            # 3. Calculate Capacity Factor
            # CF = P / P_rated
            CF = (P / p_rated).clip(min=0.0, max=1.0).astype("float32")

            # 4. Metadata
            CF.name = CF_VAR
            CF.attrs = {
                "long_name": "Hourly Capacity Factor (Offshore)",
                "standard_name": "wind_capacity_factor",
                "units": "1",
                "valid_range": [0.0, 1.0],
                "p_rated_used": p_rated,
                "description": "Ratio of actual power output to rated power (Pwind / Prated)."
            }

            # 5. Save Output
            # Construct Filename: CF_hourly_YYYY_MM_offshore.nc
            year, month = extract_date_from_filename(p_nc.name)

            if year and month:
                out_name = f"CF_hourly_{year}_{month}_offshore.nc"
            else:
                print(f"[WARN] Could not parse date from {p_nc.name}. Using simple replacement.")
                out_name = p_nc.name.replace(P_VAR, "CF") + "_offshore.nc"

            out_nc = OUT_DIR / out_name

            ds_out = xr.Dataset({CF_VAR: CF}, coords=ds.coords)
            ds_out.attrs = ds.attrs

            ds_out.to_netcdf(out_nc, encoding=ENC)
            print(f"Saved: {out_name}")

        finally:
            ds.close()

    print("Processing completed successfully.")