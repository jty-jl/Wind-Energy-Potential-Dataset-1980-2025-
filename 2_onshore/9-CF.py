#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script converts hourly Wind Power (Pwind) into Capacity Factor (CF).
CF is a dimensionless ratio defined as:
    CF = P_wind / P_rated
The result is clipped to the range [0, 1].

- Input: Hourly Pwind NetCDF files (Watts).
- Output: Hourly CF NetCDF files (Dimensionless, 0-1).
- Naming Convention: CF_hourly_{YYYY}_{MM}_onshore.nc
"""

from pathlib import Path
import numpy as np
import xarray as xr
import re

# ================= Configuration =================

# Input Directory
P_DIR = Path("data") / "Pwind" / "land"

# Output Directory
OUT_DIR = Path("data") / "CF" / "land"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable Names
VAR_P = "Pwind"
VAR_CF = "CF"

# Fallback Parameters
CP = 0.45
D = 135.0
U_R = 11.0
RHO_REF = 1.225
AREA = np.pi * (D ** 2) / 4.0
P_RATED_FALLBACK = 0.5 * RHO_REF * AREA * CP * (U_R ** 3)

# Compression Settings
ENC = {
    VAR_CF: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}

# ================= Helper Functions =================

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

        try:
            ds = xr.open_dataset(p_nc, chunks="auto")
        except Exception as e:
            print(f"  [ERROR] Failed to open {p_nc.name}: {e}")
            continue

        try:
            if VAR_P not in ds:
                raise KeyError(f"Variable '{VAR_P}' missing in {p_nc.name}")

            # 1. Load Data
            P = ds[VAR_P].astype("float64")
            P = xr.where(P < 0, 0, P)

            # 2. Determine Rated Power
            p_rated = None

            # Check variable attribute
            try:
                val = P.attrs.get("P_rated_W")
                if val is not None and np.isfinite(val) and val > 0:
                    p_rated = float(val)
            except Exception:
                pass

            # Check global attribute
            if p_rated is None:
                try:
                    val = ds.attrs.get("P_rated_W")
                    if val is not None and np.isfinite(val) and val > 0:
                        p_rated = float(val)
                except Exception:
                    pass

            # Use Fallback
            if p_rated is None:
                p_rated = float(P_RATED_FALLBACK)
                print(f"  [WARN] P_rated_W missing. Using fallback: {p_rated / 1e6:.3f} MW")

            # 3. Calculate Capacity Factor
            CF = (P / p_rated).clip(min=0.0, max=1.0).astype("float32")

            # 4. Metadata
            CF.name = VAR_CF
            CF.attrs = {
                "long_name": "Hourly Capacity Factor (Onshore)",
                "standard_name": "wind_capacity_factor",
                "units": "1",
                "valid_range": [0.0, 1.0],
                "p_rated_used": p_rated,
                "description": "Ratio of actual power output to rated power (Pwind / Prated)."
            }

            # 5. Save Output
            # Construct Filename: CF_hourly_YYYY_MM_onshore.nc
            year, month = extract_date_from_filename(p_nc.name)

            if year and month:
                out_name = f"CF_hourly_{year}_{month}_onshore.nc"
            else:
                # Fallback naming if regex fails
                print(f"[WARN] Could not parse date from {p_nc.name}. Using simple replacement.")
                out_name = p_nc.name.replace(VAR_P, "CF") + "_onshore.nc"

            out_nc = OUT_DIR / out_name

            ds_out = xr.Dataset({VAR_CF: CF}, coords=ds.coords)

            ds_out.to_netcdf(out_nc, encoding=ENC)
            print(f"Saved: {out_name}")

        finally:
            ds.close()

    print("Processing completed successfully.")