#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script calculates the hourly air density (rho) from Surface Pressure (sp)
and 2-metre Temperature (t2m) using the Ideal Gas Law for dry air.
Formula: rho = sp / (Rd * t2m)

- Input 1: t2m NetCDF files (Must be processed: GEBCO + Sea Ice filtered).
- Input 2: sp NetCDF files (Must be processed: GEBCO + Sea Ice filtered).
- Output: Air Density (rho) NetCDF files.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Input Directories
T2M_DIR = Path("data") / "t2m" / "ocean" / "t2m_noice"
SP_DIR = Path("data") / "sp" / "ocean" / "sp_noice"

# Output Directory
OUT_DIR = Path("data") / "rho" / "ocean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable Names
T2M_VAR = "t2m"  # Temperature at 2m (K)
SP_VAR = "sp"  # Surface Pressure (Pa)
OUT_VAR = "rho"  # Air Density (kg m-3)

# Constants
RD = 287.05  # Specific gas constant for dry air, J/(kg*K)

# Time dimension candidates
TIME_CANDIDATES = ("valid_time", "time", "step")

# Compression settings
ENC = {
    OUT_VAR: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


def detect_time_dim(ds):
    for nm in TIME_CANDIDATES:
        if nm in ds.dims:
            return nm
    raise RuntimeError(f"Time dimension not found. Candidates: {TIME_CANDIDATES}")


def open_ds_safe(path):
    for eng in ("netcdf4", "h5netcdf", None):
        try:
            return xr.open_dataset(path, engine=eng, mask_and_scale=False)
        except Exception:
            continue
    raise RuntimeError(f"Failed to open file: {path.name}")


# ================= Main Execution =================

if __name__ == "__main__":

    # 1. List t2m files
    t2m_files = sorted(T2M_DIR.glob("*.nc"))
    if not t2m_files:
        raise FileNotFoundError(f"No NetCDF files found in {T2M_DIR}")

    print(f"Found {len(t2m_files)} t2m files. Starting processing...")

    for t2m_nc in t2m_files:
        # 2. Find corresponding sp file
        # Assumption: Filenames match exactly except for the variable name
        # e.g., "era5_t2m_2025_01_noice.nc" -> "era5_sp_2025_01_noice.nc"
        sp_name = t2m_nc.name.replace(T2M_VAR, SP_VAR)
        sp_nc = SP_DIR / sp_name

        if not sp_nc.exists():
            print(f"[WARN] Matching sp file not found: {sp_nc.name}. Skipping.")
            continue

        print(f"[Processing] {t2m_nc.name} + {sp_nc.name}")

        ds_t = open_ds_safe(t2m_nc)
        ds_p = open_ds_safe(sp_nc)

        try:
            # 3. Validation
            if T2M_VAR not in ds_t:
                raise KeyError(f"Variable '{T2M_VAR}' missing in {t2m_nc.name}")
            if SP_VAR not in ds_p:
                raise KeyError(f"Variable '{SP_VAR}' missing in {sp_nc.name}")

            # 4. Align coordinates
            ds_p = ds_p.reindex_like(ds_t, method=None)

            # 5. Calculate Air Density
            # rho = P / (R * T)
            T = ds_t[T2M_VAR].astype("float64")
            P = ds_p[SP_VAR].astype("float64")

            rho = (P / (RD * T)).astype("float32")

            # 6. Metadata
            rho.name = OUT_VAR
            rho.attrs = {
                "long_name": "Near-surface air density",
                "standard_name": "air_density",
                "units": "kg m-3",
                "note": "Calculated using Ideal Gas Law (dry air approximation): rho = sp / (Rd * t2m)"
            }

            # 7. Create Output Dataset
            ds_out = xr.Dataset(
                {OUT_VAR: rho},
                coords=ds_t.coords
            )
            # Inherit global attributes
            ds_out.attrs = ds_t.attrs
            ds_out.attrs["description"] = "Hourly air density derived from ERA5 t2m & sp (Sea Ice excluded)"

            # 8. Save
            # Output naming: replace 't2m' with 'rho'
            out_name = t2m_nc.name.replace(T2M_VAR, OUT_VAR)
            out_nc = OUT_DIR / out_name

            ds_out.to_netcdf(out_nc, encoding=ENC)
            print(f"    -> Saved: {out_nc.name}")

        finally:
            ds_t.close()
            ds_p.close()

    print("Processing completed successfully.")