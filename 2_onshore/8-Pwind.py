#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script calculates the hourly wind power output (Pwind) using a piecewise
power curve model.
1. Partial Load (Cut-in <= U < Rated): P = min(0.5 * rho * A * Cp * U^3, P_rated)
2. Full Load (Rated <= U < Cut-out): P = P_rated
3. Shutdown / Idle: P = 0.0 W
4. Invalid/Masked: P = NaN

- Input: V100 (m/s) and rho (kg/m3) NetCDF files.
- Output: Pwind (Watts) NetCDF files.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Input Directories
V100_DIR = Path("data") / "wind100" / "land" / "wind100"
RHO_DIR = Path("data") / "rho" / "land"

# Output Directory
OUT_DIR = Path("data") / "Pwind" / "land"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable Names
VAR_V100 = "V100"
VAR_RHO = "rho"
VAR_OUT = "Pwind"

# Turbine Parameters
CP = 0.45
D = 135.0  # Rotor Diameter (m)
U_IN = 3.0  # Cut-in Speed (m/s)
U_R = 11.0  # Rated Speed (m/s)
U_OUT = 25.0  # Cut-out Speed (m/s)
RHO_REF = 1.225  # Reference Air Density (kg/m^3)

# Derived Constants
AREA = np.pi * (D ** 2) / 4.0
P_RATED = 0.5 * RHO_REF * AREA * CP * (U_R ** 3)

# Compression Settings
ENC = {
    VAR_OUT: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}

# ================= Main Execution =================

if __name__ == "__main__":

    print(f"[Config] Land Turbine: D={D}m, Rated Power={P_RATED / 1e6:.3f} MW")

    # List input files
    rho_files = sorted(RHO_DIR.glob("*.nc"))
    if not rho_files:
        raise FileNotFoundError(f"No rho files found in {RHO_DIR}")

    print(f"Found {len(rho_files)} files to process...")

    for rho_nc in rho_files:
        # Construct corresponding V100 filename
        # Assumption: Filenames align (e.g., rho_1980_01.nc -> V100_1980_01.nc)
        v_name = rho_nc.name.replace(VAR_RHO, VAR_V100)
        v_nc = V100_DIR / v_name

        if not v_nc.exists():
            print(f"[WARN] Matching V100 file not found: {v_name}. Skipping.")
            continue

        print(f"[Processing] {v_nc.name} + {rho_nc.name}")

        ds_v = xr.open_dataset(v_nc, chunks="auto")
        ds_r = xr.open_dataset(rho_nc, chunks="auto")

        try:
            if VAR_V100 not in ds_v:
                raise KeyError(f"Variable {VAR_V100} missing in {v_nc.name}")
            if VAR_RHO not in ds_r:
                raise KeyError(f"Variable {VAR_RHO} missing in {rho_nc.name}")

            # 1. Alignment
            ds_v = ds_v.reindex_like(ds_r, method=None)

            # 2. Load Data
            U = ds_v[VAR_V100].astype("float64")
            rho = ds_r[VAR_RHO].astype("float64")

            # Clean Data
            U = U.where(np.isfinite(U))
            rho = rho.where(np.isfinite(rho))

            # 3. Calculate Power
            P = xr.full_like(U, np.nan, dtype="float64")

            # A. Calculate Aerodynamic Power
            P_aero = 0.5 * rho * AREA * CP * (U ** 3)

            # B. Region 1: Partial Load (Cut-in <= U < Rated)
            # Logic: min(P_aero, P_rated) to handle high air density cases
            mask_partial = (U >= U_IN) & (U < U_R)
            P_capped = P_aero.clip(max=P_RATED)
            P = xr.where(mask_partial, P_capped, P)

            # C. Region 2: Full Load (Rated <= U < Cut-out)
            mask_rated = (U >= U_R) & (U < U_OUT)
            P = xr.where(mask_rated, P_RATED, P)

            # D. Region 3 & Invalid: Shutdown and masking
            # Logic:
            # 1. fillna(0.0) sets cut-in/cut-out regions to 0 W
            # 2. where(isfinite(U)) restores NaNs for invalid geographic areas
            P = P.fillna(0.0).where(np.isfinite(U))

            # 4. Metadata & Formatting
            P = P.astype("float32")
            P.name = VAR_OUT
            P.attrs = {
                "long_name": "Estimated Hourly Wind Power Output",
                "standard_name": "wind_power_output",
                "units": "W",
                "turbine_D_m": D,
                "turbine_Cp": CP,
                "cut_in_speed": U_IN,
                "rated_speed": U_R,
                "cut_out_speed": U_OUT,
                "rated_power_W": float(P_RATED),
                "description": "Piecewise function with density correction. "
                               "Partial: min(0.5*rho*A*Cp*U^3, P_rated). "
                               "Rated: P_rated. Shutdown: 0 W."
            }

            # 5. Save Output
            ds_out = xr.Dataset({VAR_OUT: P}, coords=ds_r.coords)
            ds_out.attrs = ds_r.attrs
            ds_out.attrs["comment"] = "Wind power calculated for land areas."

            out_nc = OUT_DIR / rho_nc.name.replace(VAR_RHO, VAR_OUT)
            ds_out.to_netcdf(out_nc, encoding=ENC)
            print(f"    -> Saved: {out_nc.name}")

        finally:
            ds_v.close()
            ds_r.close()

    print("Processing completed successfully.")