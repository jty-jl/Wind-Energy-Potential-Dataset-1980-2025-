#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script calculates the hourly wind power output (Pwind) using a piecewise
power curve model based on the logic:
1. Partial Load (Cut-in <= U < Rated): P = min(P_aero, P_rated)
   - P_aero = 0.5 * rho * A * Cp * U^3
   - Uses variable air density (rho) for physical accuracy.
2. Full Load (Rated <= U < Cut-out): P = P_rated
3. Shutdown / Idle: P = 0.0 W
4. Invalid/Land: P = NaN

- Input:
  - V100 (100m Wind Speed, m/s)
  - rho  (Air Density, kg/m^3)
- Output: Pwind (Watts)
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Input Directories (Relative Paths)
V100_DIR = Path("data") / "wind100" / "ocean"
RHO_DIR = Path("data") / "rho" / "ocean"

# Output Directory
OUT_DIR = Path("data") / "Pwind" / "ocean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Variable Names
V100_VAR = "V100"
RHO_VAR = "rho"
OUT_VAR = "Pwind"

# Time Dimension Candidates
TIME_CAND = ("valid_time", "time", "step")

# ================= Turbine Parameters =================
# Scenario: Offshore Wind Turbine
D = 200.0  # Rotor Diameter (m)
CP = 0.45  # Power Coefficient
U_IN = 3.0  # Cut-in Speed (m/s)
U_R = 11.0  # Rated Speed (m/s)
U_OUT = 25.0  # Cut-out Speed (m/s)
RHO_REF = 1.225  # Reference Air Density (kg/m^3)

# Derived Constants
AREA = np.pi * (D ** 2) / 4.0
# Rated Power (Watts) calculated at reference conditions
P_RATED = 0.5 * RHO_REF * AREA * CP * (U_R ** 3)

# Output Compression Settings
ENC = {
    OUT_VAR: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


def detect_time_dim(da):
    for nm in TIME_CAND:
        if nm in da.dims:
            return nm
    raise RuntimeError(f"Time dimension not found. Candidates: {TIME_CAND}")


def open_ds_safe(path):
    for eng in ("netcdf4", "h5netcdf", None):
        try:
            return xr.open_dataset(path, engine=eng, mask_and_scale=False, chunks="auto")
        except Exception:
            continue
    raise RuntimeError(f"Failed to open file: {path.name}")


# ================= Main Execution =================

if __name__ == "__main__":

    print(f"[Config] Turbine: D={D}m, Cp={CP}, Rated Power={P_RATED / 1e6:.2f} MW")

    # List input files
    rho_files = sorted(RHO_DIR.glob("*.nc"))
    if not rho_files:
        raise FileNotFoundError(f"No rho files found in {RHO_DIR}")

    print(f"Found {len(rho_files)} files to process...")

    for rho_nc in rho_files:
        v_name = rho_nc.name.replace(RHO_VAR, V100_VAR)
        v_nc = V100_DIR / v_name

        if not v_nc.exists():
            print(f"[WARN] Matching V100 file not found: {v_name}. Skipping.")
            continue

        print(f"[Processing] {v_nc.name} + {rho_nc.name}")

        ds_v = open_ds_safe(v_nc)
        ds_r = open_ds_safe(rho_nc)

        try:
            if V100_VAR not in ds_v:
                raise KeyError(f"Variable {V100_VAR} missing in {v_nc.name}")
            if RHO_VAR not in ds_r:
                raise KeyError(f"Variable {RHO_VAR} missing in {rho_nc.name}")

            # 1. Alignment
            ds_v = ds_v.reindex_like(ds_r, method=None)

            # 2. Data Loading & Cleaning
            U = ds_v[V100_VAR].astype("float64")
            rho = ds_r[RHO_VAR].astype("float64")

            # Filter logic
            U = U.where(np.isfinite(U))
            rho = rho.where(np.isfinite(rho))

            # 3. Power Calculation
            print("    -> Calculating power curve...")

            # Initialize with NaNs
            P = xr.full_like(U, np.nan, dtype="float64")

            # A. Calculate Aerodynamic Power
            # P_aero = 0.5 * rho * A * Cp * U^3
            P_aero = 0.5 * rho * AREA * CP * (U ** 3)

            # B. Partial Load Region: Cut-in <= U < Rated
            # Apply Cap: min(P_aero, P_rated)
            # This handles high-density events where P_aero > P_rated even below U_rated
            mask_partial = (U >= U_IN) & (U < U_R)
            P_capped = P_aero.clip(max=P_RATED)
            P = xr.where(mask_partial, P_capped, P)

            # C. Full Load Region: Rated <= U < Cut-out
            # Fixed at Rated Power
            mask_rated = (U >= U_R) & (U < U_OUT)
            P = xr.where(mask_rated, P_RATED, P)

            # D. Shutdown & Invalid Regions
            # 1. fillna(0.0): Sets all uncalculated areas (Low wind, Storm, AND Land) to 0.0
            # 2. where(isfinite(U)): Restores NaN where the original Wind Speed was NaN (Land)
            P = P.fillna(0.0).where(np.isfinite(U))

            # 4. Metadata & Formatting
            P = P.astype("float32")
            P.name = OUT_VAR
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
                "description": "Calculated using variable air density. "
                               "Partial load: min(0.5*rho*A*Cp*U^3, P_rated). "
                               "Full load: P_rated. "
                               "Shutdown/Calm: 0 W. Land: NaN."
            }

            # 5. Save Output
            ds_out = xr.Dataset({OUT_VAR: P}, coords=ds_r.coords)
            # Inherit global attributes
            ds_out.attrs = ds_r.attrs
            ds_out.attrs["comment"] = "Offshore wind power estimated from ERA5 data."

            out_file = OUT_DIR / rho_nc.name.replace(RHO_VAR, OUT_VAR)
            ds_out.to_netcdf(out_file, encoding=ENC)
            print(f"    -> Saved: {out_file.name}")

        finally:
            ds_v.close()
            ds_r.close()

    print("Processing completed successfully.")