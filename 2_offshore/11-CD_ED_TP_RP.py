#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script performs the following operations:
1. Hourly Processing: Reads Gross CF, applies 0.82 efficiency, saves Net CF files.
2. Aggregation: Calculates CF (Mean), CD, ED (Sum), TP (Sum), RP (Sum).

- Input: Hourly Gross CF NetCDF files.
- Output: Hourly Net CF files + Aggregated Metric files.
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ================= Configuration =================

# Input Directory
CF_GROSS_DIR = Path("data") / "CF" / "ocean"

# Output Base Directory
OUT_BASE = Path("data") / "Potential" / "ocean"

# 1. Output Directory for Hourly Net CF
OUT_CF_HOURLY_DIR = OUT_BASE / "CF_hourly_net_082"

# 2. Sub-directories for Derived Metrics
DIRS = {
    "CF_mon": OUT_BASE / "CF_mon",
    "CF_year": OUT_BASE / "CF_year",
    "CapDen": OUT_BASE / "CapacityDensity",
    "Eden_mon": OUT_BASE / "EnergyDensity_mon",
    "Eden_year": OUT_BASE / "EnergyDensity_year",
    "Tech_mon": OUT_BASE / "TechnicalPotential_mon",
    "Tech_year": OUT_BASE / "TechnicalPotential_year",
    "Real_mon": OUT_BASE / "RealisticPotential_mon",
    "Real_year": OUT_BASE / "RealisticPotential_year",
}

# Create directories
OUT_CF_HOURLY_DIR.mkdir(parents=True, exist_ok=True)
for p in DIRS.values():
    p.mkdir(parents=True, exist_ok=True)

# Turbine Parameters (Offshore)
P_RATED_MW = 11.525148634189092  # Rated Power (MW)
D_M = 200.0  # Rotor Diameter (m)
EFFICIENCY = 0.82  # Net Efficiency Factor

# Spacing Scenarios
SPACINGS = {
    "9x7": (9.0, 7.0),
    "7x6": (7.0, 6.0),
    "11x8": (11.0, 8.0),
}

# Realistic Potential Scenarios
SCENARIOS = {
    "p005": 0.005,
    "p01": 0.01,
    "p03": 0.03,
}

# Input Variable Name
CF_VAR_IN = "CF"
TIME_CAND = ("valid_time", "time", "step")

# Compression Settings
ENC_F32 = {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(np.nan)}


def get_time_name(da):
    for nm in TIME_CAND:
        if nm in da.dims:
            return nm
    raise RuntimeError("Time dimension not found.")


def turbines_per_km2(D_m, kx, ky):
    """Calculates turbine density (turbines/km^2)."""
    # Area per turbine (km^2) = (kx*D/1000) * (ky*D/1000)
    area_turb = (kx * D_m / 1000.0) * (ky * D_m / 1000.0)
    return 1.0 / area_turb


def cell_area_km2(da_template):
    lat_name = next(c for c in da_template.coords if "lat" in c.lower())
    lon_name = next(c for c in da_template.coords if "lon" in c.lower())
    lat = da_template[lat_name].values
    lon = da_template[lon_name].values

    # Regular grid assumption
    dlat = np.abs(np.diff(lat)[0])
    dlon = np.abs(np.diff(lon)[0])
    R = 6371.0  # Earth Radius km

    lat_rad = np.deg2rad(lat)
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)

    # Strip area calculation
    lat_n = lat_rad + dlat_rad / 2.0
    lat_s = lat_rad - dlat_rad / 2.0
    strip_area = (R ** 2) * dlon_rad * (np.sin(lat_n) - np.sin(lat_s))

    # Broadcast
    area_grid = np.repeat(strip_area[:, np.newaxis], len(lon), axis=1)

    return xr.DataArray(
        area_grid.astype("float32"),
        coords={lat_name: lat, lon_name: lon},
        dims=(lat_name, lon_name)
    )


def save_da(da, out_path, var_name):
    da.name = var_name
    da.attrs = {}

    ds = da.to_dataset()
    ds.to_netcdf(out_path, encoding={var_name: ENC_F32})
    print(f"    -> Saved: {out_path.name}")


# ================= Main Execution =================

if __name__ == "__main__":

    # 1. Scan Input Files
    files = sorted(CF_GROSS_DIR.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No CF files found in {CF_GROSS_DIR}")

    new_net_files = []

    # ====================================================
    # PHASE 1: Hourly Net CF (x 0.82)
    # ====================================================
    print(f"\n[PHASE 1] Generating Hourly Net CF ({EFFICIENCY})...")

    # Pre-calculate Area
    with xr.open_dataset(files[0]) as ds0:
        AREA_KM2 = cell_area_km2(ds0)

    for f in files:
        with xr.open_dataset(f) as ds:
            if CF_VAR_IN not in ds:
                continue

            # Calculate Net
            CF_Gross = ds[CF_VAR_IN].astype("float32")
            CF_Net = CF_Gross * EFFICIENCY

            # Save (Exact naming)
            out_path = OUT_CF_HOURLY_DIR / f.name

            # Construct dataset manually
            ds_out = CF_Net.to_dataset(name=CF_VAR_IN)
            ds_out.attrs = ds.attrs

            ds_out.to_netcdf(out_path, encoding={CF_VAR_IN: ENC_F32})
            new_net_files.append(out_path)
            print(f"Processed: {f.name}")

    print(f"\n Phase 1 Complete. Saved to: {OUT_CF_HOURLY_DIR}")

    # ====================================================
    # PHASE 2: Aggregated Metrics
    # ====================================================
    print("\n[PHASE 2] Generating Aggregated Metrics...")

    # Load NEW Net files
    ds_net_all = xr.open_mfdataset(new_net_files, chunks="auto", parallel=True)
    CFh_Net = ds_net_all[CF_VAR_IN].astype("float32")
    tname = get_time_name(CFh_Net)

    # Valid mask (for clean output)
    valid2d = np.isfinite(AREA_KM2)

    # --- 1. CF (Mean) ---
    print("[OUTPUT] Generating CF (Mean)...")

    # CF Mon (Net)
    CF_mon_mean = CFh_Net.resample({tname: "1MS"}).mean(skipna=True)
    save_da(CF_mon_mean.where(valid2d), DIRS["CF_mon"] / "CF_mon_offshore.nc", "CF_mon")

    # CF Year (Net)
    CF_year_mean = CFh_Net.mean(dim=tname, skipna=True)
    save_da(CF_year_mean.where(valid2d), DIRS["CF_year"] / "CF_year_offshore.nc", "CF_year")

    # --- 2. CD (Capacity Density, MW/km²) ---
    print("[OUTPUT] Generating CapacityDensity...")
    cap_vars = {}
    for key, (kx, ky) in SPACINGS.items():
        var_name = f"CapDen_{key}_MW_per_km2"
        val = P_RATED_MW * turbines_per_km2(D_M, kx, ky)
        cap_vars[var_name] = ([], val)

    # Save combined CD file
    ds_cd = xr.Dataset(cap_vars)
    ds_cd.to_netcdf(DIRS["CapDen"] / "CapacityDensity_offshore.nc")
    print("    -> Saved: CapacityDensity_offshore.nc")

    # --- 3. ED, TP, RP (Cumulative Sum) ---
    print("[OUTPUT] Generating ED, TP, RP (Cumulative)...")

    # Pre-calculate Generation (MWh)
    # Resample Sum = Sum of (CF_hourly * P_rated)
    Turbine_Gen_Month_Sum = (CFh_Net * P_RATED_MW).resample({tname: "1MS"}).sum(skipna=True)
    Turbine_Gen_Year_Sum = (CFh_Net * P_RATED_MW).sum(dim=tname, skipna=True)

    for key, (kx, ky) in SPACINGS.items():
        print(f"  -> Processing Spacing: {key}")
        N_km2 = turbines_per_km2(D_M, kx, ky)

        # ==========================
        #      Monthly Total
        # ==========================

        # ED: MWh/km²/month
        Eden_mon_total = Turbine_Gen_Month_Sum * N_km2
        save_da(Eden_mon_total.where(valid2d),
                DIRS["Eden_mon"] / f"Eden_mon_{key}_offshore.nc",
                "Eden_MWhkm2_per_month")

        # TP: MWh/month
        Tech_mon_total = Eden_mon_total * AREA_KM2
        save_da(Tech_mon_total.where(valid2d),
                DIRS["Tech_mon"] / f"Tech_mon_{key}_offshore.nc",
                "TechPot_cell_MWh_per_month")

        # RP: MWh/month
        for sk, frac in SCENARIOS.items():
            Real_mon_total = Tech_mon_total * frac
            save_da(Real_mon_total.where(valid2d),
                    DIRS["Real_mon"] / f"Real_mon_{key}_{sk}_offshore.nc",
                    "RealPot_cell_MWh_per_month")

        # ==========================
        #      Annual Total
        # ==========================

        # ED: MWh/km²/year
        Eden_year_total = Turbine_Gen_Year_Sum * N_km2
        save_da(Eden_year_total.where(valid2d),
                DIRS["Eden_year"] / f"Eden_year_{key}_offshore.nc",
                "Eden_MWhkm2_per_year")

        # TP: MWh/year
        Tech_year_total = Eden_year_total * AREA_KM2
        save_da(Tech_year_total.where(valid2d),
                DIRS["Tech_year"] / f"Tech_year_{key}_offshore.nc",
                "TechPot_cell_MWh_per_year")

        # RP: MWh/year
        for sk, frac in SCENARIOS.items():
            Real_year_total = Tech_year_total * frac
            save_da(Real_year_total.where(valid2d),
                    DIRS["Real_year"] / f"Real_year_{key}_{sk}_offshore.nc",
                    "RealPot_cell_MWh_per_year")

    print("Processing completed successfully.")