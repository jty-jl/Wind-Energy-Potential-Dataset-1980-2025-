#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script filters the 'sp' (Surface Pressure) dataset based on 2_offshore wind suitability criteria:
1.  **Water Depth**: Filters for suitable depths (e.g., 5m - 1200m).
2.  **Seafloor Slope**: Excludes areas with steep slopes (> 10 degrees).
3.  **Distance to Coast**: Filters based on operational constraints (e.g., 10km - 300km).

- Input 1: GEBCO (NetCDF from Step 5a).
- Input 2: 'sp' NetCDF files (Output from WDPA step).
- Output: 'sp' NetCDF files with unsuitable ocean areas masked out (set to NaN).
"""

from pathlib import Path
import numpy as np
import xarray as xr
from scipy import ndimage

# ================= Configuration =================

# 1. Paths
# Input 1: GEBCO Ocean file
GEBCO_OCEAN_NC = Path("data") / "GEBCO" / "GEBCO_2025_0p25_ocean.nc"

# Input 2: 'sp' data from previous step
IN_DIR = Path("data") / "sp" / "ocean" / "sp_WDPA"

# Output: Final filtered data
OUT_DIR = Path("data") / "sp" / "ocean" / "sp_GEBCO"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Variable Names
ELEV_NAME = "elevation"
VAR_NAME = "sp"
LON_NAME = "longitude"
LAT_NAME = "latitude"

# 3. Filtering Criteria
DEPTH_MIN = 5.0
DEPTH_MAX = 1200.0

SLOPE_MAX_DEG = 10.0

DIST_MIN_KM = 10.0
DIST_MAX_KM = 300.0

# 4. Compression Settings
ENC = {
    VAR_NAME: {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan)
    }
}


def detect_time_dim(da):
    for nm in ("valid_time", "time", "step"):
        if nm in da.dims:
            return nm
    return None


def ensure_grid_sorted(ds, lon_name=LON_NAME, lat_name=LAT_NAME):
    if lon_name in ds and not np.all(np.diff(ds[lon_name].values) > 0):
        ds = ds.sortby(lon_name)
    if lat_name in ds and not np.all(np.diff(ds[lat_name].values) < 0):
        ds = ds.sortby(lat_name, ascending=False)
    return ds


def compute_slope_deg(elev_da, lon_name=LON_NAME, lat_name=LAT_NAME):
    lons = elev_da[lon_name].values
    lats = elev_da[lat_name].values

    # Calculate grid step size
    dlon_deg = float(np.diff(lons)[0])
    dlat_deg = float(np.diff(lats)[0])
    meters_per_deg = 111_320.0

    # Grid spacing in meters
    cosphi = np.cos(np.deg2rad(lats))
    dx = meters_per_deg * abs(dlon_deg) * cosphi
    dy = meters_per_deg * abs(dlat_deg)

    z = elev_da.values

    # Calculate gradients (dz/dx, dz/dy)
    dzdx = np.empty_like(z, dtype="float32")
    dzdx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / 2.0
    dzdx[:, 0] = (z[:, 1] - z[:, 0])
    dzdx[:, -1] = (z[:, -1] - z[:, -2])
    dzdx = dzdx / dx[:, None]

    dzdy = np.empty_like(z, dtype="float32")
    dzdy[1:-1, :] = (z[2:, :] - z[:-2, :]) / 2.0
    dzdy[0, :] = (z[1, :] - z[0, :])
    dzdy[-1, :] = (z[-1, :] - z[-2, :])
    dzdy = dzdy / dy

    # Calculate Slope
    slope_rad = np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy))
    slope_deg = np.degrees(slope_rad)

    out = xr.DataArray(
        slope_deg,
        coords={lat_name: lats, lon_name: lons},
        dims=(lat_name, lon_name),
        name="slope_deg",
        attrs={"long_name": "Seafloor slope", "units": "degree"}
    )
    # Mask out non-ocean areas
    out = out.where(np.isfinite(elev_da))
    return out


def compute_distance_to_coast_km(ocean_mask, lons, lats):
    """
    Computes Euclidean distance from ocean pixels to the nearest land pixel.
    """
    ocean_mask = np.asarray(ocean_mask, dtype=bool)  # True=Ocean, False=Land (NaN in GEBCO)

    # Handle edge cases (all land or all ocean)
    if ocean_mask.size == 0 or (ocean_mask.sum() == 0) or ((~ocean_mask).sum() == 0):
        return xr.DataArray(
            np.full(ocean_mask.shape, np.nan, dtype="float32"),
            coords={LAT_NAME: lats, LON_NAME: lons},
            dims=(LAT_NAME, LON_NAME),
            name="dist_coast_km"
        )

    # Approximate pixel size in meters for the transform
    meters_per_deg = 111_320.0
    dlon_deg = float(np.diff(lons)[0])
    dlat_deg = float(np.diff(lats)[0])

    dx_row = meters_per_deg * abs(dlon_deg) * np.cos(np.deg2rad(lats))
    dx_mean = float(np.mean(dx_row))
    dy = meters_per_deg * abs(dlat_deg)

    # edt calculates distance to nearest ZERO. So we input Ocean Mask (Land is 0).
    dist_m = ndimage.distance_transform_edt(ocean_mask, sampling=(dy, dx_mean)).astype("float32")

    # Correct for latitude distortion (approximate)
    scale = (dx_row / dx_mean)[:, None]
    dist_m *= np.sqrt((scale ** 2 + 1.0) / 2.0)

    dist_km = dist_m / 1000.0

    # Mask land pixels
    dist_km = np.where(ocean_mask, dist_km, np.nan)

    return xr.DataArray(
        dist_km,
        coords={LAT_NAME: lats, LON_NAME: lons},
        dims=(LAT_NAME, LON_NAME),
        name="dist_coast_km",
        attrs={"long_name": "Distance to nearest land", "units": "km"}
    )


# ================= Main Execution =================

if __name__ == "__main__":
    # 1. Load GEBCO Data
    if not GEBCO_OCEAN_NC.exists():
        raise FileNotFoundError(f"GEBCO file not found: {GEBCO_OCEAN_NC}")

    ds_g = xr.open_dataset(GEBCO_OCEAN_NC)
    if ELEV_NAME not in ds_g:
        raise KeyError(f"Variable '{ELEV_NAME}' not found in GEBCO file.")
    ds_g = ensure_grid_sorted(ds_g)

    elev = ds_g[ELEV_NAME].astype("float32")
    lons = ds_g[LON_NAME].values
    lats = ds_g[LAT_NAME].values

    # 2. Compute Metrics (Depth, Slope, Distance)
    print("Computing geophysical metrics...")

    # Ocean Mask
    ocean_mask = np.isfinite(elev) & (elev < 0)

    # Depth
    depth = (-elev).where(ocean_mask)

    # Slope
    slope_deg = compute_slope_deg(elev, LON_NAME, LAT_NAME)

    # Distance to Coast
    dist_km = compute_distance_to_coast_km(ocean_mask.values, lons, lats)

    # 3. Create Combined Filter Mask
    print("Generating Suitability Mask...")

    # Criteria: Depth Range AND Slope Limit AND Distance Range
    mask_combined = (
            (depth >= DEPTH_MIN) & (depth <= DEPTH_MAX) &
            (slope_deg <= SLOPE_MAX_DEG) &
            (dist_km >= DIST_MIN_KM) & (dist_km <= DIST_MAX_KM)
    )

    # Ensure mask is False where data is NaN (Ocean only)
    mask_combined = mask_combined.where(ocean_mask, False)
    mask_bool = mask_combined.fillna(False).values

    n_valid = mask_bool.sum()
    pct_valid = (n_valid / mask_bool.size) * 100
    print(f"[STATS] Valid Suitability Pixels: {n_valid} ({pct_valid:.2f}%)")

    # 4. Batch Process 'sp' Files
    nc_list = sorted(IN_DIR.glob("*.nc"))
    if not nc_list:
        raise FileNotFoundError(f"No .nc files found in {IN_DIR}")

    print(f"Processing {len(nc_list)} files...")

    # Create DataArray for broadcasting the mask
    mask_da = xr.DataArray(
        mask_bool,
        coords={LAT_NAME: lats, LON_NAME: lons},
        dims=(LAT_NAME, LON_NAME)
    )

    for nc in nc_list:
        print(f"  -> Processing: {nc.name}")
        ds = xr.open_dataset(nc)
        try:
            if VAR_NAME not in ds:
                print(f"     [Skip] Missing variable {VAR_NAME}")
                continue

            # Ensure grid alignment
            ds = ensure_grid_sorted(ds)
            if not np.array_equal(ds[LON_NAME].values, lons) or not np.array_equal(ds[LAT_NAME].values, lats):
                raise ValueError(f"Grid mismatch in {nc.name}. Please resample to GEBCO grid.")

            da = ds[VAR_NAME].astype("float32")

            # Apply Mask
            da_filtered = da.where(mask_da)

            # Save Output
            ds_out = ds.copy()
            ds_out[VAR_NAME] = da_filtered

            # Update history
            ds_out[VAR_NAME].attrs["note"] = "Filtered by GEBCO Depth, Slope, and Distance to Coast."

            out_nc = OUT_DIR / nc.name.replace(".nc", "_gebco.nc")
            ds_out.to_netcdf(out_nc, encoding=ENC)

        finally:
            ds.close()

    print("Processing completed successfully.")