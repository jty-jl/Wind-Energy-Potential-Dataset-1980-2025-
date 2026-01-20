#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GEBCO Bathymetry Processing Script.

This script processes the high-resolution GEBCO dataset:
1. Resamples the elevation data to a target 0.25-degree grid using average resampling.
2. Standardizes longitude to [0, 360) to match ERA5 conventions.
3. Splits the data into Land (elevation >= 0) and Ocean (elevation < 0) files.
4. Saves the results as NetCDF files with optimized compression.

- Input: GEBCO global grid (NetCDF).
- Output: Two NetCDF files (Land and Ocean) on a 0.25-degree grid.
"""

from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
from affine import Affine

# ================= Configuration =================

# Input Path
GEBCO_NC = Path("data") / "GEBCO" / "GEBCO_2025.nc"

# Output Directory
OUT_DIR = Path("data") / "GEBCO"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_LAND_PATH = OUT_DIR / "GEBCO_2025_0p25_land.nc"
OUT_OCEAN_PATH = OUT_DIR / "GEBCO_2025_0p25_ocean.nc"

# Target Grid Definition (0.25 degree resolution)
# Bounds: Lon -0.125 to 359.875 (1440 cols), Lat 90.125 to -90.125 (721 rows)
DLON, DLAT = 0.25, -0.25
WEST, NORTH = -0.125, 90.125
NCOLS, NROWS = 1440, 721

TRANSFORM = Affine(DLON, 0, WEST, 0, DLAT, NORTH)
OUT_SHAPE = (NROWS, NCOLS)

CENTER_LONS = WEST + (np.arange(NCOLS) + 0.5) * DLON
CENTER_LATS = NORTH + (np.arange(NROWS) + 0.5) * DLAT

ENC = {
    "elevation": {
        "zlib": True, "complevel": 4, "shuffle": True,
        "dtype": "float32", "_FillValue": np.float32(np.nan),
        "chunksizes": (180, 360),
    }
}


def detect_lon_lat(ds):
    lon_name = next((c for c in ("longitude", "lon", "x") if c in ds.coords), None)
    lat_name = next((c for c in ("latitude", "lat", "y") if c in ds.coords), None)
    if lon_name is None or lat_name is None:
        raise KeyError(f"Coordinates not found. Available: {list(ds.coords)}")
    return lon_name, lat_name


def detect_elev_var(ds):
    return "elevation" if "elevation" in ds.data_vars else list(ds.data_vars)[0]


def to_lon0360(da, lon_name="longitude"):
    lon = da.coords[lon_name].values
    if np.nanmin(lon) < 0:
        print(f"[INFO] Converting {lon_name} range to [0, 360)...")
        lon2 = (lon + 360.0) % 360.0
        da = da.assign_coords({lon_name: lon2}).sortby(lon_name)
    return da


# ================= Main Execution =================

if __name__ == "__main__":
    if not GEBCO_NC.exists():
        raise FileNotFoundError(f"GEBCO file not found at: {GEBCO_NC}")

    print(f"Processing GEBCO file: {GEBCO_NC.name}")

    # 1. Load GEBCO Dataset
    ds_g = xr.open_dataset(GEBCO_NC, chunks="auto")
    vname = detect_elev_var(ds_g)
    lon_g, lat_g = detect_lon_lat(ds_g)

    # Standardize names and CRS
    elev = ds_g[vname].astype("float32")
    if lon_g != "longitude" or lat_g != "latitude":
        elev = elev.rename({lon_g: "longitude", lat_g: "latitude"})

    # Write CRS for rioxarray
    elev = elev.rio.write_crs("EPSG:4326")
    elev = to_lon0360(elev, "longitude")

    # 2. Resample to 0.25 degree grid
    elev_025 = elev.rio.reproject(
        dst_crs="EPSG:4326",
        transform=TRANSFORM,
        shape=OUT_SHAPE,
        resampling=Resampling.average,
    )

    # 3. Assign Center Coordinates
    elev_025 = elev_025.assign_coords({"x": ("x", CENTER_LONS), "y": ("y", CENTER_LATS)})
    elev_025 = elev_025.rename({"x": "longitude", "y": "latitude"})

    # 4. Split into Land and Ocean
    # Criteria: Land >= 0m, Ocean < 0m
    print("Splitting into Land and Ocean datasets...")
    land_da = elev_025.where(elev_025 >= 0)
    ocean_da = elev_025.where(elev_025 < 0)

    # 5. Create Output Datasets
    ds_land = xr.Dataset(
        {"elevation": land_da},
        coords={"longitude": CENTER_LONS, "latitude": CENTER_LATS}
    )
    ds_ocean = xr.Dataset(
        {"elevation": ocean_da},
        coords={"longitude": CENTER_LONS, "latitude": CENTER_LATS}
    )

    # Add Metadata attributes
    for d in (ds_land, ds_ocean):
        d["elevation"].attrs = {"units": "m", "long_name": "Elevation (Land >= 0, Ocean < 0)"}
        d["longitude"].attrs.update({"standard_name": "longitude", "units": "degrees_east", "axis": "X"})
        d["latitude"].attrs.update({"standard_name": "latitude", "units": "degrees_north", "axis": "Y"})

    # 6. Save to Disk
    print(f"Saving Land file to {OUT_LAND_PATH}...")
    ds_land.to_netcdf(OUT_LAND_PATH, encoding=ENC)

    print(f"Saving Ocean file to {OUT_OCEAN_PATH}...")
    ds_ocean.to_netcdf(OUT_OCEAN_PATH, encoding=ENC)

    print("Processing completed successfully.")