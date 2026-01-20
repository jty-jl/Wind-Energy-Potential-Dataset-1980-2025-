#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to generate a binary EEZ mask from a Shapefile.
This script rasterizes a vector Shapefile (EEZ) onto a target NetCDF grid.
- Input 1: EEZ Shapefile (.shp)
- Input 2: A reference NetCDF file (to define the target lat/lon grid)
- Output: A boolean numpy array (.npy) saved to disk.
"""

import os
from pathlib import Path
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd
from rasterio import features
from affine import Affine

# ================= Configuration =================

# Path to the source EEZ Shapefile
SHP_PATH = Path("data") / "masks" / "source" / "eez_v12.shp"

# Path to a REFERENCE NetCDF file
# The script needs one file to know the target grid resolution and bounds
REF_NC_PATH = Path("data") / "sp" / "1980" / "ocean" / "era5_sp_1980_01_ocean.nc"

# Output path for the generated mask
OUT_MASK_PATH = Path("data") / "masks" / "eez_mask_bool_targetgrid.npy"

# Coordinate names in the reference NetCDF
LON_NAME = "longitude"
LAT_NAME = "latitude"


# =================================================

def detect_grid(nc_path: Path):
    if not nc_path.exists():
        raise FileNotFoundError(f"Reference NetCDF not found: {nc_path}")

    ds = xr.open_dataset(nc_path)
    try:
        if LON_NAME not in ds.coords or LAT_NAME not in ds.coords:
            raise KeyError(f"Coordinates {LON_NAME}/{LAT_NAME} missing in {nc_path.name}")

        # Enforce sort order to ensure mask aligns with Step 2b logic
        if not np.all(np.diff(ds[LON_NAME].values) > 0):
            ds = ds.sortby(LON_NAME)
        if not np.all(np.diff(ds[LAT_NAME].values) < 0):
            ds = ds.sortby(LAT_NAME, ascending=False)

        return ds[LON_NAME].values, ds[LAT_NAME].values
    finally:
        ds.close()


def load_eez_seamless(shp_path: Path) -> gpd.GeoDataFrame:
    print(f"Loading Shapefile: {shp_path} ...")
    gdf = gpd.read_file(shp_path)

    # Keep only geometry to save memory
    gdf = gdf[['geometry']]

    print("Applying seamless buffer (Replicating polygons at +/- 360°)...")
    # Shift East (+360)
    gdf_east = gdf.copy()
    gdf_east['geometry'] = gdf.translate(xoff=360)

    # Shift West (-360)
    gdf_west = gdf.copy()
    gdf_west['geometry'] = gdf.translate(xoff=-360)

    # Combine original, east, and west
    gdf_seamless = pd.concat([gdf, gdf_east, gdf_west], ignore_index=True)

    return gdf_seamless


def rasterize_mask(gdf: gpd.GeoDataFrame, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """
    Rasterizes the vector data onto the defined lat/lon grid.
    Returns a boolean numpy array (True = Inside EEZ).
    """
    # 1. Define the transform
    dlon = lons[1] - lons[0]
    dlat = lats[1] - lats[0]

    west = lons[0] - dlon / 2.0
    north = lats[0] - dlat / 2.0

    transform = Affine.translation(west, north) * Affine.scale(dlon, dlat)

    # 2. Rasterize
    shapes = ((geom, 1) for geom in gdf.geometry)

    mask = features.rasterize(
        shapes=shapes,
        out_shape=(len(lats), len(lons)),
        transform=transform,
        fill=0,
        default_value=1,
        dtype='uint8'
    )

    return mask.astype(bool)


def main():
    OUT_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Get Target Grid
    lons, lats = detect_grid(REF_NC_PATH)

    # 2. Load & Prepare Shapefile
    if not SHP_PATH.exists():
        raise FileNotFoundError(f"Source Shapefile not found: {SHP_PATH}")

    gdf_seamless = load_eez_seamless(SHP_PATH)

    # 3. Rasterize
    mask_bool = rasterize_mask(gdf_seamless, lons, lats)

    # 4. Save
    print(f"Saving mask to {OUT_MASK_PATH} ...")
    np.save(OUT_MASK_PATH, mask_bool)

    # Verification stats
    coverage_pct = (mask_bool.sum() / mask_bool.size) * 100
    print(f"Done. EEZ Coverage: {coverage_pct:.2f}% of global grid.")
    print("You can now run Step 2b to apply this mask.")


if __name__ == "__main__":
    main()