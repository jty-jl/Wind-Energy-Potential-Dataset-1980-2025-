#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
[Step 3a] Script to generate a binary WDPA (Protected Areas) mask.

This script processes WDPA vector data to create a boolean mask (.npy).
It handles the 180-degree meridian crossing by duplicating geometries
at +/- 360 degrees before rasterization.
- Output: A boolean numpy array (.npy) where True indicates a Protected Area.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.ops import transform as shp_transform
from rasterio import features
from affine import Affine

# ================= Configuration =================

# Directory containing WDPA Shapefiles (split files 0/1/2...)
WDPA_SOURCE_DIR = Path("data") / "masks" / "source" / "WDPA"

# Reference NetCDF to define the target grid
REF_NC_PATH = Path("data") / "sp" / "1980" / "ocean" / "era5_sp_1980_01_ocean_eez.nc"

# Output path for the generated mask
OUT_MASK_PATH = Path("data") / "masks" / "wdpa_ocean_mask_bool_targetgrid.npy"

# Coordinate names
LON_NAME = "longitude"
LAT_NAME = "latitude"


# ================= Helper Functions =================

def detect_grid(nc_path):
    ds = xr.open_dataset(nc_path)
    try:
        if LON_NAME not in ds.coords or LAT_NAME not in ds.coords:
            raise KeyError(f"Missing coordinates {LON_NAME}/{LAT_NAME} in {nc_path.name}")
        if not np.all(np.diff(ds[LON_NAME].values) > 0):
            ds = ds.sortby(LON_NAME)
        if not np.all(np.diff(ds[LAT_NAME].values) < 0):
            ds = ds.sortby(LAT_NAME, ascending=False)

        lons = ds[LON_NAME].values
        lats = ds[LAT_NAME].values
        print(f"[GRID] lon start={lons[0]:.3f}, end={lons[-1]:.3f}, size={len(lons)}")
        print(f"[GRID] lat start={lats[0]:.3f}, end={lats[-1]:.3f}, size={len(lats)}")
        return lons, lats
    finally:
        ds.close()


def safe_col(gdf, name):
    m = {c.lower(): c for c in gdf.columns}
    if name.lower() not in m:
        raise KeyError(f"Column {name} not found. Available: {list(gdf.columns)}")
    return m[name.lower()]


def load_filter_merge_wdpa(shp_dir):
    """
    Loads WDPA shapefiles, standardizes CRS, filters for Marine/Coastal
    Designated/Established areas, and merges geometries.
    """
    shps = sorted(shp_dir.glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No .shp files found in {shp_dir}")

    out = []
    for shp in shps:
        g = gpd.read_file(shp)
        g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)

        c_status = safe_col(g, "STATUS")
        c_padef = safe_col(g, "PA_DEF")
        c_marine = safe_col(g, "MARINE")

        # Standardize text for filtering
        status = g[c_status].astype(str).str.strip().str.upper()
        padef = g[c_padef].astype(str).str.strip().str.upper()
        marine = g[c_marine].astype(str).str.strip()

        # Filter criteria:
        # STATUS: Designated or Established
        # MARINE: 1 (Coastal) or 2 (Marine)
        ok = (
                status.isin({"DESIGNATED", "ESTABLISHED"}) &
                padef.isin({"0", "1", "TRUE", "FALSE", "T", "F", "Y", "N"}) &
                marine.isin({"1", "2"})
        )

        g = g[ok & g.geometry.notnull()].copy()
        if not g.empty:
            g["geometry"] = g.geometry.buffer(0)
            out.append(g[["geometry"]])
        print(f"[WDPA] {shp.name}: Kept {len(g)} features")

    if not out:
        raise RuntimeError("No WDPA features remained after filtering.")

    return gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs="EPSG:4326")


def shift_lon_geom(geom, offset):
    return shp_transform(lambda x, y, z=None: (x + offset, y) if z is None else (x + offset, y, z), geom)


def to_lon180_vals(vals):
    v = ((vals + 180.0) % 360.0) - 180.0
    v[v == 180.0] = -180.0
    return v


def make_transform_from_centers(lons_center_sorted, lats_center):
    dlon = float(np.diff(lons_center_sorted)[0])
    dlat = float(np.diff(lats_center)[0])

    west = -180.0
    north = float(lats_center[0]) - abs(dlat) / 2.0
    return Affine(dlon, 0, west, 0, dlat, north)


def build_wdpa_mask_on_target_grid(wdpa_gdf, lons_target, lats_target):
    # 1. Seamless buffer
    g0 = wdpa_gdf
    g_plus = g0.copy();
    g_plus["geometry"] = g_plus.geometry.apply(lambda g: shift_lon_geom(g, +360.0))
    g_minus = g0.copy();
    g_minus["geometry"] = g_minus.geometry.apply(lambda g: shift_lon_geom(g, -360.0))
    g_all = gpd.GeoDataFrame(pd.concat([g0, g_plus, g_minus], ignore_index=True), crs="EPSG:4326")

    # 2. Reordering Logic
    lons180 = to_lon180_vals(lons_target)
    order = np.argsort(lons180)
    invord = np.argsort(order)
    lons180_sorted = lons180[order]

    # 3. Rasterization
    H, W = lats_target.size, lons_target.size
    transform = make_transform_from_centers(lons180_sorted, lats_target)

    # Generator for shapes to save memory
    shapes = ((geom, 1) for geom in g_all.geometry.values if geom is not None)

    mask_sorted = features.rasterize(
        shapes=shapes, out_shape=(H, W), transform=transform,
        fill=0, all_touched=True, dtype="uint8"
    ).astype(bool)

    # 4. Restore original order
    mask = mask_sorted[:, invord]
    print(f"[MASK] Protected pixels: {int(mask.sum())} / {mask.size}")
    return mask


# ================= Main Execution =================

if __name__ == "__main__":
    OUT_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Get Grid
    print(f"Loading grid from: {REF_NC_PATH.name}")
    lons, lats = detect_grid(REF_NC_PATH)

    # 2. Load or Generate Mask
    if OUT_MASK_PATH.exists():
        print(f"[CACHE] Mask exists: {OUT_MASK_PATH}")
        print("Delete file to regenerate.")
    else:
        print("Loading and processing WDPA shapefiles...")
        wdpa = load_filter_merge_wdpa(WDPA_SOURCE_DIR)

        print("Building raster mask...")
        wdpa_mask = build_wdpa_mask_on_target_grid(wdpa, lons, lats)

        print(f"Saving to {OUT_MASK_PATH}...")
        np.save(OUT_MASK_PATH, wdpa_mask)
        print("Done.")