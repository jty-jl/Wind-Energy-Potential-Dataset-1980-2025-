#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script generates a boolean grid mask for Land-based Protected Areas (WDPA).
It performs two main stages:
1. Vector Processing: Filters WDPA polygons by attributes and intersects them
   with a Land Border shapefile to exclude marine areas.
   -> Output: Intermediate GeoPackage (.gpkg).
2. Rasterization: Rasterizes the filtered polygons onto the grid.
   -> Output: Boolean NumPy array (.npy) where True indicates a Protected Area.

- Input 1: Raw WDPA Shapefiles.
- Input 2: Land Border Shapefile.
- Input 3: Reference ERA5 NetCDF.
- Output: wdpa_land_mask.npy.
"""

import os
import sys
import glob
import time
import fiona
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union
from shapely.prepared import prep
from fiona.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin

# ================= Configuration =================

# 1. Directory containing WDPA Shapefiles (split files 0/1/2...)
WDPA_RAW_DIR = Path("data") / "masks" / "source" / "WDPA"

# Land Border Shapefile
LAND_SHP_PATH = Path("data") / "masks" / "World_Land_Shape" / "World_Map.shp"

# Reference File
ERA5_REF_NC = Path("data") / "sp" / "land" / "sp_MCD"

# 2. Outputs
OUT_DIR = Path("data") / "masks" / "WDPA_Processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Intermediate Vector Output
OUT_GPKG = OUT_DIR / "wdpa_land_filtered.gpkg"
OUT_LAYER = "wdpa_land"

# Final Raster Output
OUT_MASK_NPY = OUT_DIR / "wdpa_land_mask_bool.npy"

# 3. Parameters
# WDPA Filter Settings
VALID_STATUS = {"DESIGNATED", "ESTABLISHED"}
VALID_MARINE = {0, 1}
VALID_PADEF = {0, 1}

# Geometry Simplification Tolerance (degrees) to speed up intersection
SIMPLIFY_TOL = 0.005


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def to_multipolygon(geom):
    if geom.is_empty: return None

    # Handle single Polygon
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])

    # Handle MultiPolygon
    if isinstance(geom, MultiPolygon):
        return geom

    # Handle GeometryCollection
    if isinstance(geom, GeometryCollection):
        polys = []
        for g in geom.geoms:
            if isinstance(g, Polygon):
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend(g.geoms)
        if not polys: return None
        return MultiPolygon(polys)

    return None


def normalize_status(val):
    return (val or "").strip().upper()


def normalize_int(val):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return -9999


# ================= Stage 1: Vector Processing =================

def process_vectors():
    if OUT_GPKG.exists():
        log(f"[Vector] Intermediate GPKG found: {OUT_GPKG}. Skipping vector processing.")
        return

    log(f"[Vector] Loading Land Shapefile: {LAND_SHP_PATH.name}")
    if not LAND_SHP_PATH.exists():
        raise FileNotFoundError(f"Land shapefile not found: {LAND_SHP_PATH}")

    # Load and union land polygons
    land_polys = []
    with fiona.open(LAND_SHP_PATH) as src:
        # Check CRS
        if src.crs and CRS.from_user_input(src.crs).to_epsg() != 4326:
            log("[WARN] Land CRS is not EPSG:4326. Intersection might fail if WDPA is 4326.")

        for feat in src:
            g = shape(feat["geometry"])
            mp = to_multipolygon(g)
            if mp:
                land_polys.append(mp)

    if not land_polys:
        raise RuntimeError("No valid polygons found in Land Shapefile.")

    log("[Vector] Merging land polygons (Unary Union)...")
    land_union = unary_union(land_polys).buffer(0)
    land_prep = prep(land_union)

    # Find WDPA files
    wdpa_files = sorted(list(WDPA_RAW_DIR.rglob("*.shp")))
    if not wdpa_files:
        raise FileNotFoundError(f"No .shp files found in {WDPA_RAW_DIR}")

    log(f"[Vector] Found {len(wdpa_files)} WDPA shapefiles. Starting filtering...")

    # Schema for output GPKG
    schema = {
        "geometry": "MultiPolygon",
        "properties": {"PA_DEF": "int", "MARINE": "int", "STATUS": "str:32"}
    }

    count_in = 0
    count_out = 0

    with fiona.open(OUT_GPKG, "w", driver="GPKG", crs="EPSG:4326", schema=schema, layer=OUT_LAYER) as sink:
        for shp_path in wdpa_files:
            log(f"  -> Processing {shp_path.name}...")
            with fiona.open(shp_path) as src:
                for feat in src:
                    count_in += 1
                    props = feat["properties"]

                    # 1. Attribute Filter
                    status = normalize_status(props.get("STATUS"))
                    if status not in VALID_STATUS: continue

                    marine = normalize_int(props.get("MARINE"))
                    if marine not in VALID_MARINE: continue

                    padef = normalize_int(props.get("PA_DEF"))
                    if padef not in VALID_PADEF: continue

                    # 2. Geometry Check
                    geom = shape(feat["geometry"])
                    if not geom.is_valid:
                        geom = geom.buffer(0)

                    # Quick bounding box check
                    if not land_prep.intersects(geom):
                        continue

                    # 3. Intersection with Land
                    try:
                        inter = geom.intersection(land_union)
                    except Exception:
                        continue

                    if inter.is_empty:
                        continue

                    # 4. Simplify and Standardize
                    inter = inter.simplify(SIMPLIFY_TOL, preserve_topology=True)
                    mp = to_multipolygon(inter)

                    if mp:
                        sink.write({
                            "geometry": mapping(mp),
                            "properties": {
                                "PA_DEF": padef,
                                "MARINE": marine,
                                "STATUS": status
                            }
                        })
                        count_out += 1

    log(f"[Vector] Finished. {count_out}/{count_in} features retained.")
    log(f"[Vector] Saved to: {OUT_GPKG}")


# ================= Stage 2: Rasterization =================

def get_grid_info(nc_dir):
    files = sorted(list(nc_dir.rglob("*.nc")))
    files = [f for f in files if not f.name.startswith("._") and f.stat().st_size > 1024]

    if not files:
        raise FileNotFoundError(f"No reference NetCDF files found in {nc_dir}")

    ref_file = files[0]
    log(f"[Grid] Using reference file: {ref_file.name}")

    ds = xr.open_dataset(ref_file)

    # Identify lat/lon names
    lat_name = next((c for c in ds.coords if c in ["latitude", "lat"]), None)
    lon_name = next((c for c in ds.coords if c in ["longitude", "lon"]), None)

    if not lat_name or not lon_name:
        raise KeyError("Latitude/Longitude coordinates not found in reference file.")

    lats = ds[lat_name].values
    lons = ds[lon_name].values
    ds.close()

    # Calculate Transform
    # Assuming regular grid
    ny, nx = lats.size, lons.size
    res_lat = np.abs(np.mean(np.diff(lats)))
    res_lon = np.abs(np.mean(np.diff(lons)))

    # West edge (center - half pixel)
    west = lons.min() - 0.5 * res_lon
    # North edge (center + half pixel)
    north = lats.max() + 0.5 * res_lat

    transform = from_origin(west, north, res_lon, res_lat)

    return transform, (ny, nx), lats, lons


def rasterize_mask():
    # 1. Get Grid
    transform, shape_hw, lats, lons = get_grid_info(ERA5_REF_NC)
    ny, nx = shape_hw
    log(f"[Grid] Shape: {ny}x{nx}, Transform: {transform}")

    # 2. Load Shapes
    log(f"[Raster] Loading polygons from {OUT_GPKG}...")
    shapes = []
    with fiona.open(OUT_GPKG, layer=OUT_LAYER) as src:
        for feat in src:
            geom = shape(feat["geometry"])
            if not geom.is_empty:
                shapes.append((geom, 1))

    if not shapes:
        raise RuntimeError("No polygons found in GPKG to rasterize.")

    # 3. Rasterize
    log("[Raster] Rasterizing...")
    mask_arr = rasterize(
        shapes=shapes,
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8"
    )

    # Convert to Bool
    mask_bool = mask_arr.astype(bool)

    # 4. Save
    np.save(OUT_MASK_NPY, mask_bool)
    log(f"[Output] Saved boolean mask to: {OUT_MASK_NPY}")

    # Print Stats
    coverage = mask_bool.sum() / mask_bool.size
    log(f"[Stats] Protected Area Coverage: {coverage:.2%} of grid cells.")


# ================= Main Execution =================

if __name__ == "__main__":
    try:
        # Step 1: Create cleaned vector file
        process_vectors()

        # Step 2: Create raster mask
        rasterize_mask()

        log("Processing completed successfully.")

    except Exception as e:
        log(f"Error: {e}")
        import traceback

        traceback.print_exc()