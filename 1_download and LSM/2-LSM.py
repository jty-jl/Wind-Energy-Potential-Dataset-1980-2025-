#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to split ERA5 NetCDF files into land and ocean components based on a land-sea mask.
- Input: Hourly ERA5 NetCDF files.
- Mask: A static Land-Sea Mask (LSM) file (threshold: >= 0.5 for land).
- Output: Two separate NetCDF files for land and ocean data.
"""

import os
import glob
import numpy as np
import xarray as xr
from tqdm import tqdm

# ================= Configuration =================

VAR_NAME = "sp"
TARGET_YEAR = "1980"

# Paths
BASE_DIR = os.path.join("data", VAR_NAME, TARGET_YEAR)
OUT_LAND_DIR = os.path.join(BASE_DIR, "land")
OUT_OCEAN_DIR = os.path.join(BASE_DIR, "ocean")
LSE_PATH = os.path.join("data", "masks", "lse2024.nc")

# Variable name for land-sea mask in the NetCDF file
LSM_VAR_NAME = "lsm"

# Threshold (>= 0.5 is land)
THRESHOLD = 0.5

COMPRESSION = dict(zlib=True, complevel=4, dtype="float32", _FillValue=np.nan)


# =================================================

def _to_latlon_names(ds: xr.Dataset) -> xr.Dataset:
    """Standardizes coordinate names to 'latitude' and 'longitude'."""
    rename_map = {}
    if "lat" in ds.coords or "lat" in ds.dims:
        rename_map["lat"] = "latitude"
    if "lon" in ds.coords or "lon" in ds.dims:
        rename_map["lon"] = "longitude"
    if rename_map:
        ds = ds.rename(rename_map)
    return ds


def load_land_ocean_mask(lse_path: str, var_name: str, threshold: float) -> tuple[xr.DataArray, xr.DataArray]:
    """Loads the mask file and returns boolean masks for land and ocean."""
    lse = xr.open_dataset(lse_path)
    lse = _to_latlon_names(lse)

    if var_name not in lse.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in mask file.")

    lsm = lse[var_name]

    if 'time' in lsm.dims:
        lsm = lsm.isel(time=0)
    if 'valid_time' in lsm.dims:
        lsm = lsm.isel(valid_time=0)

    land_mask = (lsm >= threshold).astype(bool)
    ocean_mask = (lsm < threshold).astype(bool)
    return land_mask, ocean_mask


def _align_mask_to_data(mask: xr.DataArray, data_ds: xr.Dataset) -> xr.DataArray:
    """Aligns the mask grid to the data grid using nearest-neighbor interpolation."""
    mask = _to_latlon_names(mask.to_dataset(name="mask"))["mask"]
    data_ds = _to_latlon_names(data_ds)

    same_lat = (
            "latitude" in data_ds.coords
            and mask.sizes.get("latitude") == data_ds.sizes.get("latitude")
            and np.array_equal(mask["latitude"].values, data_ds["latitude"].values)
    )
    same_lon = (
            "longitude" in data_ds.coords
            and mask.sizes.get("longitude") == data_ds.sizes.get("longitude")
            and np.array_equal(mask["longitude"].values, data_ds["longitude"].values)
    )
    if same_lat and same_lon:
        return mask

    return mask.interp(
        latitude=data_ds["latitude"],
        longitude=data_ds["longitude"],
        method="nearest",
    )


def _build_encoding(var_name: str) -> dict:
    return {var_name: COMPRESSION.copy()}


def split_and_save_single_file(nc_path: str,
                               var_name: str,
                               land_mask: xr.DataArray,
                               ocean_mask: xr.DataArray,
                               out_dir_land: str,
                               out_dir_ocean: str):
    ds = xr.open_dataset(nc_path)
    ds = _to_latlon_names(ds)

    if var_name not in ds.data_vars:
        raise KeyError(f"Variable '{var_name}' not found in {os.path.basename(nc_path)}.")

    land_m = _align_mask_to_data(land_mask, ds)
    ocean_m = _align_mask_to_data(ocean_mask, ds)

    da = ds[var_name].astype("float32")

    if 'time' in da.dims and 'time' not in land_m.dims:
        land_m = land_m.expand_dims(time=da.time)
        ocean_m = ocean_m.expand_dims(time=da.time)
    if 'valid_time' in da.dims and 'valid_time' not in land_m.dims:
        land_m = land_m.expand_dims(valid_time=da.valid_time)
        ocean_m = ocean_m.expand_dims(valid_time=da.valid_time)

    # Apply masking
    da_land = da.where(land_m)
    da_ocean = da.where(ocean_m)

    # Create datasets
    ds_land = xr.Dataset({var_name: da_land})
    ds_ocean = xr.Dataset({var_name: da_ocean})

    # Update attributes
    ds_land[var_name].attrs = dict(ds[var_name].attrs)
    ds_ocean[var_name].attrs = dict(ds[var_name].attrs)

    for coord in ds.coords:
        if coord in ds_land.coords:
            ds_land[coord].attrs = dict(ds[coord].attrs)
        if coord in ds_ocean.coords:
            ds_ocean[coord].attrs = dict(ds[coord].attrs)

    # Output paths
    base = os.path.basename(nc_path)
    root, ext = os.path.splitext(base)
    out_land = os.path.join(out_dir_land, f"{root}_land{ext}")
    out_ocean = os.path.join(out_dir_ocean, f"{root}_ocean{ext}")

    os.makedirs(out_dir_land, exist_ok=True)
    os.makedirs(out_dir_ocean, exist_ok=True)

    # Save
    enc = _build_encoding(var_name)
    ds_land.to_netcdf(out_land, format="NETCDF4", encoding=enc)
    ds_ocean.to_netcdf(out_ocean, format="NETCDF4", encoding=enc)

    ds.close()
    ds_land.close()
    ds_ocean.close()


def process_folder(var_name: str,
                   in_dir: str,
                   target_year: str,
                   land_mask: xr.DataArray,
                   ocean_mask: xr.DataArray,
                   out_dir_land: str,
                   out_dir_ocean: str):
    pattern = os.path.join(in_dir, f"era5_{var_name}_{target_year}_*.nc")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No matching files found in {in_dir}")

    print(f"[{var_name}] Files to process: {len(files)}")
    for nc in tqdm(files, desc=f"Processing {var_name}"):
        split_and_save_single_file(nc, var_name, land_mask, ocean_mask, out_dir_land, out_dir_ocean)


def main():
    print("Loading Land-Sea Mask...")
    land_mask, ocean_mask = load_land_ocean_mask(LSE_PATH, LSM_VAR_NAME, THRESHOLD)

    process_folder(
        var_name=VAR_NAME,
        in_dir=BASE_DIR,
        target_year=TARGET_YEAR,
        land_mask=land_mask,
        ocean_mask=ocean_mask,
        out_dir_land=OUT_LAND_DIR,
        out_dir_ocean=OUT_OCEAN_DIR
    )
    print("Processing completed successfully.")


if __name__ == "__main__":
    main()