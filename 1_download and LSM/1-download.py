#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to batch download ERA5 hourly surface pressure (sp) data by month.
Data is retrieved from the Copernicus Climate Data Store (CDS).
"""

import cdsapi
import os
import time


def download_era5_sp(year, output_dir):
    """
    Downloads ERA5 surface pressure data for a specific year, separated by month.
    """
    # Initialize CDS API client
    # Ensure your .cdsapirc file is configured correctly
    c = cdsapi.Client()

    os.makedirs(output_dir, exist_ok=True)

    # Define temporal resolution
    months = [f"{m:02d}" for m in range(1, 13)]  # Jan to Dec
    days = [f"{d:02d}" for d in range(1, 32)]  # Days 01-31
    hours = [f"{h:02d}:00" for h in range(24)]  # 00:00 to 23:00 UTC

    print(f"Starting download for Year: {year}...")

    for month in months:
        # Define output filename
        filename = f"era5_sp_{year}_{month}.nc"
        out_path = os.path.join(output_dir, filename)

        # Check if file already exists to avoid re-downloading
        if os.path.exists(out_path):
            print(f"[Skipped] File already exists: {out_path}")
            continue

        # Construct the CDS API request
        request_params = {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "surface_pressure",
            ],
            "year": year,
            "month": month,
            "day": days,
            "time": hours,
        }

        # Download with retry logic to handle network interruptions
        max_retries = 3
        attempt = 0
        while True:
            try:
                print(f"[Requesting] {out_path} ...")
                c.retrieve("reanalysis-era5-single-levels", request_params, out_path)
                print(f"[Completed] Downloaded: {out_path}")
                break
            except Exception as e:
                attempt += 1
                print(f"[Warning] specific error: {e}")
                print(f"[Retry] Attempt {attempt}/{max_retries} for {filename}")

                if attempt >= max_retries:
                    print(f"[Error] Failed to download {filename} after {max_retries} attempts.")
                    raise e

                # Wait before retrying
                time.sleep(30)


if __name__ == '__main__':
    TARGET_YEAR = "1980"
    SAVE_DIR = os.path.join("data", "era5", "sp", TARGET_YEAR)

    # Execute download
    download_era5_sp(TARGET_YEAR, SAVE_DIR)