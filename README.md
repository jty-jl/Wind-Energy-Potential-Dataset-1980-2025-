# High-resolution global onshore and offshore wind energy potential dataset

**Changqing Xu**<sup>a,b</sup>, **Tianyu Jia**<sup>a,b,&#42;</sup>, **Jianchuan Qi**<sup>c,&#42;</sup>, **Peng Wang**<sup>d,e</sup>, **Xi Chen**<sup>a</sup>, **Siqi Wang**<sup>a</sup>, **Yunzhi Tang**<sup>a</sup>, **Yuqiao Lan**<sup>a,b</sup>, **Jing Guo**<sup>f</sup>, **Bo Wang**<sup>b,g</sup>, **Bin Zhang**<sup>b,g</sup>, **Chuke Chen**<sup>c</sup>, **Nan Li**<sup>h</sup>, **Ming Xu**<sup>h,&#42;</sup>, **Zhaohua Wang**<sup>a,b,&#42;</sup>

<br>

<small>
<sup>a</sup> School of Economics, Beijing Institute of Technology, Beijing 100081, China.<br>
<sup>b</sup> Digital Economy and Policy Intelligentization Key Laboratory of Ministry of Industry and Information Technology, Beijing 100081, China.<br>
<sup>c</sup> School of Environment, Tsinghua University, Beijing 100084, China.<br>
<sup>d</sup> Key Lab of Urban Environment and Health, Institute of Urban Environment, Chinese Academy of Sciences, Xiamen 361021, China.<br>
<sup>e</sup> University of Chinese Academy of Sciences, Beijing 100049, China.<br>
<sup>f</sup> College of Management Science and Engineering, Beijing Information Science & Technology University, Beijing, 102206, P.R. China.<br>
<sup>g</sup> School of Management, Beijing Institute of Technology, Beijing 100081, China.<br>
<sup>h</sup> State Key Laboratory of Iron and Steel Industry Environmental Protection, School of Environment, Tsinghua University, Beijing 100084, China.
</small>

<br>
<small>* Corresponding authors.</small>


## Abstract
This repository contains the source code and processing pipeline for the manuscript "High-resolution global onshore and offshore wind energy potential dataset". We developed a harmonized geospatial assessment framework to estimate hourly wind power output, capacity factors (CF), and aggregated energy potentials—including Capacity Density (CD), Energy Density (ED), Technical Potential (TP), and Realistic Potential (RP)—for both onshore and offshore environments globally. Multi-variate validation methods were adopted to demonstrate the robustness of the dataset. The provided scripts allow for the complete reproduction of the global wind energy assessment and spatial validation presented in the study.

## Project Data & Availability
Due to the size of high-resolution hourly reanalysis data, the raw input files are not hosted in this repository. Users must obtain the raw data from the official sources listed in the table below.
### Data Sources

| Dataset | Source / Link | Variables Used |
| :--- | :--- | :--- |
| **ERA5 Reanalysis** | [ECMWF CDS](https://cds.climate.copernicus.eu/datasets) | `100u`, `100v` (Wind)<br>`sp`, `t2m` (Density) <br> `Land-Sea Mask (LSM)` <br> `Sea ice cover`|
| **MODIS MCD12Q1(v6.1)** | [Land Use and Land Cover](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd12c1-061) | Excluded: Evergreen Needleleaf Forests (1), Evergreen Broadleaf Forests (2), Deciduous Needleleaf Forests (3), Deciduous Broadleaf Forests (4), Mixed Forests (5), Permanent Wetlands (11), Urban and Built-up Lands (13), Snow/Ice (15), Water Bodies (17) <br> Retained: Closed Shrublands (6), Open Shrublands (7), Woody Savannas (8), Savannas (9), Grasslands (10), Croplands (12), Cropland/Natural Vegetation Mosaics (14), Barren (16) |
| **EEZ** | [Exclusive Economic Zones](https://www.marineregions.org/) | Assessment restricted to maritime areas within national EEZs (up to 200 nautical miles) |
| **WDPA** | [Protected Areas](https://www.protectedplanet.net/en/thematic-areas/wdpa%20) | Exclusion of all protected areas with designated or established status in the following IUCN categories: Strict Nature Reserve, Wilderness Area, National Park, Natural Monument/Feature, Habitat/Species Management Area, Protected Landscape/Seascape, Protected Area with Sustainable Resource Use, Indigenous and Community Conserved Area, Drinking-Water Protection Zones, Permanent No-Take Fisheries Zones, Military Exclusion Areas, Long-Term Private Conservation Easements |
| **GEBCO Grid** | [GEBCO](https://www.gebco.net/data-products-gridded-bathymetry-data/gebco2025-grid) | Slope / Elevation / Depth / Distance |
| **Global Wind Atlas** | [GWA 3.0](https://globalwindatlas.info/) | Spatial distribution validation |
| **ESTON-E** | [European Network of Transmission System Operators for Electricity](https://transparency.entsoe.eu/) | High-frequency time series validation |

## Directory Structure
To ensure the scripts run successfully, please organize your local data directory as follows (using relative paths):
```text
.
├── code/                      # Python scripts
├── figures/                   # Output figures
└── data/                      # Data Storage
    ├── download/              # [Common Step 1] Raw ERA5 Data
    │   ├── u100/
    │   ├── v100/
    │   ├── sp/
    │   └── t2m/
    ├── ancillary/             # Static GIS Data for Masking
    │   ├── LSM/               # [Common Step 2] Land-Sea Mask
    │   ├── MCD121/            # [Onshore] Land Cover
    │   ├── WDPA/              # [Common] Protected Areas
    │   ├── GEBCO/             # [Common] Slope / Elevation / Depth / Distance
    │   ├── EEZ/               # [Offshore] Exclusive Economic Zones
    │   └── Sea_Ice/           # [Offshore] Sea Ice extent
    ├── onshore/               # [Onshore Branch]
    │   ├── wind_rho/          # Processed V100 & Rho
    │   ├── Pwind/
    │   ├── CF/
    │   └── Potential/         # CD / ED / TP / RP
    ├── offshore/              # [Offshore Branch]
    │   ├── wind_rho/          # Processed V100 & Rho
    │   ├── Pwind/
    │   ├── CF/
    │   └── Potential/         # CD / ED / TP / RP
    └── Validation/            # GWA / ESTON-E / Other literatures
```

## Reproduce My Work
All code used to create, process, and validate the data in this publication can be reproduced using the provided Python scripts.

### Prerequisites
These scripts require Python 3.8+. You can install the necessary dependencies (including cdsapi for data download) using pip:
```text
pip install numpy xarray pandas netCDF4 scipy matplotlib cartopy rioxarray cdsapi
```
### ERA5 API Configuration
To run the download scripts in Phase 1, you must configure the Climate Data Store (CDS) API client:
1. Register: Create an account at the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu).
2. Get Credentials: Log in and visit your `User Profile` to copy your UID and API Key.
3. Create Config File: Create a file named `.cdsapirc` in your user home directory (Linux/Mac: `~/.cdsapirc`, Windows: `C:\Users\Username\.cdsapirc`) containing:
```text
url: https://cds.climate.copernicus.eu/api/v2
key: {UID}:{API_KEY}
```
