"""Download ERA5-Land data for land-sea mask creation."""

import cdsapi
from pathlib import Path


def download_era5land_mask():
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data/raw/era5-land"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 't2m_era5-land_land-sea-mask.nc'

    client = cdsapi.Client()
    request = {
        "variable": ["2m_temperature"],
        "year": "2000",
        "month": "01",
        "day": ["01"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [55, -15, 25, 22]
    }

    client.retrieve("reanalysis-era5-land", request, str(output_file))


if __name__ == '__main__':
    download_era5land_mask()
