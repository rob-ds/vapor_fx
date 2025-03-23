#!/usr/bin/env python
import cdsapi
import shutil
import argparse
from pathlib import Path


def check_disk_space(required_space_mb=3000):
    total, used, free = shutil.disk_usage("..")
    free_mb = free // (2**20)
    if free_mb < required_space_mb:
        raise RuntimeError(f"Insufficient disk space. Need {required_space_mb}MB, have {free_mb}MB")


def download_era5_data(year, output_dir):
    client = cdsapi.Client()
    dataset = "reanalysis-era5-single-levels"
    
    north, west, south, east = 55, -15, 25, 22
    
    request = {
        "product_type": ["reanalysis"],
        "variable": ["total_precipitation"],
        "year": [str(year)],
        "month": ["09", "10", "11"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [north, west, south, east],
        "format": "netcdf",
    }
    
    filename = f"era5_precipitation_{year}_SON_{south}N-{north}N_{-west}W-{east}E.nc"
    output_file = output_dir / filename
    
    if not output_file.exists():
        client.retrieve(dataset, request, str(output_file))
        print(f"Downloaded {filename}")
    else:
        print(f"File {filename} already exists, skipping...")


def main():
    # Get project root directory (2 levels up from script location)
    project_root = Path(__file__).resolve().parents[2]
    default_output = project_root / "data/raw/era5"
    
    parser = argparse.ArgumentParser(description='Download ERA5 precipitation data')
    parser.add_argument('--output_dir', type=str, default=str(default_output), help='Directory to store ERA5 data')
    args = parser.parse_args()

    check_disk_space()
    output_dir = Path(args.output_dir)
    
    # Verify it's within project structure
    if not str(output_dir).startswith(str(project_root)):
        raise ValueError(f"Output directory must be within project root: {project_root}")
        
    if not output_dir.exists():
        print(f"Warning: {output_dir} doesn't exist")
        return
    
    for year in range(1950, 2025):
        try:
            download_era5_data(year, output_dir)
        except Exception as e:
            print(f"Error processing {year}: {e}")


if __name__ == "__main__":
    main()
