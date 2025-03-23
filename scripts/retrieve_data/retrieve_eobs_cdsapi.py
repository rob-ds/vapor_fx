#!/usr/bin/env python
import cdsapi
import shutil
import zipfile
import argparse
from pathlib import Path


def check_disk_space(required_space_mb=3000):
    """Check if sufficient disk space is available."""
    total, used, free = shutil.disk_usage("..")
    free_mb = free // (2**20)
    if free_mb < required_space_mb:
        raise RuntimeError(f"Insufficient disk space. Need {required_space_mb}MB, have {free_mb}MB")


def extract_zip_file(zip_path, extract_dir):
    """Extract the contents of a zip file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted contents to {extract_dir}")


def download_eobs_data(period, output_dir):
    """Download and extract E-OBS precipitation data for a specific period."""
    client = cdsapi.Client()
    dataset = "insitu-gridded-observations-europe"
    
    request = {
        "product_type": "ensemble_mean",
        "variable": ["precipitation_amount"],
        "grid_resolution": "0_25deg",
        "period": period,
        "version": ["30_0e"],
        "format": "zip"
    }
    
    filename = f"eobs_precipitation_{period}.zip"
    output_file = output_dir / filename
    
    if not output_file.exists():
        client.retrieve(dataset, request, str(output_file))
        print(f"Downloaded {filename}")
        
        # Extract files directly to output directory
        extract_zip_file(output_file, output_dir)
        
        # Remove zip file after successful extraction
        try:
            output_file.unlink()
            print(f"Removed {filename} after extraction")
        except Exception as e:
            print(f"Warning: Could not remove {filename}: {e}")
    else:
        print(f"File {filename} already exists")


def main():
    project_root = Path(__file__).resolve().parents[2]
    default_output = project_root / "data/raw/eobs"
    
    parser = argparse.ArgumentParser(description='Download E-OBS precipitation data')
    parser.add_argument('--output_dir', type=str, default=str(default_output), help='Directory to store E-OBS data')
    args = parser.parse_args()

    check_disk_space()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Warning: {output_dir} doesn't exist")
        return

    periods = ["1950_1964", "1965_1979", "1980_1994", 
               "1995_2010", "2011_2024"]
    
    for period in periods:
        try:
            download_eobs_data(period, output_dir)
        except Exception as e:
            print(f"Error processing {period}: {e}")


if __name__ == "__main__":
    main()
