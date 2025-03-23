"""Create land-sea mask from ERA5-Land temperature data with 0.25 degree resolution."""

import xarray as xr
import numpy as np
from pathlib import Path


def create_mask():
    """Create binary land-sea mask from ERA5-Land data at 0.25 degree resolution."""
    project_root = Path(__file__).resolve().parents[2]
    input_file = project_root / 'data/raw/era5-land/t2m_era5-land_land-sea-mask.nc'
    output_file = project_root / 'data/raw/era5-land/land_sea_mask_0.25deg.nc'

    # Load data
    ds = xr.open_dataset(input_file)

    # Create high-res mask (1 for land, 0 for ocean)
    land_sea_mask = (~ds.t2m.isnull()).astype(int).rename('mask')

    # Create 0.25 degree coordinate arrays
    new_lon = np.arange(-15, 22.25, 0.25)
    new_lat = np.arange(25, 55.25, 0.25)

    # Regrid to 0.25 degrees using nearest neighbor
    coarse_mask = land_sea_mask.interp(
        longitude=new_lon,
        latitude=new_lat,
        method='nearest'
    )

    # Print statistics
    print(f"Original resolution - Land points: {(land_sea_mask == 1).sum().values}")
    print(f"Original resolution - Ocean points: {(land_sea_mask == 0).sum().values}")
    print(f"0.25° resolution - Land points: {(coarse_mask == 1).sum().values}")
    print(f"0.25° resolution - Ocean points: {(coarse_mask == 0).sum().values}")

    # Save mask
    coarse_mask.to_netcdf(output_file)
    print(f"Mask saved to: {output_file}")


if __name__ == '__main__':
    create_mask()
