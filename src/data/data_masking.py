"""
Mask creation and application for climate data processing.

This module provides functions for creating, manipulating, and applying
spatial masks to climate datasets in the VAPOR_FX project.
"""
import logging
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Literal, Union

logger = logging.getLogger(__name__)


def create_binary_mask(ds: xr.Dataset,
                       var_name: str,
                       condition: str = 'notnull',
                       mask_name: str = 'mask') -> xr.DataArray:
    """Create a binary mask based on conditions in a dataset.

    Args:
        ds: Dataset to create mask from
        var_name: Variable name to apply condition to
        condition: Condition for masking ('notnull', 'positive', 'nonzero', etc.)
        mask_name: Name for the mask variable

    Returns:
        Binary mask as a DataArray (1 for included points, 0 for excluded)
    """
    if condition == 'notnull':
        mask = xr.where(~ds[var_name].isnull(), 1, 0)
    elif condition == 'positive':
        mask = xr.where(ds[var_name] > 0, 1, 0)
    elif condition == 'nonzero':
        mask = xr.where(ds[var_name] != 0, 1, 0)
    else:
        raise ValueError(f"Unsupported mask condition: {condition}")

    # Remove time dimension if present by taking first time slice
    if 'time' in mask.dims:
        mask = mask.isel(time=0).drop('time')

    # Rename the mask
    mask = mask.rename(mask_name)

    # Add metadata
    mask.attrs['long_name'] = f"Binary mask (1 = included, 0 = excluded)"
    mask.attrs['description'] = f"Created based on '{condition}' condition for {var_name}"

    return mask


def align_mask_to_dataset(mask: xr.DataArray,
                          ds: xr.Dataset,
                          lon_dim: str = 'longitude',
                          lat_dim: str = 'latitude',
                          method: Literal['linear', 'nearest', 'zero', 'slinear',
                                          'quadratic', 'cubic'] = 'nearest') -> xr.DataArray:
    """Align a mask to match the grid of a dataset.

    Args:
        mask: Mask data array to align
        ds: Target dataset to align with
        lon_dim: Name of longitude dimension
        lat_dim: Name of latitude dimension
        method: Interpolation method ('nearest', 'bilinear', etc.)

    Returns:
        Aligned mask as a DataArray
    """
    # Extract coordinates from the target dataset
    target_coords = {
        lon_dim: ds[lon_dim].values,
        lat_dim: ds[lat_dim].values
    }

    # Regrid the mask to match the target coordinates
    aligned_mask = mask.interp(
        **{lon_dim: target_coords[lon_dim],
           lat_dim: target_coords[lat_dim]},
        method=method
    )

    # Check if mask is binary
    mask_values = np.unique(mask.values)
    if set(mask_values).issubset({0, 1}) or set(mask_values).issubset({0.0, 1.0}):
        aligned_mask = xr.where(aligned_mask > 0.5, 1, 0)

    return aligned_mask


def create_mask_from_era5land(input_file: Union[str, Path],
                              output_file: Union[str, Path],
                              resolution: float = 0.25) -> xr.DataArray:
    """Create a land-sea mask from ERA5-Land temperature data.

    Args:
        input_file: Path to ERA5-Land input file
        output_file: Path to save the created mask
        resolution: Target resolution in degrees

    Returns:
        Created mask as a DataArray
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    # Load data
    ds = xr.open_dataset(input_file)
    var_name = list(ds.data_vars)[0]  # Assume first variable is the relevant one

    # Create high-res mask (1 for land, 0 for ocean)
    land_sea_mask = create_binary_mask(ds, var_name, condition='notnull')

    # Get domain extent
    lon_min, lon_max = float(land_sea_mask.longitude.min()), float(land_sea_mask.longitude.max())
    lat_min, lat_max = float(land_sea_mask.latitude.min()), float(land_sea_mask.latitude.max())

    # Create new coordinate arrays at the desired resolution
    new_lon = np.arange(lon_min, lon_max + resolution / 2, resolution)
    new_lat = np.arange(lat_min, lat_max + resolution / 2, resolution)

    # Create target dataset
    target_ds = xr.Dataset(coords={
        'longitude': new_lon,
        'latitude': new_lat
    })

    # Regrid to the target resolution using nearest neighbor
    coarse_mask = align_mask_to_dataset(land_sea_mask, target_ds, method='nearest')

    # Save the mask
    coarse_mask.to_dataset().to_netcdf(output_file)

    # Print statistics
    land_points = int(np.sum(land_sea_mask.values == 1))
    ocean_points = int(np.sum(land_sea_mask.values == 0))
    coarse_land_points = int(np.sum(coarse_mask.values == 1))
    coarse_ocean_points = int(np.sum(coarse_mask.values == 0))

    print(f"Original resolution - Land points: {land_points}")
    print(f"Original resolution - Ocean points: {ocean_points}")
    print(f"{resolution}° resolution - Land points: {coarse_land_points}")
    print(f"{resolution}° resolution - Ocean points: {coarse_ocean_points}")

    return coarse_mask


def load_mask(mask_file: Union[str, Path]) -> xr.DataArray:
    """Load and standardize a land-sea mask.

    Args:
        mask_file: Path to mask file

    Returns:
        Standardized mask data array
    """
    mask_file = Path(mask_file)
    mask_ds = xr.open_dataset(mask_file)
    mask_var = list(mask_ds.data_vars)[0]
    mask = mask_ds[mask_var]

    # Ensure latitude is south-to-north
    if 'latitude' in mask.dims and mask.latitude[0] > mask.latitude[-1]:
        logger.info("Flipping mask latitude to south-to-north ordering")
        mask = mask.reindex(latitude=mask.latitude[::-1])

    return mask
