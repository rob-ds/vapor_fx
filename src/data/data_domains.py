"""
Spatial domain utilities for climate data processing.

This module provides functions for standardizing spatial domains between
datasets in the VAPOR_FX project, finding overlapping regions, and regridding datasets to common grids.
"""

import numpy as np
import xarray as xr
import logging
from typing import List, Literal

logger = logging.getLogger(__name__)


def regrid_to_test(test_ds: xr.Dataset,
                   reference_ds: xr.Dataset,
                   lon_dim: str = 'longitude',
                   lat_dim: str = 'latitude',
                   method: Literal['linear', 'nearest', 'zero', 'slinear',
                                   'quadratic', 'cubic', 'quintic', 'polynomial',
                                   'barycentric', 'krogh', 'pchip', 'spline',
                                   'akima', 'makima'] = 'linear') -> xr.Dataset:
    """Regrid reference dataset (EOBS) to match test dataset (ERA5) grid.

    Args:
        test_ds: Test dataset (ERA5) with the grid to match
        reference_ds: Reference dataset (EOBS) to be regridded
        lon_dim: Name of longitude dimension
        lat_dim: Name of latitude dimension
        method: Interpolation method ('linear', 'nearest', etc.)

    Returns:
        Regridded reference dataset
    """
    # Extract test grid coordinates
    test_coords = {
        lon_dim: test_ds[lon_dim].values,
        lat_dim: test_ds[lat_dim].values
    }

    # Regrid reference to test coordinates
    regridded_ds = reference_ds.interp(**{
        lon_dim: test_coords[lon_dim],
        lat_dim: test_coords[lat_dim]
    }, method=method)

    return regridded_ds


def verify_grid_consistency(datasets: List[xr.Dataset],
                            lon_dim: str = 'longitude',
                            lat_dim: str = 'latitude',
                            tolerance: float = 1e-8) -> bool:
    """Verify that all datasets have exactly the same grid.

    Args:
        datasets: List of datasets to check
        lon_dim: Name of longitude dimension
        lat_dim: Name of latitude dimension
        tolerance: Maximum allowed difference between coordinate values

    Returns:
        True if all grids are consistent

    Raises:
        ValueError: If any dataset has different grid dimensions or coordinates
    """
    if not datasets:
        return True

    reference_lons = datasets[0][lon_dim].values
    reference_lats = datasets[0][lat_dim].values

    for i, ds in enumerate(datasets[1:], 1):
        # Check dimensions match
        if ds[lon_dim].size != reference_lons.size or ds[lat_dim].size != reference_lats.size:
            raise ValueError(f"Dataset {i} has different grid dimensions than reference dataset")

        # Check coordinates match within tolerance
        lon_diff = np.max(np.abs(ds[lon_dim].values - reference_lons))
        lat_diff = np.max(np.abs(ds[lat_dim].values - reference_lats))

        if lon_diff > tolerance or lat_diff > tolerance:
            raise ValueError(f"Dataset {i} has different grid coordinates than reference dataset")

    return True
