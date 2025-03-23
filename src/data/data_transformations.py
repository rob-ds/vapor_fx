"""
Data transformation utilities for climate data processing.

This module provides functions for common transformations applied to climate datasets
in the VAPOR_FX project, including temporal aggregation, unit conversion, and data standardization.
"""

import xarray as xr
from typing import List


def convert_precipitation_units(ds: xr.Dataset,
                                var_name: str,
                                from_unit: str,
                                to_unit: str) -> xr.Dataset:
    """Convert precipitation data between different units.

    Supported unit conversions:
    - m/h to mm/day
    - m/s to mm/day
    - mm/day to m/s

    Args:
        ds: Dataset containing precipitation data
        var_name: Name of the precipitation variable
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Dataset with converted precipitation units
    """
    result_ds = ds.copy()

    # Handle different unit conversions
    if from_unit == 'm/h' and to_unit == 'mm/day':
        # m/h -> mm/day: × 1000 (m->mm) × 24 (h->day)
        result_ds[var_name] = ds[var_name] * 1000 * 24
    elif from_unit == 'm/s' and to_unit == 'mm/day':
        # m/s -> mm/day: × 1000 (m->mm) × 86400 (s->day)
        result_ds[var_name] = ds[var_name] * 1000 * 86400
    elif from_unit == 'mm/day' and to_unit == 'm/s':
        # mm/day -> m/s: ÷ 1000 (mm->m) ÷ 86400 (day->s)
        result_ds[var_name] = ds[var_name] / 1000 / 86400
    else:
        raise ValueError(f"Unsupported unit conversion from {from_unit} to {to_unit}")

    # Update attributes to reflect new units
    if 'units' in result_ds[var_name].attrs:
        result_ds[var_name].attrs['units'] = to_unit
    else:
        result_ds[var_name].attrs = {'units': to_unit}

    return result_ds


def aggregate_by_time(ds: xr.Dataset,
                      var_name: str,
                      freq: str,
                      time_dim: str = 'time',
                      method: str = 'mean') -> xr.Dataset:
    """Aggregate dataset by time frequency.

    Args:
        ds: Dataset to aggregate
        var_name: Variable name to aggregate and rename
        freq: Pandas-compatible frequency string ('6h', '24h', etc.)
        time_dim: Name of time dimension
        method: Aggregation method ('mean', 'sum', 'max', 'min')

    Returns:
        Time-aggregated dataset
    """
    # Create new variable name
    new_var_name = f"{var_name}_{freq}"

    # Apply the resampling operation based on method
    if method == 'mean':
        resampled_var = ds[var_name].resample({time_dim: freq}).mean()
    elif method == 'sum':
        resampled_var = ds[var_name].resample({time_dim: freq}).sum()
    elif method == 'max':
        resampled_var = ds[var_name].resample({time_dim: freq}).max()
    elif method == 'min':
        resampled_var = ds[var_name].resample({time_dim: freq}).min()
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")

    # Create a new dataset with the resampled variable
    result_ds = xr.Dataset({new_var_name: resampled_var})

    # Preserve attributes if available
    if hasattr(ds[var_name], 'attrs'):
        result_ds[new_var_name].attrs = ds[var_name].attrs.copy()

    return result_ds


def aggregate_daily_by_hour(ds: xr.Dataset,
                            var_name: str,
                            base_hours: List[int],
                            time_dim: str = 'time',
                            freq: str = '24h') -> xr.Dataset:
    """Create daily aggregates with different starting hours.

    Args:
        ds: Dataset with sub-daily time resolution
        var_name: Variable name to aggregate
        base_hours: List of starting hours for daily aggregation (e.g., [0, 6, 12, 18])
        time_dim: Name of time dimension
        freq: Frequency string ('24h', 'D', etc.)

    Returns:
        Dataset containing daily aggregates with different starting hours
    """
    # Create result dataset
    result_ds = xr.Dataset()

    # Process each hour separately using time shifting
    for hour in base_hours:
        hour_str = str(hour).zfill(2)
        var_name_daily = f"{var_name}_{freq}_{hour_str}"

        # Shift time first, then resample
        if hour == 0:
            daily_values = ds[var_name].resample({time_dim: '24h'}, closed='left').mean()
        else:
            # Shift by the specified hours before resampling
            daily_values = ds[var_name].shift({time_dim: hour}).resample({time_dim: '24h'}, closed='left').mean()

        # Add to result dataset
        result_ds[var_name_daily] = daily_values

    return result_ds


def extract_season(ds: xr.Dataset,
                   season: str,
                   time_dim: str = 'time') -> xr.Dataset:
    """Extract a specific season from a dataset."""
    # First check if the provided time_dim actually exists
    if time_dim not in ds.dims:
        # Try to autodetect time dimension
        time_dims = [dim for dim in ds.dims if str(dim).lower() in
                     ('time', 'valid_time', 't', 'datetime')]
        if not time_dims:
            raise ValueError(f"No time dimension found in dataset. Available dimensions: {list(ds.dims)}")
        time_dim = time_dims[0]

    season_months = {
        'JFM': [1, 2, 3],
        'FMA': [2, 3, 4],
        'MAM': [3, 4, 5],
        'AMJ': [4, 5, 6],
        'MJJ': [5, 6, 7],
        'JJA': [6, 7, 8],
        'JAS': [7, 8, 9],
        'ASO': [8, 9, 10],
        'SON': [9, 10, 11],
        'OND': [10, 11, 12],
        'NDJ': [11, 12, 1],
        'DJF': [12, 1, 2]
    }

    if season not in season_months:
        raise ValueError(f"Unknown season: {season}. Must be one of {list(season_months.keys())}")

    # Filter by month
    months = season_months[season]
    season_ds = ds.sel(**{time_dim: ds[time_dim].dt.month.isin(months)})

    return season_ds
