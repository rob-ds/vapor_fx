"""
Dataset-specific implementations of precipitation threshold processors.

This module provides concrete implementations of the ThresholdProcessor
abstract base class for different precipitation datasets (E-OBS and ERA5).
"""

import logging
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

from .threshold_base import ThresholdProcessor
from .threshold_utils import ensure_directory

# Set up logging
logger = logging.getLogger(__name__)


class EOBSThresholdProcessor(ThresholdProcessor):
    """Processor for E-OBS precipitation threshold calculations."""

    def __init__(self, data_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        """Initialize the E-OBS threshold processor.

        Args:
            data_dir: Root directory for processed data
            output_dir: Directory for output files (defaults to data_dir/eobs_percentiles)
        """
        super().__init__(data_dir, output_dir)
        self.dataset_name = 'eobs'
        self.input_dir = self.data_dir / 'processed' / 'eobs_analysis'
        self.output_dir = self.output_dir or (self.data_dir / 'processed' / 'eobs_percentiles')
        ensure_directory(self.output_dir)

    def load_data(self, years: List[int], var_name: str, chunks: Optional[Dict] = None, **kwargs) -> xr.DataArray:
        """Load processed E-OBS precipitation data for the specified years."""
        # Default chunking for time dimension
        if chunks is None:
            chunks = {'time': -1}

        # Create list of file paths
        files = [str(self.input_dir / f"eobs_analysis_{year}.nc") for year in years]

        # Load and concatenate data
        logger.info(f"Loading {len(files)} E-OBS files for variable '{var_name}'")

        data_arrays = self._load_file_data(files, var_name, chunks)

        # Concatenate all data arrays along the time dimension
        return xr.concat(data_arrays, dim='time')

    def calculate_thresholds(self, data: xr.DataArray, percentile: int,
                             wet_day_threshold: float = 1.0, **kwargs) -> xr.DataArray:
        """Calculate percentile thresholds for each grid point using wet days."""
        # Get wet days (≥wet_day_threshold mm/day)
        wet_days = data.where(data >= wet_day_threshold, 0)

        # Fill NaN values with 0 before calculating percentile
        wet_days = wet_days.fillna(0)

        # Calculate percentile per grid point
        thresholds = wet_days.quantile(percentile / 100, dim='time')

        return thresholds

    def save_processed_data(self, data: xr.Dataset, output_file: Union[str, Path], **kwargs) -> Path:
        """Save processed E-OBS data with standardized format."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Force single precision for all data variables
        encoding = {}
        for var in data.data_vars:
            encoding[var] = {
                'dtype': 'float32',  # Single precision
                'zlib': True,  # Enable compression
                'complevel': 5,  # Compression level
                '_FillValue': np.nan  # Standard fill value
            }

        # Standardize time dimension if needed
        if 'time' in data.dims and kwargs.get('standardize_dims', True):
            # E-OBS already uses 'time', but we could rename if needed
            pass

        # print(f"Saving E-OBS data to {output_path} with encoding: {encoding}")
        data.to_netcdf(output_path, encoding=encoding)

        return output_path


class ERA5ThresholdProcessor(ThresholdProcessor):
    """Processor for ERA5 precipitation threshold calculations."""

    def __init__(self, data_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        """Initialize the ERA5 threshold processor."""
        super().__init__(data_dir, output_dir)
        self.dataset_name = 'era5'
        self.input_dir = self.data_dir / 'processed' / 'era5_analysis'
        self.output_dir = self.output_dir or (self.data_dir / 'processed' / 'era5_percentiles')
        ensure_directory(self.output_dir)

    def load_data(self, years: List[int], var_name: str, chunks: Optional[Dict] = None, **kwargs) -> xr.DataArray:
        """Load ERA5 data with simplified dimension handling."""
        # Set file pattern based on variable type
        file_pattern = "era5_analysis_24h_{year}.nc" if var_name.startswith('tp_24h') else "era5_analysis_6h_{year}.nc"

        # Build file list
        files = []
        for year in years:
            file_path = self.input_dir / file_pattern.format(year=year)
            if file_path.exists():
                files.append(str(file_path))

        if not files:
            raise ValueError(f"No files found for years {years}")

        # Determine which time dimension to use
        with xr.open_dataset(files[0]) as test_ds:
            # Explicitly identify the time dimension to use
            time_dims = []
            if 'valid_time' in test_ds.dims and test_ds.sizes['valid_time'] > 1:
                time_dims.append('valid_time')
            if 'valid_time_24h' in test_ds.dims and test_ds.sizes['valid_time_24h'] > 1:
                time_dims.append('valid_time_24h')

            if not time_dims:
                raise ValueError(f"No valid time dimension found in {files[0]}")

            # Choose the appropriate time dimension
            time_dim = time_dims[0]  # Use the first valid time dimension

        # Load and concatenate the data using the identified time dimension
        data_list = []
        for file in files:
            data = xr.open_dataset(file)[var_name]
            data_list.append(data)

        # Concatenate along the identified time dimension
        result = xr.concat(data_list, dim=time_dim)

        return result

    def calculate_thresholds(self, data: xr.DataArray, percentile: int,
                             wet_day_threshold: float = 1.0, **kwargs) -> xr.DataArray:
        """Calculate percentile thresholds for each grid point using wet days."""
        # Determine time dimension (with type checking)
        time_dims = [d for d in data.dims if isinstance(d, str) and d.startswith('valid_time')]
        if not time_dims:
            raise ValueError(f"No valid time dimension found in data. Available dimensions: {list(data.dims)}")
        time_dim = time_dims[0]

        # Get wet days (≥wet_day_threshold mm/day)
        wet_days = data.where(data >= wet_day_threshold, 0)

        # Fill NaN values with 0 before calculating percentile
        wet_days = wet_days.fillna(0)

        # Calculate percentile per grid point
        thresholds = wet_days.quantile(percentile / 100, dim=time_dim)

        return thresholds

    def save_processed_data(self, data: xr.Dataset, output_file: Union[str, Path], **kwargs) -> Path:
        """Save processed ERA5 data with standardized format."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Force single precision for all data variables
        encoding = {}
        for var in data.data_vars:
            encoding[var] = {
                'dtype': 'float32',  # Single precision
                'zlib': True,  # Enable compression
                'complevel': 5,  # Compression level (higher = smaller file)
                '_FillValue': np.nan  # Standard fill value
            }

        # Standardize time dimension if needed
        if kwargs.get('standardize_dims', True):
            if 'valid_time_24h' in data.dims:
                data = data.rename({'valid_time_24h': 'time'})
            elif 'valid_time' in data.dims:
                data = data.rename({'valid_time': 'time'})

        # print(f"Saving ERA5 data to {output_path} with encoding: {encoding}")
        data.to_netcdf(output_path, encoding=encoding)

        return output_path
