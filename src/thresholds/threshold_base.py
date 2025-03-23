"""
Base classes for precipitation threshold processing.

This module defines the abstract base class for threshold processing operations
in the VAPOR_FX project, establishing a common interface for different dataset implementations.
"""

import logging
import xarray as xr
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union, Dict

# Set up logging
logger = logging.getLogger(__name__)


class ThresholdProcessor(ABC):
    """Abstract base class for precipitation threshold processing."""

    def __init__(self, data_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        """Initialize the threshold processor.

        Args:
            data_dir: Root directory for processed data
            output_dir: Directory for output files (defaults to data_dir/[dataset]_percentiles)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else None

    @abstractmethod
    def load_data(self, years: List[int], var_name: str, **kwargs) -> xr.DataArray:
        """Load processed precipitation data for the specified years.

        Args:
            years: List of years to process
            var_name: Name of precipitation variable
            **kwargs: Additional dataset-specific parameters

        Returns:
            DataArray containing concatenated precipitation data
        """
        pass

    @abstractmethod
    def calculate_thresholds(self, data: xr.DataArray, percentile: int,
                             wet_day_threshold: float = 1.0, **kwargs) -> xr.DataArray:
        """Calculate percentile thresholds for each grid point using wet days.

        Args:
            data: Input precipitation DataArray
            percentile: Percentile to calculate (e.g., 95)
            wet_day_threshold: Threshold for wet days (default 1.0 mm/day)
            **kwargs: Additional dataset-specific parameters

        Returns:
            DataArray of threshold values per grid point
        """
        pass

    @abstractmethod
    def save_processed_data(self, data: xr.Dataset, output_file: Union[str, Path], **kwargs) -> Path:
        """Save processed data to file.

        Args:
            data: Dataset to save
            output_file: Path to output file
            **kwargs: Additional dataset-specific parameters

        Returns:
            Path to saved file
        """
        pass

    @staticmethod
    def _load_file_data(files: List[str], var_name: str, chunks: Optional[Dict] = None) -> List[xr.DataArray]:
        """Load and process a list of data files."""
        data_arrays = []
        for file in files:
            try:
                da = xr.open_dataset(file, chunks=chunks)[var_name]
                data_arrays.append(da)
            except (FileNotFoundError, KeyError) as e:
                logger.warning(f"Skipping file {file}: {str(e)}")

        if not data_arrays:
            raise ValueError(f"No valid files found in provided file list")

        return data_arrays

    @staticmethod
    def apply_thresholds(data: xr.DataArray, thresholds: xr.DataArray,
                         wet_day_threshold: float = 1.0) -> xr.DataArray:
        """Apply thresholds to precipitation data.

        Args:
            data: Input precipitation DataArray
            thresholds: Threshold values per grid point
            wet_day_threshold: Threshold for wet days (default 1.0 mm/day)

        Returns:
            Threshold-ed precipitation DataArray
        """
        # Create mask of valid (non-NaN) values to preserve ocean/land distinction
        valid_mask = ~np.isnan(data)

        # Set values below wet_day_threshold to 0, but only where data is not NaN
        filtered_data = xr.where(valid_mask & (data >= wet_day_threshold), data, np.nan)

        # Apply calculated thresholds, preserving NaNs in ocean regions
        result = xr.where(valid_mask & (filtered_data > thresholds), filtered_data, np.nan)

        # Convert back to 0 only for land areas below threshold (not oceans)
        result = xr.where(valid_mask & np.isnan(result), 0, result)

        return result

    def process(self, years: List[int], var_name: str, percentile: int = 95,
                wet_day_threshold: float = 1.0, output_file: Optional[Union[str, Path]] = None,
                **kwargs) -> xr.Dataset:
        """Process precipitation data to calculate and apply thresholds.

        Workflow:
            1. Load data for specified years
            2. Calculate percentile thresholds for each grid point
            3. Apply thresholds to precipitation data
            4. Save processed data if output_file is provided

        Args:
            years: List of years to process
            var_name: Name of precipitation variable
            percentile: Percentile threshold (default 95)
            wet_day_threshold: Threshold for wet days (default 1.0 mm/day)
            output_file: Optional path to output file
            **kwargs: Additional dataset-specific parameters

        Returns:
            Dataset with threshold-ed precipitation
        """
        # Load data
        logger.info(f"Loading precipitation data for years {min(years)}-{max(years)}")
        data = self.load_data(years, var_name, **kwargs)

        # Calculate thresholds
        logger.info(f"Calculating {percentile}th percentile thresholds")
        thresholds = self.calculate_thresholds(data, percentile, wet_day_threshold, **kwargs)

        # Apply thresholds
        logger.info("Applying thresholds to precipitation data")
        thresholded_data = self.apply_thresholds(data, thresholds, wet_day_threshold)

        # Standardize output variable name
        standard_var_name = f"rr_r{percentile}p"

        # Create output dataset with standardized variable name
        result_ds = thresholded_data.to_dataset(name=standard_var_name)

        # Set standardized units attribute
        result_ds[standard_var_name].attrs['units'] = 'mm/day'

        # Add metadata
        result_ds.attrs['percentile'] = percentile
        result_ds.attrs['wet_day_threshold'] = wet_day_threshold
        result_ds.attrs['source_variable'] = var_name

        # Save data if output file is provided
        if output_file is not None:
            self.save_processed_data(result_ds, output_file, **kwargs)

        return result_ds
