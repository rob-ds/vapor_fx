"""
E-OBS data loader for VAPOR_FX precipitation analysis.

This module provides utilities for loading, processing, and standardizing
E-OBS observational precipitation data.
"""

import logging
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, List

from .. import data_base
from .. import data_transformations

logger = logging.getLogger(__name__)


class EOBSLoader(data_base.DataLoader):
    """Loader for E-OBS precipitation data."""

    def __init__(self, data_dir: Path):
        """Initialize the E-OBS data loader.

        Args:
            data_dir: Root directory for data storage
        """
        super().__init__(data_dir)
        self.raw_dir = self.data_dir / 'raw' / 'eobs'
        self.processed_dir = self.data_dir / 'processed'
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for dir_name in ['eobs_analysis']:
            (self.processed_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, period_range: Tuple[int, int],
                      version: str = 'v30.0e') -> xr.Dataset:
        """Load raw E-OBS data for a specific period range.

        Args:
            period_range: Tuple of (start_year, end_year)
            version: E-OBS dataset version

        Returns:
            Dataset containing raw E-OBS data
        """
        start_year, end_year = period_range

        # Construct filename based on period range and version
        filename = f"rr_ens_mean_0.25deg_reg_{start_year}-{end_year}_{version}.nc"
        file_path = self.raw_dir / filename

        # Check if file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"E-OBS data file not found: {file_path}")

        # Load dataset
        logger.info(f"Loading E-OBS data from {file_path}")
        ds = xr.open_dataset(file_path)

        return ds

    def process_data(self, ds: xr.Dataset,
                     var_name: str = 'rr',
                     season: Optional[str] = 'SON') -> xr.Dataset:
        """Process E-OBS dataset with standard transformations.

        Args:
            ds: Raw E-OBS dataset
            var_name: Name of precipitation variable
            season: Season to extract (None for all data)

        Returns:
            Processed dataset
        """
        # Extract specified season if requested
        if season:
            logger.info(f"Extracting {season} season")
            ds = data_transformations.extract_season(ds, season)

        return ds

    @staticmethod
    def extract_year(ds: xr.Dataset, year: int, time_dim: str = 'time') -> xr.Dataset:
        """Extract data for a specific year from a multi-year dataset.

        Args:
            ds: Multi-year dataset
            year: Year to extract
            time_dim: Name of time dimension

        Returns:
            Dataset for the specified year only
        """
        logger.info(f"Extracting data for year {year}")
        return ds.sel({time_dim: ds[time_dim].dt.year == year})

    def load_and_process(self, year: int,
                         period_ranges: List[Tuple[int, int]],
                         mask: Optional[xr.DataArray] = None) -> xr.Dataset:
        """Load and process E-OBS data for a specific year.

        Args:
            year: Year to process
            period_ranges: List of period ranges (start_year, end_year)
            mask: Optional land-sea mask to apply

        Returns:
            Processed E-OBS dataset for specified year
        """
        logger.info(f"Processing E-OBS data for year {year}")

        # Find appropriate period range containing the target year
        period_range = next((pr for pr in period_ranges if pr[0] <= year <= pr[1]), None)

        if period_range is None:
            logger.error(f"Year {year} not found in any provided period range")
            raise ValueError(f"Year {year} not found in any provided period range")

        # Load raw data for the entire period
        ds = self.load_raw_data(period_range)

        # Process data (extract season, fill NaNs)
        processed_ds = self.process_data(ds)

        # Extract the target year only
        year_ds = self.__class__.extract_year(processed_ds, year)

        # Apply mask if provided
        if mask is not None:
            logger.info("Applying land-sea mask")
            processor = data_base.DataProcessor()
            year_ds = processor.apply_mask(year_ds, mask, ['rr'])

        return year_ds

    def save_processed_data(self, ds: xr.Dataset, year: int) -> Path:
        """Save processed E-OBS data.

        Args:
            ds: Processed dataset to save
            year: Year of the data

        Returns:
            Path to saved file
        """
        output_dir = self.processed_dir / 'eobs_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"eobs_analysis_{year}.nc"
        logger.debug(f"Saving processed E-OBS data to {output_file}")

        ds.to_netcdf(output_file)
        return output_file
