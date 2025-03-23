"""
ERA5 data loader for VAPOR_FX precipitation analysis.

This module provides utilities for loading, processing, and standardizing
ERA5 reanalysis precipitation data.
"""

import logging
import xarray as xr
from pathlib import Path
from typing import Optional

from .. import data_base
from .. import data_transformations

logger = logging.getLogger(__name__)


class ERA5Loader(data_base.DataLoader):
    """Loader for ERA5 precipitation data."""

    def __init__(self, data_dir: Path):
        """Initialize the ERA5 data loader.

        Args:
            data_dir: Root directory for data storage
        """
        super().__init__(data_dir)
        self.raw_dir = self.data_dir / 'raw' / 'era5'
        self.processed_dir = self.data_dir / 'processed'
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        for dir_name in ['era5_analysis']:
            (self.processed_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, year: int, domain: Optional[str] = 'clustering') -> xr.Dataset:
        """Load raw ERA5 data for a specific year.

        Args:
            year: Year to load data for
            domain: Domain identifier ('clustering' or 'analysis')

        Returns:
            Dataset containing raw ERA5 data
        """
        # Define domain coordinate ranges
        domain_ranges = {
            'clustering': {
                'latitude': (25, 55),
                'longitude': (-15, 22)
            },
            'analysis': {
                'latitude': (35, 45),
                'longitude': (-5, 12)
            }
        }

        domain_str = domain_ranges[domain]
        lat_range = domain_str['latitude']
        lon_range = domain_str['longitude']

        # Construct filename based on domain
        filename = (
            f"era5_precipitation_{year}_SON_"
            f"{lat_range[0]}N-{lat_range[1]}N_{abs(lon_range[0])}W-{lon_range[1]}E.nc"
        )
        file_path = self.raw_dir / filename

        # Check if file exists
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"ERA5 data file not found: {file_path}")

        # Load dataset
        logger.info(f"Loading ERA5 data from {file_path}")
        ds = xr.open_dataset(file_path)

        # Ensure latitude is sorted south-to-north
        if ds.latitude[0] > ds.latitude[-1]:
            logger.info("Flipping latitude to south-to-north ordering")
            ds = ds.reindex(latitude=ds.latitude[::-1])

        return ds

    def process_data(self, ds: xr.Dataset,
                     var_name: str = 'tp',
                     from_unit: str = 'm/h',
                     to_unit: str = 'mm/day') -> dict:
        """Process ERA5 dataset with standard transformations.

        Args:
            ds: Raw ERA5 dataset
            var_name: Name of precipitation variable
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Processed dataset with aggregated variables
        """
        # Convert units
        logger.info(f"Converting units from {from_unit} to {to_unit}")
        ds = data_transformations.convert_precipitation_units(
            ds, var_name, from_unit, to_unit
        )

        # Create 6-hourly aggregates (364 time steps)
        logger.info("Creating 6-hourly aggregates")
        ds_6h = data_transformations.aggregate_by_time(
            ds, var_name, '6h', 'valid_time', 'mean'
        )

        # Create daily aggregates (91 time steps)
        logger.info("Creating daily aggregates for different starting hours")
        ds_24h = data_transformations.aggregate_daily_by_hour(
            ds, var_name, [0, 6, 12, 18], 'valid_time', '24h'
        )

        # Return separately to maintain different time dimensions
        return {'6h': ds_6h, '24h': ds_24h}

    def load_and_process(self, year: int,
                         mask: Optional[xr.DataArray] = None) -> dict:
        """Load and process ERA5 data for a specific year.

        Args:
            year: Year to process
            mask: Optional land-sea mask to apply

        Returns:
            Processed ERA5 dataset
        """
        logger.info(f"Processing ERA5 data for year {year}")

        # Load raw data
        ds = self.load_raw_data(year)

        # Process data - returns dict with separate timeframes
        processed_datasets = self.process_data(ds)

        # Apply mask to each timeframe if provided
        if mask is not None:
            logger.info("Applying land-sea mask")
            processor = data_base.DataProcessor()
            mask_2d = mask.isel(valid_time=0).drop_vars('valid_time') if 'valid_time' in mask.dims else mask

            for key in processed_datasets:
                vars_to_mask = list(processed_datasets[key].data_vars)
                processed_datasets[key] = processor.apply_mask(
                    processed_datasets[key], mask_2d, vars_to_mask
                )

        return processed_datasets

    def save_processed_data(self, datasets: dict, year: int, data_type: str = 'analysis') -> dict:
        """Save processed ERA5 data with different time dimensions.

        Args:
            datasets: Dictionary containing datasets with different time resolutions
                     (keys are timeframe identifiers like '6h', '24h')
            year: Year of the data
            data_type: Type of processed data ('analysis' or 'clustering')

        Returns:
            Dictionary mapping timeframe identifiers to saved file paths
        """
        output_dir = self.processed_dir / f'era5_{data_type}'
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}
        for timeframe, ds in datasets.items():
            output_file = output_dir / f"era5_{data_type}_{timeframe}_{year}.nc"
            logger.debug(f"Saving {timeframe} ERA5 data to {output_file}")
            ds.to_netcdf(output_file)
            saved_files[timeframe] = output_file

        return saved_files
