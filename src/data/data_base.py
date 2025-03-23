"""
Base classes and interfaces for climate data processing.

This module defines the foundational abstractions for data loading and processing
used throughout the VAPOR_FX project.
"""

import abc
import logging
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader(abc.ABC):
    """Abstract base class for climate dataset loaders."""

    def __init__(self, data_dir: Path):
        """Initialize the data loader.

        Args:
            data_dir: Root directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        pass

    @abc.abstractmethod
    def load_raw_data(self, year: int, **kwargs) -> xr.Dataset:
        """Load raw data for a specific year.

        Args:
            year: The year to load data for
            **kwargs: Additional loading parameters

        Returns:
            Dataset containing raw data
        """
        pass

    @abc.abstractmethod
    def process_data(self, ds: xr.Dataset, **kwargs) -> xr.Dataset:
        """Process a dataset according to dataset-specific requirements.

        Args:
            ds: Dataset to process
            **kwargs: Additional processing parameters

        Returns:
            Processed dataset
        """
        pass


class DataProcessor:
    """Base class for data processing operations."""

    @staticmethod
    def extract_domain(ds: xr.Dataset,
                       lon_range: Tuple[float, float],
                       lat_range: Tuple[float, float],
                       lon_dim: str = 'longitude',
                       lat_dim: str = 'latitude') -> xr.Dataset:
        """Extract a spatial domain from a dataset.

        Args:
            ds: Source dataset
            lon_range: (min_lon, max_lon) tuple
            lat_range: (min_lat, max_lat) tuple
            lon_dim: Name of longitude dimension
            lat_dim: Name of latitude dimension

        Returns:
            Dataset subset for the specified domain
        """
        subset = ds.sel(
            **{lon_dim: slice(lon_range[0], lon_range[1]),
               lat_dim: slice(lat_range[0], lat_range[1])}
        )
        return subset

    @staticmethod
    def apply_mask(ds: xr.Dataset,
                   mask: xr.DataArray,
                   vars_to_mask: Optional[List[str]] = None) -> xr.Dataset:
        """Apply a spatial mask to dataset variables."""
        masked_ds = ds.copy()

        # Remove time dimension from mask if present
        if 'valid_time' in mask.dims:
            mask = mask.squeeze('valid_time')  # Remove singleton dimension
        elif 'time' in mask.dims:
            mask = mask.squeeze('time')

        # Ensure mask is binary
        binary_mask = xr.where(mask > 0.5, 1, 0)

        for var in vars_to_mask:
            if var in ds:
                # Get variable data
                var_data = ds[var]

                # Apply mask with proper broadcasting
                # This handles time dimensions correctly
                masked_ds[var] = var_data.where(binary_mask)

        return masked_ds
