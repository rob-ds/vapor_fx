"""
Analysis data processing for the VAPOR_FX project.

This module contains functions for processing precipitation data specifically
for analysis purposes, including standardization, domain extraction, and data preparation.
"""

import xarray as xr
import numpy as np
import logging
from typing import Optional, Tuple, Dict

from .. import data_base, data_domains, data_transformations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnalysisProcessor:
    """Process climate data for analysis purposes."""

    def __init__(self,
                 analysis_domain: Optional[Dict[str, Tuple[float, float]]] = None,
                 apply_mask: bool = True):
        """Initialize the analysis processor.

        Args:
            analysis_domain: Optional fixed analysis domain dictionary with
                            'longitude' and 'latitude' keys containing (min, max) tuples
            apply_mask: Whether to apply land-sea masking
        """
        self.analysis_domain = analysis_domain
        self.apply_mask = apply_mask

    def process_datasets(self,
                         reference_ds: xr.Dataset,
                         test_ds: xr.Dataset,
                         mask: Optional[xr.DataArray] = None,
                         season: Optional[str] = 'SON') -> Tuple[xr.Dataset, xr.Dataset, Optional[xr.DataArray]]:
        """Process reference and test datasets for analysis.

        Args:
            reference_ds: Reference dataset (EOBS)
            test_ds: Test dataset (ERA5)
            mask: Optional land-sea mask
            season: Season to extract (None to skip seasonal extraction)

        Returns:
            Tuple of (processed_reference_ds, processed_test_ds, domain_mask)
        """
        # Extract season if specified
        if season:
            logger.info(f"Extracting {season} season")
            ref_time_dim = next((d for d in reference_ds.dims if str(d).lower() in ('time', 't')), 'time')
            test_time_dim = next((d for d in test_ds.dims if str(d).lower() in ('valid_time', 'time', 't')), 'time')

            reference_ds = data_transformations.extract_season(reference_ds, season, time_dim=ref_time_dim)
            test_ds = data_transformations.extract_season(test_ds, season, time_dim=test_time_dim)

        # Fill NaNs with zeros in reference dataset (EOBS) before regridding
        logger.info("Filling NaNs with zeros in reference dataset before regridding")
        reference_ds = reference_ds.fillna(0)

        # Define analysis domain
        analysis_domain = self.analysis_domain or {
            'longitude': (-5, 12),
            'latitude': (35, 45)
        }

        logger.info(
            f"Using analysis domain: lon=({analysis_domain['longitude'][0]}, {analysis_domain['longitude'][1]}), "
            f"lat=({analysis_domain['latitude'][0]}, {analysis_domain['latitude'][1]})")

        # Extract domain for test dataset (ERA5)
        processor = data_base.DataProcessor()
        test_domain = processor.extract_domain(
            test_ds,
            analysis_domain['longitude'],
            analysis_domain['latitude']
        )

        # Extract domain for mask if provided
        mask_domain = None
        if mask is not None:
            mask_name = getattr(mask, 'name', 'mask')
            mask_ds = mask.to_dataset(name=mask_name)

            mask_domain = processor.extract_domain(
                mask_ds,
                analysis_domain['longitude'],
                analysis_domain['latitude']
            )
            mask_domain = mask_domain[mask_name]

            # Verify test dataset and mask have consistent grids
            try:
                data_domains.verify_grid_consistency([test_domain, mask_domain.to_dataset()])
                logger.info("Grid consistency verified between test dataset and mask")
            except ValueError as e:
                logger.error(f"Grid consistency check failed: {str(e)}")
                raise

        # Regrid reference dataset to test dataset grid
        logger.info("Regridding reference dataset to test dataset grid")
        reference_regridded = data_domains.regrid_to_test(
            test_domain, reference_ds, method='linear'
        )

        # Extract domain for regridded reference dataset
        reference_domain = processor.extract_domain(
            reference_regridded,
            analysis_domain['longitude'],
            analysis_domain['latitude']
        )

        return reference_domain, test_domain, mask_domain

    @staticmethod
    def compute_statistics(reference_ds: xr.Dataset,
                           test_ds: xr.Dataset,
                           reference_var: str = 'rr',
                           test_var: str = 'tp') -> Dict[str, xr.DataArray]:
        """Compute comparison statistics between reference and test datasets."""
        # Detect time dimensions
        ref_time_dim = next((d for d in reference_ds.dims if str(d).lower() in
                             ('time', 't', 'valid_time')), 'time')
        test_time_dim = next((d for d in test_ds.dims if str(d).lower() in
                              ('time', 't', 'valid_time')), 'time')

        # Compute statistics with the correct dimensions
        stats: dict[str, xr.DataArray] = {
            'reference_mean': reference_ds[reference_var].mean(dim=ref_time_dim),
            'test_mean': test_ds[test_var].mean(dim=test_time_dim)
        }

        # Bias (test - reference)
        stats['bias'] = stats['test_mean'] - stats['reference_mean']

        # Relative bias (%) - avoid division by zero
        with xr.set_options(keep_attrs=True):
            # Create a safe denominator that replaces zeros with NaN
            safe_ref_mean = xr.where(stats['reference_mean'] != 0, stats['reference_mean'], np.nan)
            stats['relative_bias'] = (stats['bias'] / safe_ref_mean).pipe(lambda x: x * 100.0)

        # Prepare for correlation/RMSE - handle time dimension differences
        if ref_time_dim == test_time_dim:
            ref_data = reference_ds[reference_var]
            test_data = test_ds[test_var]
            time_dim = ref_time_dim
        else:
            test_renamed = test_ds.rename({test_time_dim: ref_time_dim})
            ref_data = reference_ds[reference_var]
            test_data = test_renamed[test_var]
            time_dim = ref_time_dim

        # Calculate correlation where we have sufficient non-zero data
        # Count non-NaN and non-zero values along time dimension
        valid_ref = ref_data.notnull().sum(dim=time_dim)
        valid_test = test_data.notnull().sum(dim=time_dim)

        # Calculate correlation only where we have enough data points
        # and both datasets have variation
        template = xr.full_like(stats['reference_mean'], np.nan)

        # Only calculate correlation where there are sufficient data points
        try:
            # Convert to numpy array to avoid attribute access issues
            calc_mask_array = np.array((valid_ref > 1) & (valid_test > 1))

            # Check if any points are valid
            if np.any(calc_mask_array):
                corr = xr.corr(ref_data, test_data, dim=time_dim)
                stats['correlation'] = template.copy()
                stats['correlation'] = xr.where(calc_mask_array, corr, np.nan)
            else:
                stats['correlation'] = template
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {e}")
            stats['correlation'] = template

        # Calculate RMSE with similar precautions
        try:
            squared_diff = (test_data - ref_data) ** 2
            # Skip warnings by only calculating where we have data
            rmse = np.sqrt(squared_diff.mean(dim=time_dim, skipna=True))
            stats['rmse'] = rmse.where(valid_ref > 1)
        except (ValueError, TypeError, FloatingPointError, RuntimeError) as e:
            logger.warning(f"RMSE calculation failed: {e}")
            stats['rmse'] = template.copy()

        return stats
