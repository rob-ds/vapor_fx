#!/usr/bin/env python
"""
Process ERA5 and E-OBS precipitation data for the VAPOR_FX project.

This script processes ERA5 and E-OBS precipitation data for analysis,
standardizing domains, resolutions, and applying optional land-sea masks.

Example usage:
    python process_data.py --years 1979-2020 --mask --analysis-only
"""
import os
import argparse
import logging
import xarray as xr
import sys
from pathlib import Path
from datetime import datetime

from src.utils import setup_script_logger, parse_years, parse_period_ranges
from src.data import data_base
from src.data import data_masking
from src.data.loaders import ERA5Loader, EOBSLoader
from src.data.processor.processor import AnalysisProcessor

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Configure logging and create logger
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "process_data.log")
logger = setup_script_logger("process_data", log_file)

# Log entry
logger.info(f"===== Data Preprocessing Started at {datetime.now().isoformat()} =====")


def main():
    """Main execution function for data processing.

    Processes ERA5 reanalysis and EOBS observational data:
    1. Loads raw data without masking
    2. Regrids EOBS to ERA5 grid
    3. Extracts study domain
    4. Calculates statistics
    5. Applies land-sea mask
    6. Saves processed data
    """
    parser = argparse.ArgumentParser(description="Process ERA5 and E-OBS data")

    parser.add_argument("--years", default='1980-2023',
                        help="Years to process (e.g., 1979-2020 or 1979,1980,1981)")
    parser.add_argument("--periods", default="1980-1994,1995-2010,2011-2024",
                        help="Period ranges for E-OBS data files")
    parser.add_argument("--mask", action="store_true",
                        help="Apply land-sea mask")
    parser.add_argument("--mask-file", default=None,
                        help="Path to land-sea mask file")
    parser.add_argument("--data-dir", default=None,
                        help="Root data directory")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Process only analysis domain data")
    parser.add_argument("--season", default="SON",
                        help="3-month season to extract (JFM, FMA, MAM,... DJF)")

    args = parser.parse_args()

    # Parse years and period ranges
    years = parse_years(args.years)
    period_ranges = parse_period_ranges(args.periods)

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Default: project_root/data
        data_dir = Path(__file__).resolve().parents[2] / 'data'

    # Load mask if requested
    mask = None
    if args.mask:
        if args.mask_file:
            mask_file = Path(args.mask_file)
        else:
            mask_file = data_dir / 'raw' / 'era5-land' / 'land_sea_mask_0.25deg.nc'

        if mask_file.exists():
            logger.info(f"Loading land-sea mask from {mask_file} \n")
            mask = data_masking.load_mask(mask_file)
        else:
            logger.warning(f"Mask file not found: {mask_file}. Creating a new mask.")
            era5_land_file = data_dir / 'raw' / 'era5-land' / 't2m_era5-land_land-sea-mask.nc'
            mask = data_masking.create_mask_from_era5land(
                era5_land_file, mask_file, resolution=0.25
            )

    # Initialize loaders
    era5_loader = ERA5Loader(data_dir)
    eobs_loader = EOBSLoader(data_dir)

    # Define analysis domain
    analysis_domain = {
        'longitude': (-5, 12),
        'latitude': (35, 45)
    }

    # Initialize analysis processor
    analysis_processor = AnalysisProcessor(
        analysis_domain=analysis_domain,
        apply_mask=args.mask
    )

    # Process each year
    for year in years:
        logger.info(f"============ Processing year {year} ============\n")

        try:
            # Step 1: Load data (without masking)
            # -------------------------------------
            # Load ERA5 data (6h and 24h timeframes)
            era5_datasets = era5_loader.load_and_process(year, None)

            # Extract the 6-hourly dataset for regridding reference
            era5_ds = era5_datasets['6h']

            # Load E-OBS data
            eobs_ds = eobs_loader.load_and_process(year, period_ranges, None)

            # Step 2: Regrid EOBS to ERA5 grid (full domain)
            # ---------------------------------------------
            # Process datasets for regridding EOBS to ERA5 grid
            std_eobs_ds, std_era5_ds, mask_domain = analysis_processor.process_datasets(
                eobs_ds, era5_ds, mask,
                season=None
            )

            # Step 3: Extract study domain for all datasets
            # -------------------------------------------
            processor = data_base.DataProcessor()

            # Extract domain for 24h ERA5 datasets
            for timeframe in era5_datasets:
                if timeframe != '6h':  # 6h already processed in process_datasets
                    era5_datasets[timeframe] = processor.extract_domain(
                        era5_datasets[timeframe],
                        analysis_domain['longitude'],
                        analysis_domain['latitude']
                    )

            # Update 6h data in era5_datasets with already processed domain
            era5_datasets['6h'] = std_era5_ds

            # Step 4: Calculate statistics (unmasked data)
            # ------------------------------------------
            # Calculate statistics for each 24h aggregation
            all_stats = {}
            for hour_start in [0, 6, 12, 18]:
                # Get the 24h dataset for this starting hour
                var_name_24h = f"tp_24h_{str(hour_start).zfill(2)}"
                if var_name_24h in era5_datasets['24h'].data_vars:
                    logger.info(f"Computing statistics for {var_name_24h}")

                    # Calculate statistics between EOBS (daily) and 24h ERA5 data
                    hour_stats = analysis_processor.compute_statistics(
                        std_eobs_ds, era5_datasets['24h'],
                        reference_var='rr', test_var=var_name_24h
                    )

                    # Add prefix to stats keys
                    all_stats.update({f"{key}_{hour_start}h": value for key, value in hour_stats.items()})

            # Combine all statistics
            stats_ds = xr.Dataset(all_stats)

            # Step 5: Apply masks to all datasets
            # ---------------------------------
            if mask is not None and args.mask and mask_domain is not None:
                logger.info("Applying land-sea mask to all datasets")

                # Ensure mask is binary (1=land, 0=ocean)
                binary_mask = xr.where(mask_domain > 0.5, 1, 0)

                # Log mask information for debugging
                logger.info(f"Mask domain shape: {binary_mask.shape}")
                logger.info(f"Mask has {float(binary_mask.sum().values)} land points")

                # Mask EOBS data
                std_eobs_ds = processor.apply_mask(std_eobs_ds, binary_mask, ['rr'])

                # Mask ERA5 datasets with specific variable names
                for timeframe in era5_datasets:
                    if timeframe == '6h':
                        era5_datasets[timeframe] = processor.apply_mask(
                            era5_datasets[timeframe], binary_mask, ['tp_6h']
                        )
                    else:
                        vars_to_mask = [v for v in era5_datasets[timeframe].data_vars
                                        if v.startswith('tp_24h_')]
                        era5_datasets[timeframe] = processor.apply_mask(
                            era5_datasets[timeframe], binary_mask, vars_to_mask
                        )

                # Mask statistics
                for var in stats_ds.data_vars:
                    var_name = str(var)  # Convert to string
                    stats_ds[var] = processor.apply_mask(
                        stats_ds[var].to_dataset(name=var_name),
                        binary_mask, [var_name]
                    )[var_name]

            # Step 6: Save processed data
            # -------------------------
            era5_files = era5_loader.save_processed_data(era5_datasets, year, 'analysis')
            eobs_file = eobs_loader.save_processed_data(std_eobs_ds, year)

            # Save statistics
            stats_file = data_dir / 'processed' / 'stats' / f"stats_{year}.nc"
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            stats_ds.to_netcdf(stats_file)

            logger.info(f"Successfully processed year {year}")
            logger.info(f"E-OBS saved to: {eobs_file}")
            logger.info(f"ERA5 6h data saved to: {era5_files['6h']}")
            logger.info(f"ERA5 24h data saved to: {era5_files['24h']}")
            logger.info(f"Statistics saved to: {stats_file}\n")

        except Exception as e:
            logger.error(f"Error processing year {year}: {str(e)}", exc_info=True)

    logger.info("===== Data Preprocessing Analysis Complete =====")

    # Flush handlers explicitly
    for handler in logging.root.handlers:
        handler.flush()


if __name__ == '__main__':
    main()
