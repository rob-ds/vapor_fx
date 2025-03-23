#!/usr/bin/env python
"""
Process precipitation thresholds for the VAPOR_FX project.

This script calculates and applies precipitation thresholds (R95p, R90p, R98p)
for ERA5 and E-OBS datasets, producing standardized outputs for further analysis.

Example usage:
    python process_thresholds.py --years 1950-2023 --dataset eobs --percentiles 95
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.thresholds import EOBSThresholdProcessor, ERA5ThresholdProcessor
from src.utils import setup_script_logger, parse_years

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Configure logging and create logger
log_file = Path(__file__).parent / "process_thresholds.log"
logger = setup_script_logger("process_thresholds", log_file)


def main():
    """Main execution function for percentile threshold processing."""
    parser = argparse.ArgumentParser(description="Process ERA5 and E-OBS thresholds")

    parser.add_argument("--years", default='1980-2023',
                        help="Years to process (e.g., 1979-2020 or 1979,1980,1981)")
    parser.add_argument("--dataset", choices=['eobs', 'era5', 'all'], default='all',
                        help="Dataset to process (eobs, era5, or all)")
    parser.add_argument("--percentiles", default="95",
                        help="Percentiles to calculate (e.g., 95 or 90,95,98)")
    parser.add_argument("--data-dir", default=None,
                        help="Root data directory")
    parser.add_argument("--wet-day-threshold", type=float, default=1.0,
                        help="Threshold for wet days in mm/day (default: 1.0)")

    args = parser.parse_args()

    # Parse years and percentiles
    years = parse_years(args.years)
    percentiles = [int(p.strip()) for p in args.percentiles.split(',')]

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Default: project_root/data
        data_dir = Path(__file__).resolve().parents[2] / 'data'

    # Log entry
    logger.info(f"===== Percentile Threshold Processing Started at {datetime.now().isoformat()} =====")
    logger.info(f"Years: {min(years)}-{max(years)}")
    logger.info(f"Percentiles: {percentiles}")
    logger.info(f"Wet day threshold: {args.wet_day_threshold} mm/day")

    # Process E-OBS data if requested
    if args.dataset in ['eobs', 'all']:
        try:
            logger.info("Processing E-OBS precipitation thresholds")
            processor = EOBSThresholdProcessor(data_dir)

            for percentile in percentiles:
                output_file = (data_dir / 'processed' / 'eobs_percentiles' /
                               f"eobs_r{percentile}p_{min(years)}_{max(years)}.nc")

                logger.info(f"Processing E-OBS R{percentile}p")
                processor.process(
                    years=years,
                    var_name='rr',
                    percentile=percentile,
                    wet_day_threshold=args.wet_day_threshold,
                    output_file=output_file
                )
                logger.info(f"E-OBS R{percentile}p saved to: {output_file}")

        except Exception as e:
            logger.error(f"Error processing E-OBS data: {str(e)}", exc_info=True)

    # Process ERA5 data if requested
    if args.dataset in ['era5', 'all']:
        try:
            logger.info("Processing ERA5 precipitation thresholds")
            processor = ERA5ThresholdProcessor(data_dir)

            # Process each variable and percentile
            era5_vars = ['tp_24h_00', 'tp_24h_06', 'tp_24h_12', 'tp_24h_18']

            for var_name in era5_vars:
                for percentile in percentiles:
                    output_file = (data_dir / 'processed' / 'era5_percentiles' /
                                   f"era5_{var_name}_r{percentile}p_{min(years)}_{max(years)}.nc")

                    logger.info(f"Processing ERA5 {var_name} R{percentile}p")
                    processor.process(
                        years=years,
                        var_name=var_name,
                        percentile=percentile,
                        wet_day_threshold=args.wet_day_threshold,
                        output_file=output_file
                    )
                    logger.info(f"ERA5 {var_name} R{percentile}p saved to: {output_file}")

        except Exception as e:
            logger.error(f"Error processing ERA5 data: {str(e)}", exc_info=True)

    logger.info("===== Percentile Threshold Processing Complete =====")

    # Flush handlers explicitly
    for handler in logging.root.handlers:
        handler.flush()


if __name__ == '__main__':
    main()
