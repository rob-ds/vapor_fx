#!/usr/bin/env python
"""
Run object-based validation for extreme event detection using VAPOR_FX framework.

This script executes the VAPOR_FX framework to compare event detection in test data
against reference data, providing comprehensive validation metrics including spatial
pattern recognition with fuzzy similarity.

Example usage:
    python process_validation.py --years 1950-2023 --percentiles 95 --variables tp_24h_00
"""

import argparse
import logging
import sys
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime

# Import validation framework
from src.validation.vapor_fx import PrecipitationValidator
from src.validation.validation_utils import DEFAULT_CONFIG
from src.utils import setup_script_logger, parse_years

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Configure logging and create logger
log_file = Path(__file__).parent / "process_validation.log"
logger = setup_script_logger("process_validation", log_file)


def validate_against_reference(variable_name, percentile_value, era5_file_path, eobs_reference, config=DEFAULT_CONFIG):
    """Process a single ERA5 file and validate against reference data.

    Args:
        variable_name: ERA5 variable name (e.g., 'tp_24h_00')
        percentile_value: Percentile threshold (e.g., 95)
        era5_file_path: Path to ERA5 file
        eobs_reference: Pre-loaded reference data
        config: ValidationConfig object with parameters

    Returns:
        ValidationResult object with results
    """
    # Log entry - file_key is used for file naming and logging
    file_key = f"{variable_name}_r{percentile_value}p"
    logger.info(f"Processing ERA5 test data for: {file_key}")

    # Load ERA5 data
    ds_era5 = xr.open_dataset(era5_file_path)
    era5_data = ds_era5["rr_r95p"]

    # Binarize data
    eobs_binary = xr.where(eobs_reference > 0, 1, 0)
    era5_binary = xr.where(era5_data > 0, 1, 0)

    print(f"\n[{file_key}] After binarization:")
    print(f"[{file_key}] Reference number of detections: {np.sum(eobs_binary.values == 1)}")
    print(f"[{file_key}] Test number of detections: {np.sum(era5_binary.values == 1)}")

    # Initialize validator with the configuration
    validator = PrecipitationValidator(
        ref_data=eobs_binary,
        test_data=era5_binary,
        ref_intensity=eobs_reference,  # Original intensity values
        test_intensity=era5_data,  # Original intensity values
        config=config
    )

    # Run validation
    logger.info(f"[{file_key}] Running object-based validation...")
    validation_results = validator.validate()

    # Print result summaries
    print(f"\n[{file_key}] Validation results:")
    print(f"[{file_key}] ECR shape: {validation_results.ecr.shape}")
    print(f"[{file_key}] Similarity shape: {validation_results.similarity.shape}")
    print(f"[{file_key}] ECR p-value shape: {validation_results.ecr_pvalue.shape}")
    print(f"[{file_key}] LIR shape: {validation_results.lir.shape}")
    print(f"[{file_key}] IB shape: {validation_results.ib.shape}")
    print(f"[{file_key}] Displacement shape: {validation_results.displacement.shape}")

    print(f"\n[{file_key}] Results verification:")
    print(f"[{file_key}] Non-zero ECRs: {np.sum(validation_results.ecr.values > 0)}")
    print(f"[{file_key}] Non-zero similarities: {np.sum(validation_results.similarity.values > 0)}")
    print(f"[{file_key}] Non-zero ECR p-values: {np.sum(~np.isnan(validation_results.ecr_pvalue.values))}")
    print(f"[{file_key}] Non-zero displacements: {np.sum(np.abs(validation_results.displacement.values) > 0)}")

    print(f"\n[{file_key}] Value ranges:")
    print(f"[{file_key}] Similarity range: ["
          f"{np.nanmin(validation_results.similarity.values):.3f}, "
          f"{np.nanmax(validation_results.similarity.values):.3f}]")
    print(f"[{file_key}] ECR range: ["
          f"{np.nanmin(validation_results.ecr.values):.3f}, "
          f"{np.nanmax(validation_results.ecr.values):.3f}]")
    print(f"[{file_key}] ECR p-value range: ["
          f"{np.nanmin(validation_results.ecr_pvalue.values):.3f}, "
          f"{np.nanmax(validation_results.ecr_pvalue.values):.3f}]")
    print(f"[{file_key}] LIR range: ["
          f"{np.nanmin(validation_results.lir.values):.3f}, "
          f"{np.nanmax(validation_results.lir.values):.3f}]")
    print(f"[{file_key}] IB range: ["
          f"{np.nanmin(validation_results.ib.values):.3f}, "
          f"{np.nanmax(validation_results.ib.values):.3f}]")

    return validation_results


def process_validation(ref_path, test_vars, percentiles, years, config, output_dir):
    """Process validation between reference and test datasets.

    Args:
        ref_path: Path to reference dataset
        test_vars: List of test variables to process
        percentiles: List of percentile thresholds to analyze
        years: List of years covered by data
        config: Validation configuration object
        output_dir: Directory for output files
    """
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load reference data (common for all processing)
    logger.info("Processing E-OBS reference data...")
    logger.info(f"Loading reference data from {ref_path}")
    ds_ref = xr.open_dataset(ref_path)
    ref_data = ds_ref['rr_r95p']

    # Create project root path
    project_root = Path(__file__).resolve().parents[2]

    # Create dict of test dataset paths
    test_paths = {}
    for var in test_vars:
        for percentile in percentiles:
            key = f'{var}_r{percentile}p'
            test_paths[key] = project_root / f'data/processed/era5_percentiles/era5_{key}_{min(years)}_{max(years)}.nc'

    # Process each test file
    for key, test_path in test_paths.items():
        # Extract var and percentile from key
        var, percentile_part = key.split('_r')
        percentile = int(percentile_part.replace('p', ''))

        # Process file
        logger.info(f"Processing {key}")
        try:
            results = validate_against_reference(var, percentile, test_path, ref_data, config)

            # Save results
            logger.info(f"Saving results for {key}")

            # Save similarity results
            ds_sim = xr.Dataset({
                'similarity': results.similarity,
                'ecr': results.ecr,
                'ecr_pvalue': results.ecr_pvalue,
                'lir': results.lir,
                'lir_pvalue': results.lir_pvalue,
                'ib': results.ib,
                'ib_pvalue': results.ib_pvalue,
                'displacement': results.displacement
            })
            sim_output = output_dir / f'object_similarity_{key}.nc'
            ds_sim.to_netcdf(sim_output)

            # Save confusion matrix-derived performance metrics
            ds_metrics = xr.Dataset(results.metrics)
            ds_metrics['latitude'] = ('latitude', results.similarity.latitude.values)
            ds_metrics['longitude'] = ('longitude', results.similarity.longitude.values)
            ds_metrics['threshold'] = ('threshold', config.get_thresholds())
            metrics_output = output_dir / f'object_metrics_{key}.nc'
            ds_metrics.to_netcdf(metrics_output)

            logger.info(f"Completed processing {key}: similarity={sim_output}, metrics={metrics_output}")

        except Exception as e:
            logger.error(f"Error processing {key}: {str(e)}", exc_info=True)


def main():
    """Main execution function for extreme event validation."""
    parser = argparse.ArgumentParser(description="Run object-based extreme event validation")

    parser.add_argument("--years", default='1980-2023',
                        help="Years to process (e.g., 1979-2020 or 1979,1980,1981)")
    parser.add_argument("--percentiles", default="95",
                        help="Percentiles to analyze (e.g., 95 or 90,95,98)")
    parser.add_argument("--variables", default="tp_24h_00,tp_24h_06,tp_24h_12,tp_24h_18",
                        help="ERA5 variables to process (comma separated)")
    parser.add_argument("--reference", default=None,
                        help="Reference dataset path (defaults to E-OBS)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for validation results")
    parser.add_argument("--config", help="Path to validation configuration file")

    args = parser.parse_args()

    # Parse years and percentiles
    years = parse_years(args.years)
    percentiles = [int(p.strip()) for p in args.percentiles.split(',')]
    variables = [v.strip() for v in args.variables.split(',')]

    # Determine project root
    project_root = Path(__file__).resolve().parents[2]

    # Set reference path
    if args.reference:
        ref_path = Path(args.reference)
    else:
        ref_path = project_root / f'data/processed/eobs_percentiles/eobs_r95p_{min(years)}_{max(years)}.nc'

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / 'data/validated/'

    # Load configuration if provided
    config = DEFAULT_CONFIG
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        # TODO: Implement configuration loading from file
        # config = ValidationConfig.from_file(args.config)

    # Log entry
    logger.info(f"===== VAPOR_FX Validation Pipeline Started at {datetime.now().isoformat()} =====")
    logger.info(f"Years: {min(years)}-{max(years)}")
    logger.info(f"Percentiles: {percentiles}")
    logger.info(f"Variables: {variables}")
    logger.info(f"Reference: {ref_path}")
    logger.info(f"Output directory: {output_dir}")

    # Run validation process
    try:
        process_validation(ref_path, variables, percentiles, years, config, output_dir)
        logger.info("===== VAPOR_FX Validation Complete =====")
    except Exception as e:
        logger.error(f"Validation process failed: {str(e)}", exc_info=True)
        logger.info("===== VAPOR_FX Validation Failed =====")

    # Flush handlers explicitly
    for handler in logging.root.handlers:
        handler.flush()


if __name__ == '__main__':
    main()
