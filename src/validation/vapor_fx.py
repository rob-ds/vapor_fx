"""
Validation Approach for Pattern-based Object Recognition with Fuzzy similarity for Extreme events: VAPOR_FX

This module provides the main interface for validating detection of extreme events
between reference and test datasets using an object-based pattern recognition approach
with fuzzy similarity measurement.

Key Components:
    - ValidationConfig: Configuration management for validation parameters
    - PrecipitationValidator: Main class orchestrating the validation process
    - validate_against_reference: Function to process validation between datasets
    - DEFAULT_CONFIG: Default configuration instance with standard parameters

Example:
    ```python
    import xarray as xr
    from src.validation.validator_oop_simplified import PrecipitationValidator, DEFAULT_CONFIG

    # Load data
    ref_data = xr.open_dataset('reference_data.nc')['variable']
    test_data = xr.open_dataset('test_data.nc')['variable']

    # Create validator with intensity data
    validator = PrecipitationValidator(
        ref_data=ref_data,
        test_data=test_data,
        ref_intensity=ref_data,  # Original intensity values
        test_intensity=test_data,  # Original intensity values
        config=DEFAULT_CONFIG
    )

    # Run validation
    results = validator.validate()
    ```
"""

import logging
import xarray as xr
import numpy as np
import dask
from dask.diagnostics import ProgressBar
from tqdm import tqdm
# Import supporting classes from utils module
from .validation_utils import (
    DEFAULT_CONFIG,
    ValidationResult,
    PrecipitationField,
    ObjectMatcher,
    ValidationMetrics,
    DataManager
)

# Define public API
__all__ = ['PrecipitationValidator']

# Configure logging system for tracking validation progress and diagnostics
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PrecipitationValidator:
    """Coordinates validation process between reference and test precipitation datasets.

    This class orchestrates the entire validation workflow, integrating the component
    classes (PrecipitationField, ObjectMatcher, ValidationMetrics) to produce
    comprehensive validation results.

    Attributes:
        data_manager: DataManager instance handling data preparation
        metrics: ValidationMetrics instance for statistical calculations
        config: ValidationConfig instance with validation parameters
    """

    def __init__(self, ref_data, test_data, ref_intensity=None, test_intensity=None, config=DEFAULT_CONFIG):
        """Initialize the validator with datasets and configuration.

        Args:
            ref_data: Reference data (e.g., E-OBS)
            test_data: Test data (e.g., ERA5)
            ref_intensity: Original intensity values from reference dataset (optional)
            test_intensity: Original intensity values from test dataset (optional)
            config: ValidationConfig instance (uses default if not provided)
        """
        self.config = config
        self.data_manager = DataManager(ref_data, test_data, ref_intensity, test_intensity, config)
        self.metrics = ValidationMetrics(config)

    def validate(self):
        """Run the complete validation process.

        This method orchestrates the full validation workflow:
        1. Process each timestep to identify and match precipitation objects
        2. Calculate object-based similarity metrics
        3. Compute statistical significance through permutation testing
        4. Aggregate results into standardized data structures

        Returns:
            ValidationResult containing all validation metrics and arrays
        """
        logger.info("Starting object-based precipitation event validation...")
        logger.info(f"Configuration: min_size={self.config.min_size}, max_size={self.config.max_size}, "
                    f"max_fragments={self.config.max_fragments}")

        n_time = self.data_manager.n_time

        logger.info(f"Processing {n_time} timesteps using object-based approach...")

        # Process each timestep in parallel
        delayed_results = []
        for t in range(n_time):
            result = dask.delayed(self.process_timestep)(t)
            delayed_results.append(result)

        # Compute all timesteps in parallel
        logger.info("Starting parallel computation for object detection and matching...")
        with ProgressBar():
            timestep_results = dask.compute(*delayed_results)

        # Aggregate results from all timesteps
        logger.info("Aggregating results...")
        return self.aggregate_results(timestep_results)

    def process_timestep(self, t):
        """Process a single timestep to identify and match precipitation objects.

        Args:
            t: Timestep index

        Returns:
            Dictionary containing results for this timestep
        """
        # Get binary fields for current timestep
        ref_binary = self.data_manager.ref_data.values[t].copy()
        test_binary = self.data_manager.test_data.values[t].copy()

        # Skip processing if data is all NaN
        if np.isnan(ref_binary).all() or np.isnan(test_binary).all():
            return self._create_empty_result(ref_binary.shape)

        # Convert NaNs to zeros for object identification
        ref_binary = np.nan_to_num(ref_binary, nan=0)
        test_binary = np.nan_to_num(test_binary, nan=0)

        # Create precipitation fields
        ref_field = PrecipitationField(
            ref_binary,
            min_size=self.config.min_size,
            max_size=self.config.max_size
        )

        test_field = PrecipitationField(
            test_binary,
            min_size=self.config.min_size,
            max_size=self.config.max_size
        )

        # Match objects
        matcher = ObjectMatcher(ref_field, test_field, self.config)
        matches = matcher.match_objects()

        # Initialize arrays
        result_tp = np.zeros((len(self.data_manager.thresholds), *ref_binary.shape),
                             dtype=self.config.binary_int_type)
        result_fp = np.zeros((len(self.data_manager.thresholds), *ref_binary.shape),
                             dtype=self.config.binary_int_type)
        result_fn = np.zeros((len(self.data_manager.thresholds), *ref_binary.shape),
                             dtype=self.config.binary_int_type)
        result_displacements = np.zeros((*ref_binary.shape, 2))
        result_similarity = np.zeros_like(ref_binary)

        # Initialize intensity arrays
        result_lir = np.zeros((len(self.data_manager.thresholds), *ref_binary.shape))
        result_ib = np.zeros((len(self.data_manager.thresholds), *ref_binary.shape))

        # Classify cells for each threshold
        for i in range(len(self.data_manager.thresholds)):
            # Extract the threshold value safely, regardless of its structure
            threshold_item = self.data_manager.thresholds[i]

            # Handle different possible threshold formats
            if isinstance(threshold_item, tuple):
                # If it's a tuple (from enumeration), extract the value
                threshold_value = threshold_item[1]
            else:
                # Otherwise use the value directly
                threshold_value = threshold_item

            # Use the extracted value
            tp, fp, fn = matcher.classify_cells(float(threshold_value))
            result_tp[i] = tp
            result_fp[i] = fp
            result_fn[i] = fn

        # Store displacements and similarity values
        for ref_id, match in matches.items():
            coords = np.where(ref_field.labeled_field == ref_id)
            result_similarity[coords] = match["similarity"]

            for y, x in zip(*coords):
                result_displacements[y, x] = match["displacement"]

        # Calculate intensity metrics if intensity data is provided
        if (self.data_manager.ref_intensity is not None and
                self.data_manager.test_intensity is not None):
            # Calculate intensity metrics for matched objects
            ref_intensity_t = self.data_manager.ref_intensity.values[t]
            test_intensity_t = self.data_manager.test_intensity.values[t]

            # Calculate intensity metrics for each threshold
            for i in range(len(self.data_manager.thresholds)):
                # Get threshold value safely
                threshold_item = self.data_manager.thresholds[i]

                # Handle different threshold formats
                if isinstance(threshold_item, tuple):
                    threshold_value = float(threshold_item[1])
                else:
                    threshold_value = float(threshold_item)

                # Process only matched objects that meet this threshold
                for ref_id, match in matches.items():
                    similarity = match["similarity"]
                    if similarity >= threshold_value:
                        # Get coordinates for reference object
                        ref_coords = np.where(ref_field.labeled_field == ref_id)

                        # Create combined mask for all test objects
                        union_mask = np.zeros_like(ref_binary, dtype=bool)
                        union_mask[ref_coords] = True

                        # Add all matched test objects to the union mask
                        combined_test_coords = []
                        for test_id in match["test_ids"]:
                            test_coords = np.where(test_field.labeled_field == test_id)
                            union_mask[test_coords] = True
                            combined_test_coords.extend(zip(*test_coords))

                        # Extract intensity values
                        ref_values = ref_intensity_t[ref_coords]

                        # Get test values from all combined coordinates
                        test_values = np.array([test_intensity_t[y, x] for y, x in combined_test_coords])

                        # Skip if either has no valid intensity values
                        if len(ref_values) == 0 or len(test_values) == 0:
                            continue

                        # Calculate metrics
                        try:
                            # For LIR: Calculate point-wise ratios in overlap area
                            overlap_mask = np.zeros_like(ref_binary, dtype=bool)

                            # Create overlap mask considering all test objects
                            for y, x in zip(*ref_coords):
                                for test_id in match["test_ids"]:
                                    if test_field.labeled_field[y, x] == test_id:
                                        overlap_mask[y, x] = True
                                        break

                            if np.any(overlap_mask):
                                # Calculate ratio where both have values
                                valid_mask = (ref_intensity_t > 0) & overlap_mask
                                if np.any(valid_mask):
                                    ratio_values = np.divide(
                                        test_intensity_t[valid_mask],
                                        ref_intensity_t[valid_mask],
                                        out=np.ones_like(test_intensity_t[valid_mask]),
                                        where=ref_intensity_t[valid_mask] > 0
                                    )
                                    # Use median ratio for stability
                                    median_ratio = np.median(ratio_values)
                                    result_lir[i][union_mask] = median_ratio

                            # For Intensity Bias: Calculate mean difference
                            # Average all test values from all fragments
                            mean_test = np.mean(test_values)
                            mean_ref = np.mean(ref_values)
                            bias = mean_test - mean_ref
                            result_ib[i][union_mask] = bias

                        except (ValueError, ZeroDivisionError) as e:
                            # Skip problematic calculations
                            logger.debug(f"Error calculating intensity metrics: {e}")
                            continue

        return {
            'labeled_ref': ref_field.labeled_field,
            'labeled_test': test_field.labeled_field,
            'matches': matches,
            'tp': result_tp,
            'fp': result_fp,
            'fn': result_fn,
            'displacement': result_displacements,
            'similarity': result_similarity,
            'lir': result_lir,
            'ib': result_ib,
        }

    def _create_empty_result(self, shape):
        """Create empty result arrays for timesteps with no valid data."""
        return {
            'labeled_ref': np.zeros(shape),
            'labeled_test': np.zeros(shape),
            'matches': {},
            'tp': np.zeros((len(self.data_manager.thresholds), *shape), dtype=self.config.binary_int_type),
            'fp': np.zeros((len(self.data_manager.thresholds), *shape), dtype=self.config.binary_int_type),
            'fn': np.zeros((len(self.data_manager.thresholds), *shape), dtype=self.config.binary_int_type),
            'displacement': np.zeros((*shape, 2)),
            'similarity': np.zeros(shape),
            'lir': np.zeros((len(self.data_manager.thresholds), *shape)),
            'ib': np.zeros((len(self.data_manager.thresholds), *shape)),
        }

    def aggregate_results(self, timestep_results):
        """Aggregate results from all timesteps into final ValidationResult.

        This method combines the results from all processed timesteps, calculates
        spatial pattern matching metrics, statistical significance, and detection
        performance measures to create a comprehensive validation assessment.

        Args:
            timestep_results: List of results from each timestep

        Returns:
            ValidationResult containing all validation metrics and arrays
        """
        total_timesteps = len(timestep_results)
        logger.info(
            f"Aggregating detection results across {total_timesteps} timesteps and creating final spatial arrays...")

        # Extract dimensions for initialization
        n_time = self.data_manager.n_time
        n_lat = self.data_manager.n_lat
        n_lon = self.data_manager.n_lon
        n_thresholds = self.data_manager.n_thresholds
        thresholds = self.data_manager.thresholds

        # Initialize arrays for combined results
        similarity = np.zeros((n_time, n_lat, n_lon))
        displacement = np.zeros((n_time, n_lat, n_lon, 2))
        tp = np.zeros((n_time, n_lat, n_lon, n_thresholds), dtype=self.config.binary_int_type)
        fp = np.zeros((n_time, n_lat, n_lon, n_thresholds), dtype=self.config.binary_int_type)
        fn = np.zeros((n_time, n_lat, n_lon, n_thresholds), dtype=self.config.binary_int_type)
        lir = np.zeros((n_lat, n_lon, n_thresholds))
        ib = np.zeros((n_lat, n_lon, n_thresholds))

        # Calculate object detection statistics
        total_ref_objects = sum(
            len(np.unique(result['labeled_ref'][result['labeled_ref'] > 0]))
            for result in timestep_results
        )
        total_test_objects = sum(
            len(np.unique(result['labeled_test'][result['labeled_test'] > 0]))
            for result in timestep_results
        )
        total_matches = sum(len(result['matches']) for result in timestep_results)

        # Count total cells in each category
        total_ref_cells = sum(np.sum(result['labeled_ref'] > 0) for result in timestep_results)
        total_test_cells = sum(np.sum(result['labeled_test'] > 0) for result in timestep_results)

        # Count matched cells (cells in reference objects that have a match)
        total_matched_cells = 0
        for result in timestep_results:
            matched_ref_ids = set(result['matches'].keys())
            for ref_id in matched_ref_ids:
                total_matched_cells += np.sum(result['labeled_ref'] == ref_id)

        # Count fragmented matches
        fragmented_matches = 0
        total_fragments = 0
        for result in timestep_results:
            for match in result['matches'].values():
                if len(match['test_ids']) > 1:
                    fragmented_matches += 1
                    total_fragments += len(match['test_ids'])

        # Calculate fragmentation statistics
        fragmentation_rate = fragmented_matches / total_matches if total_matches > 0 else 0
        avg_fragments = total_fragments / fragmented_matches if fragmented_matches > 0 else 0

        # Log detection summary statistics
        logger.info("Object detection summary:")
        logger.info(f"  - Reference objects: {total_ref_objects} (containing {total_ref_cells} cells)")
        logger.info(f"  - Test objects: {total_test_objects} (containing {total_test_cells} cells)")
        logger.info(f"  - Matched objects: {total_matches} (containing {total_matched_cells} cells)")
        logger.info(f"  - Match rate: {total_matches / total_ref_objects:.1%} (matched/reference objects)")
        logger.info(f"  - Cell match rate: {total_matched_cells / total_ref_cells:.1%} (matched/reference cells)")
        logger.info(f"  - Fragmented matches: {fragmented_matches} ({fragmentation_rate:.1%} of matches)")
        if fragmented_matches > 0:
            logger.info(f"  - Average fragments per fragmented match: {avg_fragments:.2f}")

        # Fill arrays from timestep results
        with tqdm(total=len(timestep_results), desc="Combining timesteps", position=0, leave=True) as pbar:
            for t, result in enumerate(timestep_results):
                similarity[t] = result['similarity']
                displacement[t] = result['displacement']

                for i in range(n_thresholds):
                    tp[t, :, :, i] = result['tp'][i]
                    fp[t, :, :, i] = result['fp'][i]
                    fn[t, :, :, i] = result['fn'][i]

                    # Average intensity metrics over time
                    if np.any(result['lir'][i] > 0):
                        valid_mask = result['lir'][i] > 0
                        lir[:, :, i] += np.where(valid_mask, result['lir'][i], 0)

                    if np.any(result['ib'][i] != 0):
                        valid_mask = result['ib'][i] != 0
                        ib[:, :, i] += np.where(valid_mask, result['ib'][i], 0)

                # Update progress bar after each timestep is processed
                pbar.update(1)

        # Normalize intensity metrics by count of valid values
        lir_count = np.sum(tp > 0, axis=0)
        ib_count = np.sum(tp > 0, axis=0)

        # Avoid division by zero
        lir_count = np.where(lir_count > 0, lir_count, 1)
        ib_count = np.where(ib_count > 0, ib_count, 1)

        lir = lir / lir_count
        ib = ib / ib_count

        # Create confusion matrix dictionary
        confusion_matrices = {
            'TP': xr.DataArray(tp, dims=['time', 'latitude', 'longitude', 'threshold'],
                               coords={'time': self.data_manager.ref_data.time,
                                       'latitude': self.data_manager.ref_data.latitude,
                                       'longitude': self.data_manager.ref_data.longitude,
                                       'threshold': thresholds}),
            'FP': xr.DataArray(fp, dims=['time', 'latitude', 'longitude', 'threshold'],
                               coords={'time': self.data_manager.ref_data.time,
                                       'latitude': self.data_manager.ref_data.latitude,
                                       'longitude': self.data_manager.ref_data.longitude,
                                       'threshold': thresholds}),
            'FN': xr.DataArray(fn, dims=['time', 'latitude', 'longitude', 'threshold'],
                               coords={'time': self.data_manager.ref_data.time,
                                       'latitude': self.data_manager.ref_data.latitude,
                                       'longitude': self.data_manager.ref_data.longitude,
                                       'threshold': thresholds}),
            'TN': xr.DataArray(np.zeros_like(tp), dims=['time', 'latitude', 'longitude', 'threshold'],
                               coords={'time': self.data_manager.ref_data.time,
                                       'latitude': self.data_manager.ref_data.latitude,
                                       'longitude': self.data_manager.ref_data.longitude,
                                       'threshold': thresholds})
        }

        # Calculate true negatives (valid where neither dataset has precipitation)
        ref_zeros = np.array(self.data_manager.ref_data.values == 0, dtype=self.config.binary_int_type)
        test_zeros = np.array(self.data_manager.test_data.values == 0, dtype=self.config.binary_int_type)

        for i in range(n_thresholds):
            confusion_matrices['TN'].values[:, :, :, i] = ref_zeros & test_zeros

        # Create DataArrays for similarity and displacement
        similarity_da = xr.DataArray(
            similarity,
            dims=self.data_manager.ref_data.dims,
            coords=self.data_manager.ref_data.coords
        )

        displacement_da = xr.DataArray(
            displacement,
            dims=self.data_manager.ref_data.dims + ('disp',),
            coords={**self.data_manager.ref_data.coords, 'disp': [0, 1]}
        )

        # Initialize arrays for threshold-based ECR
        ecr_values = np.zeros((n_lat, n_lon, n_thresholds))
        ecr_pvalues = np.zeros((n_lat, n_lon, n_thresholds))
        lir_pvalues = np.zeros((n_lat, n_lon, n_thresholds))
        ib_pvalues = np.zeros((n_lat, n_lon, n_thresholds))

        # Calculate ECR and p-values for each grid point and threshold
        logger.info("Computing detection (ECR) and intensity metrics (LIR, IB) with statistical significance...")
        with tqdm(total=len(thresholds), desc="Processing thresholds", position=0, leave=True) as pbar:
            for i, threshold in enumerate(thresholds):
                for lat in range(n_lat):
                    for lon in range(n_lon):
                        if np.isnan(self.data_manager.ref_data.values[0, lat, lon]):
                            # Skip ocean points
                            ecr_values[lat, lon, i] = np.nan
                            ecr_pvalues[lat, lon, i] = np.nan
                            lir_pvalues[lat, lon, i] = np.nan
                            ib_pvalues[lat, lon, i] = np.nan
                            continue

                        # Get reference time series for this point
                        ref_series = self.data_manager.ref_data.values[:, lat, lon].astype(int)

                        # Get object-based similarity time series (1 if part of matched object, that is TP)
                        tp_series = tp[:, lat, lon, i].astype(int)

                        # Calculate ECR with significance (p-value)
                        ecr_values[lat, lon, i], ecr_pvalues[lat, lon, i] = self.metrics.calculate_ecr(ref_series,
                                                                                                       tp_series)

                        # Calculate intensity metric p-values if intensity data provided
                        if self.data_manager.ref_intensity is not None and self.data_manager.test_intensity is not None:
                            # Get intensity time series
                            ref_intensity_series = self.data_manager.ref_intensity.values[:, lat, lon]
                            test_intensity_series = self.data_manager.test_intensity.values[:, lat, lon]

                            # Calculate LIR p-value
                            _, lir_pvalues[lat, lon, i] = self.metrics.calculate_lir(
                                ref_intensity_series,
                                test_intensity_series,
                                observed_lir=float(lir[lat, lon, i])
                            )

                            # Calculate IB p-value
                            _, ib_pvalues[lat, lon, i] = self.metrics.calculate_ib(
                                ref_intensity_series,
                                test_intensity_series,
                                observed_ib=float(ib[lat, lon, i])
                            )

                # Update the progress bar after each threshold is processed
                pbar.update(1)

        # Create DataArrays for ECR, LIR, and IB and their p-values
        ecr_da = self.data_manager.create_threshold_data_array(ecr_values, self.data_manager.ref_data, thresholds)
        ecr_pvalue_da = self.data_manager.create_threshold_data_array(ecr_pvalues, self.data_manager.ref_data,
                                                                      thresholds)

        lir_da = self.data_manager.create_threshold_data_array(lir, self.data_manager.ref_data, thresholds)
        lir_pvalue_da = self.data_manager.create_threshold_data_array(lir_pvalues, self.data_manager.ref_data,
                                                                      thresholds)

        ib_da = self.data_manager.create_threshold_data_array(ib, self.data_manager.ref_data, thresholds)
        ib_pvalue_da = self.data_manager.create_threshold_data_array(ib_pvalues, self.data_manager.ref_data, thresholds)

        # Calculate performance metrics
        logger.info("Calculating performance metrics...")
        metrics = self._calculate_performance_metrics(confusion_matrices)

        # Create final ValidationResult
        return ValidationResult(
            similarity=similarity_da,
            ecr=ecr_da,
            ecr_pvalue=ecr_pvalue_da,
            lir=lir_da,
            lir_pvalue=lir_pvalue_da,
            ib=ib_da,
            ib_pvalue=ib_pvalue_da,
            displacement=displacement_da,
            confusion_matrix=confusion_matrices,
            metrics=metrics
        )

    @staticmethod
    def _calculate_performance_metrics(confusion_matrix):
        """Calculate performance metrics from confusion matrix elements.

        Time dimension is aggregated to produce metrics per location and threshold.
        Note that ocean points are masked by NaNs.

        Args:
            confusion_matrix: Dictionary of TP, FP, FN, TN arrays

        Returns:
            Dictionary of calculated metrics (POD, FAR, CSI)
        """
        # Sum over time dimension for each component
        tp_sum = confusion_matrix['TP'].sum(dim='time')
        fp_sum = confusion_matrix['FP'].sum(dim='time')
        fn_sum = confusion_matrix['FN'].sum(dim='time')

        # Calculate metrics
        pod = tp_sum / (tp_sum + fn_sum)
        far = fp_sum / (tp_sum + fp_sum)
        csi = tp_sum / (tp_sum + fp_sum + fn_sum)

        return {
            'POD': pod,
            'FAR': far,
            'CSI': csi
        }
