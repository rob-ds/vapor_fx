"""
Supporting Components for VAPOR_FX framework

This module contains the supporting classes and data structures for the
object-based validation framework. These components implement core aspects
of the validation methodology including object identification, matching,
and statistical significance testing.

Key Components:
    - Point: Data structure for 2D coordinate points
    - ValidationResult: Container for validation outputs
    - PrecipitationObject: Representation of a connected precipitation object
    - PrecipitationField: Container for precipitation objects in a timestep
    - ObjectMatcher: Implements object matching with fuzzy similarity
    - ValidationMetrics: Statistical calculations for validation metrics
    - DataManager: Data preparation and manipulation utilities

The implementation uses an advanced object-based approach with fuzzy matching
to detect fragmentation patterns where a single reference event may be
represented as multiple objects in the test dataset. The similarity measure
combines Jaccard overlap with a physically-based distance penalty.
"""

import logging
import xarray as xr
import numpy as np
import itertools
from scipy import ndimage
from typing import TypedDict, Tuple, NamedTuple, Dict, List
from functools import lru_cache

# Module logger
logger = logging.getLogger(__name__)


class ValidationConfig:
    """Configuration management for precipitation validation parameters.

    This class centralizes all configuration parameters used in the validation process,
    making it easier to modify settings and implement alternative configuration sources
    (files, environment variables, etc.) in the future.
    """

    def __init__(self,
                 min_size=1,
                 max_size=np.inf,
                 max_fragments=5,
                 similarity_threshold=0.08,
                 min_similarity_threshold=0.08,
                 max_similarity_threshold=0.91,
                 similarity_threshold_step=0.02,
                 permutations=1000,
                 permutation_batch_size=100,
                 lru_cache_size=8,
                 binary_int_type=np.int8):
        # Object identification parameters
        self.min_size = min_size
        self.max_size = max_size
        self.max_fragments = max_fragments

        # Similarity parameters
        self.similarity_threshold = similarity_threshold
        self.min_similarity_threshold = min_similarity_threshold
        self.max_similarity_threshold = max_similarity_threshold
        self.similarity_threshold_step = similarity_threshold_step

        # Statistical significance parameters
        self.permutations = permutations
        self.permutation_batch_size = permutation_batch_size

        # Performance parameters
        self.lru_cache_size = lru_cache_size
        self.binary_int_type = binary_int_type

    @classmethod
    def from_file(cls, filepath):
        """Load configuration from a JSON or YAML file."""
        # Implementation to be added later
        pass

    def get_thresholds(self):
        """Generate threshold array based on configuration."""
        return np.arange(
            self.min_similarity_threshold,
            self.max_similarity_threshold,
            self.similarity_threshold_step
        )


# Default configuration
DEFAULT_CONFIG = ValidationConfig()


# Class for storing 2D coordinate points
class Point(NamedTuple):
    """Represents a 2D coordinate point with y and x components."""
    y: float
    x: float


# Class for storing validation results
class ValidationResult(NamedTuple):
    """Container for validation analysis results of extreme precipitation event detection."""
    similarity: xr.DataArray  # Object-based similarity values
    ecr: xr.DataArray  # Event Coincidence Rate (ECR) at each threshold
    ecr_pvalue: xr.DataArray  # Statistical significance for ECR
    lir: xr.DataArray  # Local Intensity Ratio (LIR) at each threshold
    lir_pvalue: xr.DataArray  # Statistical significance for LIR
    ib: xr.DataArray  # Intensity Bias (IB) at each threshold
    ib_pvalue: xr.DataArray  # Statistical significance for IB
    displacement: xr.DataArray  # Centroid displacement vectors
    confusion_matrix: Dict[str, xr.DataArray]  # TP, FP, FN, TN arrays
    metrics: Dict[str, xr.DataArray]  # POD, FAR, CSI arrays


# Class for storing matched object information
class ObjectMatch(TypedDict):
    """Represents a match between precipitation objects with matching metrics."""
    test_id: int  # Primary matched test object ID
    test_ids: List[int]  # List of test object IDs (multiple for fragmentation)
    similarity: float
    displacement: Tuple[float, float]


class PrecipitationObject:
    """Represents a precipitation object identified in a dataset.

    Encapsulates properties of connected precipitation cells including
    coordinates, centroid, and size with lazy evaluation for efficiency.

    Attributes:
        id: Unique identifier for the object
        coords: Tuple of array coordinates (y_indices, x_indices)
        _mask: Optional pre-computed binary mask of the object
        _centroid: Cached centroid coordinates (y, x)
        _size: Cached size (number of cells)
    """

    def __init__(self, object_id, coords, mask=None):
        self.id = object_id  # We can still use "id" as an instance attribute
        self.coords = coords
        self._mask = mask
        self._centroid = None
        self._size = None

    @property
    def centroid(self):
        """Calculate and cache the object's centroid coordinates.

        Returns:
            Tuple (y, x) of centroid coordinates
        """
        if self._centroid is None:
            self._centroid = (np.mean(self.coords[0]), np.mean(self.coords[1]))
        return self._centroid

    @property
    def size(self):
        """Calculate and cache the object's size (number of cells).

        Returns:
            Integer count of cells in the object
        """
        if self._size is None:
            self._size = len(self.coords[0])
        return self._size

    def get_mask(self, shape):
        """Get or create a binary mask for this object.

        Args:
            shape: Tuple of (height, width) for the output mask

        Returns:
            Binary numpy array of specified shape with True at object locations
        """
        if self._mask is None:
            self._mask = np.zeros(shape, dtype=bool)
            self._mask[self.coords] = True
        return self._mask


class PrecipitationField:
    """Container for precipitation objects in a single timestep."""

    def __init__(self, binary_field, min_size=None, max_size=None, config=DEFAULT_CONFIG):
        self.binary_field = binary_field
        self.shape = binary_field.shape
        self.config = config

        # Use config values if parameters not explicitly provided
        self.min_size = min_size if min_size is not None else self.config.min_size
        self.max_size = max_size if max_size is not None else self.config.max_size

        self.labeled_field, self.objects = self._identify_objects(self.min_size, self.max_size)

    def _identify_objects(self, min_size, max_size):
        """Identify precipitation objects as connected components in binary field.

        Returns:
            Tuple containing:
                - Labeled array where each object has a unique integer ID
                - Dictionary of PrecipitationObject instances with keys as object IDs
        """
        # Use binary field as input
        field_to_analyze = self.binary_field

        # Skip if no precipitation in field
        if not np.any(field_to_analyze):
            return np.zeros_like(field_to_analyze), {}

        # Label connected components
        labeled_field, num_features = ndimage.label(field_to_analyze)

        if num_features == 0:
            return np.zeros_like(field_to_analyze), {}

        # Calculate component sizes
        component_sizes = np.bincount(np.asarray(labeled_field).ravel())

        # Filter small and large components (label 0 is background, so start from index 1)
        too_small = np.where(component_sizes[1:] < min_size)[0] + 1

        # Only apply max_size filter if the value is finite
        if np.isfinite(max_size):
            too_large = np.where(component_sizes[1:] > max_size)[0] + 1
        else:
            too_large = np.array([], dtype=int)  # Empty array if max_size is infinite

        # Remove out-of-range components from labeled field
        for invalid_label in np.concatenate([too_small, too_large]):
            labeled_field = np.where(labeled_field == invalid_label, 0, labeled_field)

        # Relabel remaining components consecutively
        if np.any(labeled_field):
            labeled_field, num_features = ndimage.label(labeled_field > 0)
        else:
            return np.zeros_like(field_to_analyze), {}

        # Extract properties for remaining components
        objects = {}
        for i in range(1, num_features + 1):
            coords = np.where(labeled_field == i)

            # Skip if empty (should not happen but just in case)
            if len(coords[0]) == 0:
                continue

            # Create PrecipitationObject instead of dictionary
            objects[i] = PrecipitationObject(object_id=i, coords=coords)

        return labeled_field, objects

    def get_object(self, object_id):
        """Get precipitation object by ID."""
        return self.objects.get(object_id)

    def get_all_objects(self):
        """Get all precipitation objects."""
        return self.objects


class ObjectMatcher:
    """Matches precipitation objects between reference and test fields.

    This class implements the object-based matching algorithm using fuzzy similarity
    with physics-based distance penalty. It supports fragmentation patterns where
    multiple test objects can collectively represent a single reference object.

    Attributes:
        ref_field: PrecipitationField containing reference objects
        test_field: PrecipitationField containing test objects
        config: ValidationConfig instance with matching parameters
        matches: Dictionary of matches between reference and test objects
    """

    def __init__(self, ref_field, test_field, config=DEFAULT_CONFIG):
        self.ref_field = ref_field
        self.test_field = test_field
        self.config = config
        self.matches = {}

    def match_objects(self):
        """Match precipitation objects between reference and test fields.

        Implements fuzzy approach with many-to-one matching where multiple test
        objects can represent a single reference object. Only considers test
        objects that spatially overlap with the reference object.

        Returns:
            Dictionary of matches with reference object IDs as keys
        """
        ref_objects = self.ref_field.get_all_objects()
        test_objects = self.test_field.get_all_objects()
        labeled_ref = self.ref_field.labeled_field
        min_similarity = self.config.similarity_threshold
        max_fragments = self.config.max_fragments

        # Track fragmentation statistics
        single_matches = 0
        fragmented_matches = 0
        fragments_count = []

        if not ref_objects or not test_objects:
            return {}

        ref_shape = labeled_ref.shape

        for ref_id, ref_obj in ref_objects.items():
            ref_mask = ref_obj.get_mask(ref_shape)
            ref_centroid = Point(y=ref_obj.centroid[0], x=ref_obj.centroid[1])
            ref_size = ref_obj.size

            # Find candidates that spatially overlap with reference
            candidates = []
            for test_id, test_obj in test_objects.items():
                test_mask = test_obj.get_mask(ref_shape)

                # Check for spatial overlap
                if np.any(ref_mask & test_mask):
                    candidates.append(test_id)

            # Skip if no candidates found
            if not candidates:
                continue

            best_match_ids = []
            best_score = 0
            best_displacement = (0.0, 0.0)

            # Evaluate individual objects and combinations up to max_fragments
            for n_fragments in range(1, min(max_fragments + 1, len(candidates) + 1)):
                for combo in itertools.combinations(candidates, n_fragments):
                    # Create unified mask and get properties
                    test_mask, test_centroid, test_size = self.__class__._create_unified_mask(
                        test_objects, list(combo), ref_shape)

                    # Calculate centroid distance and displacement
                    dist, (dy, dx) = self._calculate_centroid_distance(ref_centroid, test_centroid)

                    # Calculate Jaccard similarity with alignment
                    jaccard = self._calculate_object_jaccard(ref_mask, test_mask, displacement=(dy, dx))

                    # Calculate physically-based distance penalty
                    distance_penalty = self._calculate_distance_penalty(dist, ref_size, test_size)

                    # Combined similarity metric
                    similarity_score = jaccard * distance_penalty

                    if similarity_score > best_score:
                        best_score = similarity_score
                        best_match_ids = list(combo)
                        best_displacement = (dy, dx)

            # Store match if above threshold
            if best_match_ids and best_score >= min_similarity:
                # Use the first/primary object ID for backward compatibility
                primary_id = best_match_ids[0]
                self.matches[ref_id] = {
                    "test_id": primary_id,
                    "test_ids": best_match_ids,
                    "similarity": best_score,
                    "displacement": best_displacement
                }

                # Record fragmentation statistics
                if len(best_match_ids) == 1:
                    single_matches += 1
                else:
                    fragmented_matches += 1
                    fragments_count.append(len(best_match_ids))

        # Log fragmentation statistics
        if single_matches + fragmented_matches > 0:
            logger.debug(f"Found {single_matches} single and {fragmented_matches} fragmented matches")
            if fragmented_matches > 0:
                avg_fragments = sum(fragments_count) / len(fragments_count)
                logger.debug(f"Average fragments per fragmented match: {avg_fragments:.2f}")

        return self.matches

    @staticmethod
    def _create_unified_mask(test_objects, test_ids, shape):
        """Create a unified binary mask from multiple test objects.

        Args:
            test_objects: Dictionary of test objects
            test_ids: List of test object IDs to combine
            shape: Shape of the grid (height, width)

        Returns:
            Tuple of (unified_mask, unified_centroid, unified_size)
        """
        # Create empty mask
        unified_mask = np.zeros(shape, dtype=bool)

        # Track totals for centroid calculation
        total_cells = 0
        weighted_y = 0.0
        weighted_x = 0.0

        # Fill mask with all selected objects
        for test_id in test_ids:
            if test_id in test_objects:
                obj = test_objects[test_id]
                coords = obj.coords
                unified_mask[coords] = True

                size = obj.size
                total_cells += size

                centroid_y, centroid_x = obj.centroid
                weighted_y += centroid_y * size
                weighted_x += centroid_x * size

        # Calculate unified centroid (weighted by object size)
        if total_cells > 0:
            unified_centroid = Point(y=weighted_y / total_cells, x=weighted_x / total_cells)
        else:
            unified_centroid = Point(y=0.0, x=0.0)

        return unified_mask, unified_centroid, total_cells

    @staticmethod
    @lru_cache(maxsize=DEFAULT_CONFIG.lru_cache_size)
    def _calculate_centroid_distance(ref_point, test_point):
        """Calculate distance between two centroids and return displacement vector.

        Args:
            ref_point: Reference object centroid coordinates (Point)
            test_point: Test object centroid coordinates (Point)

        Returns:
            Tuple of (distance, (dy, dx)) where (dy, dx) is the displacement vector
        """
        dy = test_point.y - ref_point.y
        dx = test_point.x - ref_point.x

        return np.sqrt(dy ** 2 + dx ** 2), (dy, dx)

    @staticmethod
    def _calculate_object_jaccard(ref_mask, test_mask, displacement=None):
        """Calculate Jaccard similarity between two object masks with optional alignment.

        Args:
            ref_mask: Binary mask of reference object
            test_mask: Binary mask of test object
            displacement: Optional (dy, dx) displacement to align test_mask with ref_mask

        Returns:
            Jaccard similarity [0-1]
        """
        if displacement:
            # Shift test_mask to align with ref_mask
            dy, dx = int(round(displacement[0])), int(round(displacement[1]))
            shifted_test_mask = np.zeros_like(test_mask)

            # Get valid indices after shifting
            h, w = test_mask.shape
            y_src_start, x_src_start = max(0, dy), max(0, dx)
            y_src_end, x_src_end = min(h, h + dy), min(w, w + dx)
            y_dst_start, x_dst_start = max(0, -dy), max(0, -dx)
            y_dst_end, x_dst_end = min(h, h - dy), min(w, w - dx)

            # Copy values to shifted position
            shifted_test_mask[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
                test_mask[y_src_start:y_src_end, x_src_start:x_src_end]

            # Use shifted mask for comparison
            intersection = np.sum(ref_mask & shifted_test_mask)
            union = np.sum(ref_mask | shifted_test_mask)
        else:
            intersection = np.sum(ref_mask & test_mask)
            union = np.sum(ref_mask | test_mask)

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def _calculate_distance_penalty(dist, ref_size, test_size):
        """Calculate physics-based distance penalty between object centroids.

        Args:
            dist: Euclidean distance between centroids
            ref_size: Size of reference object (in cells)
            test_size: Size of test object (in cells)

        Returns:
            Distance penalty factor [0-1]
        """
        # Calculate size ratio between objects
        size_ratio = min(ref_size, test_size) / max(ref_size, test_size)

        # Natural distance scale based on object size ratio
        distance_scale = np.sqrt(max(ref_size, test_size)) * size_ratio

        # Calculate distance penalty (inverse square) with scaling
        return 1.0 / (1.0 + (dist / distance_scale))

    def classify_cells(self, similarity_threshold=None):
        """Classify cells based on object matches for confusion matrix.

        Supports many-to-one matching where multiple test objects can represent a single reference object.

        Args:
            similarity_threshold: Minimum similarity for a good match (uses config default if None)

        Returns:
            Tuple of (true_positives, false_positives, false_negatives) arrays
        """
        if similarity_threshold is None:
            similarity_threshold = self.config.similarity_threshold

        labeled_ref = self.ref_field.labeled_field
        labeled_test = self.test_field.labeled_field

        # Initialize classification arrays
        tp = np.zeros_like(labeled_ref, dtype=self.config.binary_int_type)
        fp = np.zeros_like(labeled_ref, dtype=self.config.binary_int_type)
        fn = np.zeros_like(labeled_ref, dtype=self.config.binary_int_type)

        # Skip if no objects to classify
        if not self.matches and np.max(labeled_ref) == 0 and np.max(labeled_test) == 0:
            return tp, fp, fn

        # Process matched reference objects
        matched_ref_ids = set()
        matched_test_ids = set()

        for ref_id, match in self.matches.items():
            matched_ref_ids.add(ref_id)
            similarity = match["similarity"]

            if similarity >= similarity_threshold:
                # Good match - mark reference object cells as TP
                tp[labeled_ref == ref_id] = 1

                # Handle multiple test objects (fragmentation)
                for test_id in match["test_ids"]:
                    matched_test_ids.add(test_id)
            else:
                # Poor match - mark as FN
                fn[labeled_ref == ref_id] = 1

        # Mark unmatched reference objects as FN
        self._mark_unmatched_objects(labeled_ref, matched_ref_ids, fn)

        # Mark unmatched test objects as FP
        self._mark_unmatched_objects(labeled_test, matched_test_ids, fp)

        return tp, fp, fn

    @staticmethod
    def _mark_unmatched_objects(labeled_array, matched_ids, output_array, value=1):
        """Mark cells with unmatched object IDs in the output array.

        This function modifies output_array in-place, setting the specified value
        for cells corresponding to unmatched objects in labeled_array.

        Args:
            labeled_array: Array with labeled objects (IDs)
            matched_ids: Set of object IDs that have been matched
            output_array: Array to mark unmatched objects in (modified in-place)
            value: Value to set for unmatched objects (default: 1)
        """
        max_id = np.max(labeled_array)
        if max_id <= 0:
            return

        all_ids = set(range(1, max_id + 1))
        unmatched_ids = all_ids - matched_ids

        for obj_id in unmatched_ids:
            if np.any(labeled_array == obj_id):  # Check if ID exists
                output_array[labeled_array == obj_id] = value


class ValidationMetrics:
    """Calculates and stores validation metrics between precipitation datasets.

    This class provides methods for calculating various statistical metrics to validate
    extreme precipitation event detection, including:
    - Event Coincidence Rate (ECR) with statistical significance
    - Local Intensity Ratio (LIR) with statistical significance
    - Intensity Bias (IB) with statistical significance

    Attributes:
        config: ValidationConfig with parameters for statistical calculations
    """

    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config

    def calculate_ecr(self, reference, validation, n_permutations=None):
        """Calculate Event Coincidence Rate (ECR) and its statistical significance.

        This method computes the ECR between reference and validation time series
        and determines its statistical significance through a vectorized permutation test.
        Implements early termination when statistical confidence is reached.

        Args:
            reference: Binary time series of reference events (0s and 1s)
            validation: Binary time series of validation events (0s and 1s)
            n_permutations: Maximum number of permutations (uses config default if None)

        Returns:
            Tuple containing:
                - Event Coincidence Rate [0-1]
                - P-value representing statistical significance [0-1]
        """
        if n_permutations is None:
            n_permutations = self.config.permutations

        # Filter out NaN values
        ref_clean, val_clean, total_ref_events, is_valid = self._prepare_binary_arrays(reference, validation)

        # Handle invalid or edge cases
        if not is_valid:
            return np.nan, np.nan

        if total_ref_events == 0:
            return 0.0, 1.0

        # Calculate observed ECR
        coincident_events = np.sum(ref_clean & val_clean)
        observed_ecr = coincident_events / total_ref_events

        # Early exit if observed_ecr is 0
        if observed_ecr == 0:
            return 0.0, 1.0

        # Calculate number of validation events for permutation
        n_val_events = np.sum(val_clean)

        # If validation has no events, ECR is 0
        if n_val_events == 0:
            return 0.0, 1.0

        # Pre-compute indices where reference events occur
        ref_event_indices = np.where(ref_clean == 1)[0]
        ref_length = len(ref_clean)

        # Initialize for permutation test
        count_significant = 0
        total_permutations = 0

        # Process permutations in batches
        batch_size = min(self.config.permutation_batch_size, n_permutations)

        for start in range(0, n_permutations, batch_size):
            current_batch = min(batch_size, n_permutations - start)
            total_permutations += current_batch

            # Generate all random positions at once (matrix of shape [batch_size, n_val_events])
            random_positions = np.array([
                np.random.choice(ref_length, n_val_events, replace=False)
                for _ in range(current_batch)
            ])

            # Count coincidences for all permutations at once
            batch_coincidences = np.array([
                np.sum(np.isin(positions, ref_event_indices))
                for positions in random_positions
            ])

            # Calculate ECRs and count significant results
            batch_ecrs = batch_coincidences / total_ref_events

            # Count permutations with distance >= observed
            count_significant += np.sum(batch_ecrs >= observed_ecr)

            # Early termination checks
            if self.__class__._early_permutation_termination(count_significant, total_permutations):
                break

        # Final p-value
        p_value = count_significant / total_permutations

        return observed_ecr, float(p_value)

    def calculate_lir(self, reference_intensities, validation_intensities, n_permutations=None, observed_lir=None):
        """Calculate Local Intensity Ratio (LIR) and its statistical significance.

        This method computes the median ratio between validation and reference intensities
        and determines its statistical significance through permutation testing.

        Args:
            reference_intensities: Intensity values from reference dataset
            validation_intensities: Intensity values from validation dataset
            n_permutations: Maximum number of permutations (uses config default if None)
            observed_lir: Pre-calculated LIR value (optional)

        Returns:
            Tuple containing:
                - Local Intensity Ratio (median of validation/reference) [0-inf]
                - P-value representing statistical significance [0-1]
        """
        if n_permutations is None:
            n_permutations = self.config.permutations

        # Filter out NaN values and ensure reference > 0
        valid_indices = ~(np.isnan(reference_intensities) | np.isnan(validation_intensities))
        valid_indices &= (reference_intensities > 0)

        if not np.any(valid_indices):
            return np.nan, np.nan

        ref_valid = reference_intensities[valid_indices]
        val_valid = validation_intensities[valid_indices]

        if len(ref_valid) == 0:
            return np.nan, np.nan

        # Use provided LIR if available, otherwise calculate it
        if observed_lir is not None:
            lir_value = observed_lir
        else:
            # Calculate the observed LIR (median ratio)
            ratios = val_valid / ref_valid
            lir_value = float(np.median(ratios))

        # Calculate distance from perfect ratio (1.0)
        observed_distance = abs(lir_value - 1.0)

        # Special case: if observed_distance is 0, the p-value is 1.0
        if observed_distance == 0:
            return lir_value, 1.0

        # Initialize counters for permutation test
        count_significant = 0
        total_permutations = 0

        # Process permutations in batches
        batch_size = min(self.config.permutation_batch_size, n_permutations)

        for start in range(0, n_permutations, batch_size):
            current_batch = min(batch_size, n_permutations - start)
            total_permutations += current_batch

            # Generate all permutations at once
            batch_results = np.zeros(current_batch)

            for i in range(current_batch):
                # Random permutation of validation intensities
                perm_validation = np.random.permutation(val_valid)

                # Calculate LIR for permutation
                perm_ratios = perm_validation / ref_valid
                perm_lir = np.median(perm_ratios)

                # Calculate distance from perfect ratio
                batch_results[i] = abs(perm_lir - 1.0)

            # Count permutations with distance >= observed
            count_significant += np.sum(batch_results >= observed_distance)

            # Early termination checks
            if self.__class__._early_permutation_termination(count_significant, total_permutations):
                break

        # Final p-value
        p_value = count_significant / total_permutations

        return lir_value, float(p_value)

    def calculate_ib(self, reference_intensities, validation_intensities, n_permutations=None, observed_ib=None):
        """Calculate Intensity Bias (IB) and its statistical significance.

        This method computes the mean bias between validation and reference intensities
        and determines its statistical significance through permutation testing.

        Args:
            reference_intensities: Intensity values from reference dataset
            validation_intensities: Intensity values from validation dataset
            n_permutations: Maximum number of permutations (uses config default if None)
            observed_ib: Pre-calculated IB value (optional)

        Returns:
            Tuple containing:
                - Intensity Bias (mean of validation-reference) [mm/day]
                - P-value representing statistical significance [0-1]
        """
        if n_permutations is None:
            n_permutations = self.config.permutations

        # Filter out NaN values
        valid_indices = ~(np.isnan(reference_intensities) | np.isnan(validation_intensities))

        if not np.any(valid_indices):
            return np.nan, np.nan

        ref_valid = reference_intensities[valid_indices]
        val_valid = validation_intensities[valid_indices]

        if len(ref_valid) == 0:
            return np.nan, np.nan

        # Use provided IB if available, otherwise calculate it
        if observed_ib is not None:
            ib_value = observed_ib
        else:
            # Calculate the observed IB (mean difference)
            ib_value = float(np.mean(val_valid - ref_valid))

        # Use absolute value for significance testing
        observed_magnitude = abs(ib_value)

        # Special case: if observed_magnitude is 0, the p-value is 1.0
        if observed_magnitude == 0:
            return ib_value, 1.0

        # Initialize counters for permutation test
        count_significant = 0
        total_permutations = 0

        # Process permutations in batches
        batch_size = min(self.config.permutation_batch_size, n_permutations)

        for start in range(0, n_permutations, batch_size):
            current_batch = min(batch_size, n_permutations - start)
            total_permutations += current_batch

            # Generate all permutations at once
            batch_results = np.zeros(current_batch)

            for i in range(current_batch):
                # Random permutation of validation intensities
                perm_validation = np.random.permutation(val_valid)

                # Calculate IB for permutation (mean difference)
                perm_ib = np.mean(perm_validation - ref_valid)

                # Store absolute magnitude
                batch_results[i] = abs(perm_ib)

            # Count permutations with magnitude >= observed
            count_significant += np.sum(batch_results >= observed_magnitude)

            # Early termination checks
            if self.__class__._early_permutation_termination(count_significant, total_permutations):
                break

        # Final p-value
        p_value = count_significant / total_permutations

        return ib_value, float(p_value)

    def _prepare_binary_arrays(self, reference, validation):
        """Filter NaN values from arrays and prepare for binary analysis.

        Args:
            reference: Binary time series of reference events
            validation: Binary time series of validation events

        Returns:
            Tuple containing:
                - Filtered reference array (NaNs removed, converted to binary type)
                - Filtered validation array (NaNs removed, converted to binary type)
                - Total count of reference events
                - Flag indicating if arrays contain valid data (False if all NaNs)
        """
        valid_indices = ~(np.isnan(reference) | np.isnan(validation))

        if not np.any(valid_indices):
            # Return empty arrays instead of None to maintain type consistency
            return (np.array([], dtype=self.config.binary_int_type),
                    np.array([], dtype=self.config.binary_int_type),
                    0, False)

        # Use only valid indices
        ref_clean = reference[valid_indices].astype(self.config.binary_int_type)
        val_clean = validation[valid_indices].astype(self.config.binary_int_type)

        # Count total reference events
        total_ref_events = np.sum(ref_clean)

        return ref_clean, val_clean, total_ref_events, True

    @staticmethod
    def _early_permutation_termination(count_significant, total_permutations, confidence_interval=0.01,
                                       min_samples=100):
        """Determine if permutation testing can terminate early based on statistical confidence.

        Args:
            count_significant: Number of permutations exceeding observed value
            total_permutations: Total number of permutations performed
            confidence_interval: Width of confidence interval for early termination
            min_samples: Minimum permutation samples required before considering early termination

        Returns:
            Boolean indicating whether to terminate permutation testing early
        """
        # Need minimum samples for reliable estimate
        if total_permutations < min_samples:
            return False

        # Calculate current p-value
        current_p = count_significant / total_permutations

        # Calculate confidence interval using normal approximation
        se = np.sqrt(current_p * (1 - current_p) / total_permutations)
        ci_width = 4 * se  # 95% confidence interval (±2 SE)

        # Terminate if CI is narrow enough or p-value is extreme
        return ci_width < confidence_interval or current_p < 0.01 or current_p > 0.99


class DataManager:
    """Handles data preparation and manipulation for precipitation validation.

    This class provides methods for loading, preprocessing, masking, and managing
    the various data arrays required for validation analysis.

    Attributes:
        ref_data: Reference data array
        test_data: Test data array
        ref_intensity: Reference intensity data (optional)
        test_intensity: Test intensity data (optional)
        config: ValidationConfig instance
    """

    def __init__(self, ref_data, test_data, ref_intensity=None, test_intensity=None, config=DEFAULT_CONFIG):
        # Store input data arrays
        self.ref_data = ref_data
        self.test_data = test_data
        self.ref_intensity = ref_intensity
        self.test_intensity = test_intensity
        self.config = config

        # Extract data dimensions
        self.n_time, self.n_lat, self.n_lon = self.ref_data.shape
        self.thresholds = np.arange(
            self.config.min_similarity_threshold,
            self.config.max_similarity_threshold,
            self.config.similarity_threshold_step
        )
        self.n_thresholds = len(self.thresholds)

    @staticmethod
    def create_threshold_data_array(data, ref_data, thresholds):
        """Create a DataArray with standardized dimensions and coordinates for threshold-based metrics.

        Args:
            data: Array containing data values
            ref_data: Reference data array to extract coordinate values from
            thresholds: Array of threshold values

        Returns:
            xr.DataArray: Formatted data array with proper dimensions and coordinates
        """
        return xr.DataArray(
            data,
            dims=['latitude', 'longitude', 'threshold'],
            coords={
                'latitude': ref_data.latitude,
                'longitude': ref_data.longitude,
                'threshold': thresholds
            }
        )
