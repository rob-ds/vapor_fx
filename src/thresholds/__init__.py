"""
Precipitation threshold detection module.

This module provides classes and utilities for calculating and applying
thresholds such as R95p (95th percentile of wet days).
"""

from .threshold_base import ThresholdProcessor
from .threshold_processors import EOBSThresholdProcessor, ERA5ThresholdProcessor

__all__ = ['ThresholdProcessor', 'EOBSThresholdProcessor', 'ERA5ThresholdProcessor']
