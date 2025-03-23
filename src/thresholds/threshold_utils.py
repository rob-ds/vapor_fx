"""
Utility functions for precipitation threshold processing.

This module provides common functions used across the threshold processing
implementation, such as directory handling and dataset utilities.
"""

import logging
from pathlib import Path
from typing import Union


# Set up logging
logger = logging.getLogger(__name__)


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path
