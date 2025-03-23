"""
Common utility functions for the MEDEX precipitation analysis project.
"""

import logging
from pathlib import Path
from typing import List, Union, Tuple


def setup_script_logger(logger_name: str, log_file: Union[str, Path], mode: str = 'w') -> logging.Logger:
    """Set up a script logger with file and console handlers."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to prevent duplication
    if logger.handlers:
        logger.handlers.clear()

    # Create handlers with explicit configuration
    file_handler = logging.FileHandler(log_file, mode=mode, encoding='utf-8')
    console_handler = logging.StreamHandler()

    # Create and apply formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def parse_years(years_str: str) -> List[int]:
    """Parse years string into a list of years.

    Accepts formats:
    - Single year: "2000"
    - Range: "1979-2020"
    - Comma-separated: "1979,1980,1981"
    - Combinations: "1979-1981,1990,2000-2005"

    Args:
        years_str: String representation of years

    Returns:
        List of years
    """
    years = []
    parts = years_str.split(',')

    for part in parts:
        if '-' in part:
            start, end = part.split('-')
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))

    return sorted(years)


def parse_period_ranges(ranges_str: str) -> List[Tuple[int, int]]:
    """Parse period ranges string into a list of tuples.

    Args:
        ranges_str: String representation of ranges (e.g., "1950-1964,1965-1979")

    Returns:
        List of (start_year, end_year) tuples
    """
    periods = []
    parts = ranges_str.split(',')

    for part in parts:
        start, end = part.split('-')
        periods.append((int(start), int(end)))

    return periods
