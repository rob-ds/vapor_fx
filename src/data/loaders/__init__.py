"""Data loaders for various climate datasets."""

from .load_era5 import ERA5Loader
from .load_eobs import EOBSLoader

__all__ = ['ERA5Loader', 'EOBSLoader']
