"""
Molecular3DLengthDescriptors - 3D conformational based molecular descriptors.

A modernized package for computing 3D molecular shape descriptors including:
- Flatness: Measure of molecular planarity  
- Cubeularity: Measure of cubic shape
- Plateularity: Measure of plate-like shape
- ShortOverLong: Ratio of short to long molecular axes
- MediumOverLong: Ratio of medium to long molecular axes

Features:
- NumPy 2.x compatibility
- UV package manager support
- Rich CLI interface
- Robust 3D conformer generation
"""

from .version import __version__
from .numpy_compat import get_numpy_info

# Import main functionality from original modules
try:
    from Descriptors import (
        Flatness,
        Cubeularity, 
        Plateularity,
        ShortOverLong,
        MediumOverLong,
        CalcAllDesc
    )
    from Lengths import GetLengthsFromCoords, ComputeCoordsFrom2D
except ImportError:
    # Handle import for modernized structure
    try:
        from .descriptors import (
            Flatness,
            Cubeularity,
            Plateularity, 
            ShortOverLong,
            MediumOverLong,
            CalcAllDesc
        )
        from .lengths import GetLengthsFromCoords, ComputeCoordsFrom2D
    except ImportError:
        # Fallback - will be available after migration
        pass

__all__ = [
    '__version__',
    'get_numpy_info',
    'Flatness',
    'Cubeularity', 
    'Plateularity',
    'ShortOverLong',
    'MediumOverLong',
    'CalcAllDesc',
    'GetLengthsFromCoords',
    'ComputeCoordsFrom2D'
]