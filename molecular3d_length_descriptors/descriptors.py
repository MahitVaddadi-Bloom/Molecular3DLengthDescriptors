"""
3D Molecular Descriptors Module

Modernized version of the original Descriptors.py with NumPy 2.x compatibility
and enhanced error handling for robust 3D molecular shape analysis.

This module provides five key 3D molecular descriptors:
- Flatness: Measure of molecular planarity
- Cubeularity: Measure of cubic shape  
- Plateularity: Measure of plate-like shape
- ShortOverLong: Ratio of short to long molecular axes
- MediumOverLong: Ratio of medium to long molecular axes
"""

import math
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .numpy_compat import (
    safe_asarray,
    safe_sqrt,
    get_principal_moments,
    validate_coordinates
)


def Flatness(coords: Union[np.ndarray, List[List[float]]]) -> float:
    """
    Calculate the flatness of a molecule from its 3D coordinates.
    
    Flatness measures how planar a molecule is. A perfectly flat molecule
    has flatness = 1, while a spherical molecule approaches flatness = 0.
    
    Formula: Flatness = (eigenvalue2 - eigenvalue1) / eigenvalue2
    
    Args:
        coords: Molecular coordinates as numpy array (N x 3) or list of lists
        
    Returns:
        Flatness value (0-1, where 1 is perfectly flat)
        
    Raises:
        ValueError: If coordinates are invalid or calculation fails
    """
    try:
        coords = validate_coordinates(safe_asarray(coords))
        principal_moments = get_principal_moments(coords)
        
        # principal_moments are sorted in descending order: [L1, L2, L3]
        L1, L2, L3 = principal_moments
        
        # Handle edge case where L2 is zero (perfectly linear molecule)
        if L2 < 1e-10:
            return 1.0  # Perfectly flat (linear)
            
        flatness = (L2 - L3) / L2
        
        # Ensure result is in valid range [0, 1]
        return max(0.0, min(1.0, flatness))
        
    except Exception as e:
        raise ValueError(f"Failed to calculate flatness: {e}")


def Cubeularity(coords: Union[np.ndarray, List[List[float]]]) -> float:
    """
    Calculate the cubeularity of a molecule from its 3D coordinates.
    
    Cubeularity measures how cubic (3D symmetric) a molecule is.
    A perfect cube has cubeularity = 1, while elongated molecules have lower values.
    
    Formula: Cubeularity = (eigenvalue1 * eigenvalue2 * eigenvalue3) / max_eigenvalue^3
    
    Args:
        coords: Molecular coordinates as numpy array (N x 3) or list of lists
        
    Returns:
        Cubeularity value (0-1, where 1 is perfectly cubic)
        
    Raises:
        ValueError: If coordinates are invalid or calculation fails
    """
    try:
        coords = validate_coordinates(safe_asarray(coords))
        principal_moments = get_principal_moments(coords)
        
        L1, L2, L3 = principal_moments
        
        # Handle edge case where L1 is zero
        if L1 < 1e-10:
            return 0.0
            
        cubeularity = (L1 * L2 * L3) / (L1 ** 3)
        
        # Ensure result is in valid range [0, 1]
        return max(0.0, min(1.0, cubeularity))
        
    except Exception as e:
        raise ValueError(f"Failed to calculate cubeularity: {e}")


def Plateularity(coords: Union[np.ndarray, List[List[float]]]) -> float:
    """
    Calculate the plateularity of a molecule from its 3D coordinates.
    
    Plateularity measures how plate-like a molecule is (flat but with some width).
    
    Formula: Plateularity = sqrt(eigenvalue1 * eigenvalue3) / eigenvalue2
    
    Args:
        coords: Molecular coordinates as numpy array (N x 3) or list of lists
        
    Returns:
        Plateularity value (typically 0-2, where higher values indicate more plate-like shape)
        
    Raises:
        ValueError: If coordinates are invalid or calculation fails
    """
    try:
        coords = validate_coordinates(safe_asarray(coords))
        principal_moments = get_principal_moments(coords)
        
        L1, L2, L3 = principal_moments
        
        # Handle edge case where L2 is zero
        if L2 < 1e-10:
            return 0.0
            
        plateularity = safe_sqrt(L1 * L3) / L2
        
        # Ensure non-negative result
        return max(0.0, plateularity)
        
    except Exception as e:
        raise ValueError(f"Failed to calculate plateularity: {e}")


def ShortOverLong(coords: Union[np.ndarray, List[List[float]]]) -> float:
    """
    Calculate the ratio of shortest to longest molecular axis.
    
    This descriptor measures molecular elongation. Values close to 1 indicate
    spherical molecules, while values close to 0 indicate highly elongated molecules.
    
    Formula: ShortOverLong = eigenvalue3 / eigenvalue1
    
    Args:
        coords: Molecular coordinates as numpy array (N x 3) or list of lists
        
    Returns:
        Short/Long ratio (0-1, where 1 is spherical, 0 is linear)
        
    Raises:
        ValueError: If coordinates are invalid or calculation fails
    """
    try:
        coords = validate_coordinates(safe_asarray(coords))
        principal_moments = get_principal_moments(coords)
        
        L1, L2, L3 = principal_moments
        
        # Handle edge case where L1 is zero
        if L1 < 1e-10:
            return 0.0
            
        short_over_long = L3 / L1
        
        # Ensure result is in valid range [0, 1]
        return max(0.0, min(1.0, short_over_long))
        
    except Exception as e:
        raise ValueError(f"Failed to calculate short over long ratio: {e}")


def MediumOverLong(coords: Union[np.ndarray, List[List[float]]]) -> float:
    """
    Calculate the ratio of medium to longest molecular axis.
    
    This descriptor provides additional information about molecular shape
    beyond the short/long ratio.
    
    Formula: MediumOverLong = eigenvalue2 / eigenvalue1
    
    Args:
        coords: Molecular coordinates as numpy array (N x 3) or list of lists
        
    Returns:
        Medium/Long ratio (0-1)
        
    Raises:
        ValueError: If coordinates are invalid or calculation fails
    """
    try:
        coords = validate_coordinates(safe_asarray(coords))
        principal_moments = get_principal_moments(coords)
        
        L1, L2, L3 = principal_moments
        
        # Handle edge case where L1 is zero
        if L1 < 1e-10:
            return 0.0
            
        medium_over_long = L2 / L1
        
        # Ensure result is in valid range [0, 1]
        return max(0.0, min(1.0, medium_over_long))
        
    except Exception as e:
        raise ValueError(f"Failed to calculate medium over long ratio: {e}")


def CalcAllDesc(coords: Union[np.ndarray, List[List[float]]]) -> Dict[str, float]:
    """
    Calculate all 3D molecular descriptors from coordinates.
    
    This function computes all five descriptors in one call for efficiency,
    as they all use the same eigenvalue decomposition.
    
    Args:
        coords: Molecular coordinates as numpy array (N x 3) or list of lists
        
    Returns:
        Dictionary containing all descriptor values:
        - 'Flatness': Planarity measure (0-1)
        - 'Cubeularity': Cubic symmetry measure (0-1)  
        - 'Plateularity': Plate-like shape measure (0+)
        - 'ShortOverLong': Short/long axis ratio (0-1)
        - 'MediumOverLong': Medium/long axis ratio (0-1)
        
    Raises:
        ValueError: If coordinates are invalid or calculation fails
    """
    try:
        coords = validate_coordinates(safe_asarray(coords))
        principal_moments = get_principal_moments(coords)
        
        L1, L2, L3 = principal_moments
        
        # Handle edge cases
        if L1 < 1e-10:
            return {
                'Flatness': 0.0,
                'Cubeularity': 0.0,
                'Plateularity': 0.0,
                'ShortOverLong': 0.0,
                'MediumOverLong': 0.0
            }
        
        # Calculate all descriptors
        results = {}
        
        # Flatness
        if L2 < 1e-10:
            results['Flatness'] = 1.0
        else:
            results['Flatness'] = max(0.0, min(1.0, (L2 - L3) / L2))
        
        # Cubeularity
        results['Cubeularity'] = max(0.0, min(1.0, (L1 * L2 * L3) / (L1 ** 3)))
        
        # Plateularity
        if L2 < 1e-10:
            results['Plateularity'] = 0.0
        else:
            results['Plateularity'] = max(0.0, safe_sqrt(L1 * L3) / L2)
        
        # ShortOverLong
        results['ShortOverLong'] = max(0.0, min(1.0, L3 / L1))
        
        # MediumOverLong
        results['MediumOverLong'] = max(0.0, min(1.0, L2 / L1))
        
        return results
        
    except Exception as e:
        raise ValueError(f"Failed to calculate all descriptors: {e}")


# Backward compatibility - maintain original function names
def calculate_flatness(coords):
    """Backward compatibility wrapper for Flatness."""
    return Flatness(coords)


def calculate_cubeularity(coords):
    """Backward compatibility wrapper for Cubeularity."""
    return Cubeularity(coords)


def calculate_plateularity(coords):
    """Backward compatibility wrapper for Plateularity."""
    return Plateularity(coords)


def calculate_short_over_long(coords):
    """Backward compatibility wrapper for ShortOverLong."""
    return ShortOverLong(coords)


def calculate_medium_over_long(coords):
    """Backward compatibility wrapper for MediumOverLong."""
    return MediumOverLong(coords)


def calculate_all_descriptors(coords):
    """Backward compatibility wrapper for CalcAllDesc."""
    return CalcAllDesc(coords)


# Export all functions
__all__ = [
    'Flatness',
    'Cubeularity', 
    'Plateularity',
    'ShortOverLong',
    'MediumOverLong',
    'CalcAllDesc',
    'calculate_flatness',
    'calculate_cubeularity',
    'calculate_plateularity', 
    'calculate_short_over_long',
    'calculate_medium_over_long',
    'calculate_all_descriptors'
]