"""
NumPy compatibility layer for Molecular3DLengthDescriptors.

This module provides compatibility functions to handle differences between
NumPy 1.x and 2.x versions, particularly for eigenvalue decomposition and
linear algebra operations used in 3D molecular descriptor calculations.
"""

import numpy as np
from typing import Tuple, Optional, Union, Any
import warnings

# NumPy version check
NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
NUMPY_2_PLUS = NUMPY_VERSION >= (2, 0)

# Suppress specific NumPy warnings for better user experience
if NUMPY_2_PLUS:
    # ComplexWarning was removed in NumPy 2.0
    warnings.filterwarnings('ignore', message='.*deprecated.*')
    warnings.filterwarnings('ignore', message='.*complex.*')
else:
    # For NumPy 1.x, suppress ComplexWarning if available
    try:
        warnings.filterwarnings('ignore', category=np.ComplexWarning)
    except AttributeError:
        pass
    warnings.filterwarnings('ignore', message='.*deprecated.*')


def safe_array(data: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Create a NumPy array with compatibility across versions.
    
    Args:
        data: Input data to convert to array
        dtype: Optional data type specification
        
    Returns:
        NumPy array with appropriate handling for version differences
    """
    try:
        if dtype is not None:
            return np.array(data, dtype=dtype)
        return np.array(data)
    except Exception as e:
        # Fallback with explicit float64 for numerical stability
        if dtype is None:
            return np.array(data, dtype=np.float64)
        raise e


def safe_asarray(data: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Convert input to array with compatibility handling.
    
    Args:
        data: Input data
        dtype: Optional data type
        
    Returns:
        NumPy array
    """
    try:
        if dtype is not None:
            return np.asarray(data, dtype=dtype)
        return np.asarray(data)
    except Exception as e:
        if dtype is None:
            return np.asarray(data, dtype=np.float64)
        raise e


def safe_linalg_eig(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors with compatibility handling.
    
    This is crucial for the 3D molecular descriptor calculations which
    rely on eigenvalue decomposition of coordinate matrices.
    
    Args:
        matrix: Input matrix for eigenvalue decomposition
        
    Returns:
        Tuple of (eigenvalues, eigenvectors)
        
    Raises:
        ValueError: If eigenvalue computation fails
    """
    try:
        # Ensure we have a proper matrix
        matrix = safe_asarray(matrix, dtype=np.float64)
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(matrix)
        
        # Handle complex eigenvalues by taking real parts for molecular descriptors
        if np.iscomplexobj(eigenvals):
            eigenvals = np.real(eigenvals)
        if np.iscomplexobj(eigenvecs):
            eigenvecs = np.real(eigenvecs)
            
        return eigenvals, eigenvecs
        
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Eigenvalue computation failed: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error in eigenvalue computation: {e}")


def safe_linalg_eigvals(matrix: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalues only with compatibility handling.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Array of eigenvalues
    """
    try:
        matrix = safe_asarray(matrix, dtype=np.float64)
        eigenvals = np.linalg.eigvals(matrix)
        
        # Handle complex eigenvalues
        if np.iscomplexobj(eigenvals):
            eigenvals = np.real(eigenvals)
            
        return eigenvals
        
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Eigenvalue computation failed: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error in eigenvalue computation: {e}")


def safe_sqrt(data: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Compute square root with handling for negative values in molecular descriptors.
    
    Args:
        data: Input data
        
    Returns:
        Square root, with negative values handled appropriately
    """
    data = safe_asarray(data)
    
    # Handle negative values that might arise from numerical precision issues
    if np.any(data < 0):
        # Set small negative values to zero (numerical precision issues)
        data = np.where(data < -1e-10, np.nan, np.maximum(data, 0))
        
    return np.sqrt(data)


def safe_mean(data: np.ndarray, axis: Optional[int] = None) -> Union[np.ndarray, float]:
    """
    Compute mean with NaN handling.
    
    Args:
        data: Input array
        axis: Axis along which to compute mean
        
    Returns:
        Mean value(s)
    """
    data = safe_asarray(data)
    return np.nanmean(data, axis=axis)


def safe_std(data: np.ndarray, axis: Optional[int] = None) -> Union[np.ndarray, float]:
    """
    Compute standard deviation with NaN handling.
    
    Args:
        data: Input array
        axis: Axis along which to compute std
        
    Returns:
        Standard deviation value(s)
    """
    data = safe_asarray(data)
    return np.nanstd(data, axis=axis)


def safe_sort(data: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Sort array with compatibility handling.
    
    Args:
        data: Input array to sort
        axis: Axis along which to sort
        
    Returns:
        Sorted array
    """
    data = safe_asarray(data)
    return np.sort(data, axis=axis)


def validate_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Validate and clean molecular coordinates for descriptor calculation.
    
    Args:
        coords: Molecular coordinates array (N x 3)
        
    Returns:
        Validated coordinates array
        
    Raises:
        ValueError: If coordinates are invalid
    """
    coords = safe_asarray(coords, dtype=np.float64)
    
    if coords.ndim != 2:
        raise ValueError(f"Coordinates must be 2D array, got {coords.ndim}D")
    
    if coords.shape[1] != 3:
        raise ValueError(f"Coordinates must have 3 columns (x,y,z), got {coords.shape[1]}")
    
    if coords.shape[0] < 3:
        raise ValueError(f"Need at least 3 atoms for 3D descriptors, got {coords.shape[0]}")
    
    # Check for NaN or infinite values
    if not np.isfinite(coords).all():
        raise ValueError("Coordinates contain NaN or infinite values")
    
    return coords


def compute_covariance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute covariance matrix from coordinates for 3D molecular descriptors.
    
    Args:
        coords: Molecular coordinates (N x 3)
        
    Returns:
        3x3 covariance matrix
    """
    coords = validate_coordinates(coords)
    
    # Center the coordinates
    centered = coords - safe_mean(coords, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(centered.T)
    
    return safe_asarray(cov_matrix, dtype=np.float64)


def get_principal_moments(coords: np.ndarray) -> np.ndarray:
    """
    Get principal moments of inertia from molecular coordinates.
    
    This is the core function for 3D molecular descriptor calculations.
    
    Args:
        coords: Molecular coordinates (N x 3)
        
    Returns:
        Array of principal moments sorted in descending order
    """
    cov_matrix = compute_covariance_matrix(coords)
    eigenvals = safe_linalg_eigvals(cov_matrix)
    
    # Sort in descending order for molecular descriptor conventions
    moments = safe_sort(eigenvals)[::-1]
    
    # Ensure positive values (handle numerical precision)
    moments = np.maximum(moments, 0)
    
    return moments


# Export compatibility information
def get_numpy_info() -> dict:
    """
    Get information about NumPy version and compatibility.
    
    Returns:
        Dictionary with NumPy version information
    """
    return {
        'version': np.__version__,
        'major_version': NUMPY_VERSION[0],
        'minor_version': NUMPY_VERSION[1],
        'is_numpy_2_plus': NUMPY_2_PLUS,
        'compatibility_layer': 'molecular3d_length_descriptors.numpy_compat'
    }


# Backward compatibility aliases
asarray = safe_asarray
array = safe_array
linalg_eig = safe_linalg_eig
linalg_eigvals = safe_linalg_eigvals
sqrt = safe_sqrt
mean = safe_mean
std = safe_std
sort = safe_sort