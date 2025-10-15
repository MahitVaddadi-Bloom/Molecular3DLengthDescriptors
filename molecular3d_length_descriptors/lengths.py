"""
3D Coordinate Processing Module

Modernized version of the original Lengths.py with NumPy 2.x compatibility,
enhanced RDKit integration, and robust error handling for 3D molecular
coordinate generation and processing.

This module provides:
- GetLengthsFromCoords: Extract 3D lengths from coordinate arrays
- ComputeCoordsFrom2D: Generate 3D conformers from 2D molecular structures
"""

import math
import warnings
from typing import List, Tuple, Optional, Union, Any
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. 3D conformer generation will not work.")

from .numpy_compat import (
    safe_asarray,
    safe_linalg_eig,
    safe_sqrt,
    validate_coordinates,
    compute_covariance_matrix
)


def GetLengthsFromCoords(coords: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
    """
    Calculate the principal axis lengths from 3D molecular coordinates.
    
    This function performs eigenvalue decomposition on the coordinate covariance
    matrix to extract the three principal axis lengths, which characterize
    the overall 3D shape of the molecule.
    
    Args:
        coords: Molecular coordinates as numpy array (N x 3) or list of lists
               where N is the number of atoms and columns are [x, y, z]
               
    Returns:
        numpy array of three principal lengths sorted in descending order [L1, L2, L3]
        where L1 >= L2 >= L3
        
    Raises:
        ValueError: If coordinates are invalid or eigenvalue computation fails
    """
    try:
        # Validate and convert coordinates
        coords = validate_coordinates(safe_asarray(coords))
        
        # Compute covariance matrix
        cov_matrix = compute_covariance_matrix(coords)
        
        # Eigenvalue decomposition 
        eigenvals, eigenvecs = safe_linalg_eig(cov_matrix)
        
        # Take square root to get lengths (eigenvalues are variances)
        lengths = safe_sqrt(np.abs(eigenvals))
        
        # Sort in descending order
        lengths_sorted = np.sort(lengths)[::-1]
        
        return lengths_sorted
        
    except Exception as e:
        raise ValueError(f"Failed to calculate lengths from coordinates: {e}")


class ComputeCoordsFrom2D:
    """
    Generate 3D molecular coordinates from 2D molecular structures.
    
    This class provides methods to create 3D conformers from 2D molecular
    representations using RDKit's conformer generation algorithms with
    force field optimization.
    """
    
    def __init__(self, force_field: str = "MMFF94", max_iterations: int = 200):
        """
        Initialize the coordinate computation class.
        
        Args:
            force_field: Force field to use for optimization ("MMFF94" or "UFF")
            max_iterations: Maximum iterations for conformer optimization
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for 3D coordinate generation")
            
        self.force_field = force_field.upper()
        self.max_iterations = max_iterations
        
        if self.force_field not in ["MMFF94", "UFF"]:
            raise ValueError(f"Unsupported force field: {force_field}")
    
    def from_smiles(self, smiles: str, num_conformers: int = 1) -> List[np.ndarray]:
        """
        Generate 3D coordinates from SMILES string.
        
        Args:
            smiles: SMILES string representation of the molecule
            num_conformers: Number of conformers to generate
            
        Returns:
            List of coordinate arrays, each of shape (N_atoms, 3)
            
        Raises:
            ValueError: If SMILES is invalid or conformer generation fails
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            return self._generate_conformers(mol, num_conformers)
            
        except Exception as e:
            raise ValueError(f"Failed to generate coordinates from SMILES '{smiles}': {e}")
    
    def from_mol(self, mol: 'Chem.Mol', num_conformers: int = 1) -> List[np.ndarray]:
        """
        Generate 3D coordinates from RDKit Mol object.
        
        Args:
            mol: RDKit Mol object
            num_conformers: Number of conformers to generate
            
        Returns:
            List of coordinate arrays, each of shape (N_atoms, 3)
            
        Raises:
            ValueError: If molecule is invalid or conformer generation fails
        """
        try:
            if mol is None:
                raise ValueError("Invalid molecule object")
            
            # Add hydrogens if not present
            mol = Chem.AddHs(mol)
            
            return self._generate_conformers(mol, num_conformers)
            
        except Exception as e:
            raise ValueError(f"Failed to generate coordinates from molecule: {e}")
    
    def from_mol_block(self, mol_block: str, num_conformers: int = 1) -> List[np.ndarray]:
        """
        Generate 3D coordinates from MOL block string.
        
        Args:
            mol_block: MOL block string representation
            num_conformers: Number of conformers to generate
            
        Returns:
            List of coordinate arrays, each of shape (N_atoms, 3)
            
        Raises:
            ValueError: If MOL block is invalid or conformer generation fails
        """
        try:
            # Parse MOL block
            mol = Chem.MolFromMolBlock(mol_block)
            if mol is None:
                raise ValueError("Invalid MOL block")
            
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            return self._generate_conformers(mol, num_conformers)
            
        except Exception as e:
            raise ValueError(f"Failed to generate coordinates from MOL block: {e}")
    
    def _generate_conformers(self, mol: 'Chem.Mol', num_conformers: int) -> List[np.ndarray]:
        """
        Internal method to generate conformers from RDKit molecule.
        
        Args:
            mol: RDKit molecule with hydrogens added
            num_conformers: Number of conformers to generate
            
        Returns:
            List of coordinate arrays
        """
        try:
            # Generate initial 3D coordinates
            confIds = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=num_conformers,
                randomSeed=42,  # For reproducibility
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True
            )
            
            if len(confIds) == 0:
                raise ValueError("Failed to generate any conformers")
            
            # Optimize conformers with force field
            conformer_coords = []
            
            for confId in confIds:
                try:
                    # Force field optimization
                    if self.force_field == "MMFF94":
                        # Try MMFF94 first
                        if AllChem.MMFFHasAllMoleculeParams(mol):
                            AllChem.MMFFOptimizeMolecule(mol, confId=confId, maxIters=self.max_iterations)
                        else:
                            # Fallback to UFF if MMFF94 parameters not available
                            AllChem.UFFOptimizeMolecule(mol, confId=confId, maxIters=self.max_iterations)
                    else:  # UFF
                        AllChem.UFFOptimizeMolecule(mol, confId=confId, maxIters=self.max_iterations)
                    
                    # Extract coordinates
                    conformer = mol.GetConformer(confId)
                    coords = []
                    for i in range(mol.GetNumAtoms()):
                        pos = conformer.GetAtomPosition(i)
                        coords.append([pos.x, pos.y, pos.z])
                    
                    coords_array = safe_asarray(coords, dtype=np.float64)
                    conformer_coords.append(coords_array)
                    
                except Exception as e:
                    warnings.warn(f"Failed to optimize conformer {confId}: {e}")
                    continue
            
            if len(conformer_coords) == 0:
                raise ValueError("No conformers could be successfully optimized")
            
            return conformer_coords
            
        except Exception as e:
            raise ValueError(f"Conformer generation failed: {e}")
    
    def compute_lengths_from_smiles(self, smiles: str) -> np.ndarray:
        """
        Convenience method to compute principal lengths directly from SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Array of three principal lengths [L1, L2, L3]
        """
        coords_list = self.from_smiles(smiles, num_conformers=1)
        return GetLengthsFromCoords(coords_list[0])
    
    def compute_lengths_from_mol(self, mol: 'Chem.Mol') -> np.ndarray:
        """
        Convenience method to compute principal lengths directly from RDKit Mol.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Array of three principal lengths [L1, L2, L3]
        """
        coords_list = self.from_mol(mol, num_conformers=1)
        return GetLengthsFromCoords(coords_list[0])


# Convenience functions for backward compatibility and ease of use
def generate_3d_coords_from_smiles(smiles: str, force_field: str = "MMFF94") -> np.ndarray:
    """
    Generate 3D coordinates from SMILES string.
    
    Args:
        smiles: SMILES string
        force_field: Force field for optimization
        
    Returns:
        3D coordinates array (N_atoms, 3)
    """
    generator = ComputeCoordsFrom2D(force_field=force_field)
    coords_list = generator.from_smiles(smiles)
    return coords_list[0]


def compute_principal_lengths_from_smiles(smiles: str, force_field: str = "MMFF94") -> np.ndarray:
    """
    Compute principal lengths directly from SMILES string.
    
    Args:
        smiles: SMILES string
        force_field: Force field for optimization
        
    Returns:
        Array of three principal lengths [L1, L2, L3]
    """
    generator = ComputeCoordsFrom2D(force_field=force_field)
    return generator.compute_lengths_from_smiles(smiles)


def get_rdkit_availability() -> bool:
    """
    Check if RDKit is available.
    
    Returns:
        True if RDKit is available, False otherwise
    """
    return RDKIT_AVAILABLE


def get_supported_force_fields() -> List[str]:
    """
    Get list of supported force fields.
    
    Returns:
        List of supported force field names
    """
    return ["MMFF94", "UFF"]


# Export all functions and classes
__all__ = [
    'GetLengthsFromCoords',
    'ComputeCoordsFrom2D',
    'generate_3d_coords_from_smiles',
    'compute_principal_lengths_from_smiles',
    'get_rdkit_availability',
    'get_supported_force_fields'
]