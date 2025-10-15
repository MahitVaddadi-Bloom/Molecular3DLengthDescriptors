#!/usr/bin/env python3
"""
Molecular3DLengthDescriptors CLI

A comprehensive command-line interface for computing 3D molecular descriptors
from various input formats including SMILES, MOL files, and coordinate files.

Features:
- Multiple input formats (SMILES, SDF, MOL, CSV, JSON)
- Batch processing capabilities
- Rich progress bars and colored output
- Flexible output formats (JSON, CSV, table)
- Conformer generation and optimization
- NumPy 2.x compatibility
"""

import sys
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TextIO
import warnings

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint
import numpy as np

# Import the molecular descriptor functions
try:
    from .descriptors import CalcAllDesc, Flatness, Cubeularity, Plateularity, ShortOverLong, MediumOverLong
    from .lengths import (
        GetLengthsFromCoords, 
        ComputeCoordsFrom2D, 
        get_rdkit_availability,
        compute_principal_lengths_from_smiles
    )
    from .numpy_compat import get_numpy_info, validate_coordinates, safe_asarray
    from .version import __version__
except ImportError:
    # Handle relative imports when run as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from descriptors import CalcAllDesc, Flatness, Cubeularity, Plateularity, ShortOverLong, MediumOverLong
    from lengths import (
        GetLengthsFromCoords, 
        ComputeCoordsFrom2D, 
        get_rdkit_availability,
        compute_principal_lengths_from_smiles
    )
    from numpy_compat import get_numpy_info, validate_coordinates, safe_asarray
    from version import __version__

# Initialize Rich console
console = Console()

# RDKit availability check
RDKIT_AVAILABLE = get_rdkit_availability()
if RDKIT_AVAILABLE:
    try:
        from rdkit import Chem
        from rdkit.Chem import SDMolSupplier, SmilesMolSupplier
    except ImportError:
        RDKIT_AVAILABLE = False


class MolecularDescriptorCalculator:
    """Main calculator class for 3D molecular descriptors."""
    
    def __init__(self, force_field: str = "MMFF94", max_iterations: int = 200):
        """
        Initialize the calculator.
        
        Args:
            force_field: Force field for 3D optimization ("MMFF94" or "UFF")
            max_iterations: Maximum optimization iterations
        """
        self.force_field = force_field
        self.max_iterations = max_iterations
        
        if RDKIT_AVAILABLE:
            self.coord_generator = ComputeCoordsFrom2D(force_field, max_iterations)
        else:
            self.coord_generator = None
    
    def calculate_from_coords(self, coords: Union[np.ndarray, List[List[float]]]) -> Dict[str, float]:
        """Calculate all descriptors from 3D coordinates."""
        try:
            coords = validate_coordinates(safe_asarray(coords))
            descriptors = CalcAllDesc(coords)
            
            # Add principal lengths
            lengths = GetLengthsFromCoords(coords)
            descriptors.update({
                'Length1': float(lengths[0]),
                'Length2': float(lengths[1]), 
                'Length3': float(lengths[2])
            })
            
            return descriptors
        except Exception as e:
            raise ValueError(f"Failed to calculate descriptors: {e}")
    
    def calculate_from_smiles(self, smiles: str) -> Dict[str, Any]:
        """Calculate descriptors from SMILES string."""
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is required for SMILES processing")
        
        try:
            # Generate 3D coordinates
            coords_list = self.coord_generator.from_smiles(smiles)
            coords = coords_list[0]  # Use first conformer
            
            # Calculate descriptors
            descriptors = self.calculate_from_coords(coords)
            descriptors['SMILES'] = smiles
            descriptors['NumAtoms'] = len(coords)
            
            return descriptors
        except Exception as e:
            raise ValueError(f"Failed to process SMILES '{smiles}': {e}")


def load_coordinates_from_file(filepath: Path) -> List[np.ndarray]:
    """Load coordinates from various file formats."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    coords_list = []
    
    if filepath.suffix.lower() == '.csv':
        # CSV format: each row is [x, y, z] coordinates
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            coords = []
            for row in reader:
                if len(row) >= 3:
                    try:
                        coords.append([float(row[0]), float(row[1]), float(row[2])])
                    except ValueError:
                        continue
            if coords:
                coords_list.append(safe_asarray(coords))
    
    elif filepath.suffix.lower() == '.json':
        # JSON format: array of [x, y, z] coordinates
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                coords_list.append(safe_asarray(data))
    
    elif filepath.suffix.lower() in ['.txt', '.dat']:
        # Text format: space/tab separated x y z coordinates
        with open(filepath, 'r') as f:
            coords = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
            if coords:
                coords_list.append(safe_asarray(coords))
    
    else:
        raise ValueError(f"Unsupported coordinate file format: {filepath.suffix}")
    
    return coords_list


def load_molecules_from_file(filepath: Path) -> List[str]:
    """Load SMILES from various molecular file formats."""
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for molecular file processing")
    
    filepath = Path(filepath)
    smiles_list = []
    
    if filepath.suffix.lower() == '.smi':
        # SMILES file
        with open(filepath, 'r') as f:
            for line in f:
                smiles = line.strip().split()[0]  # Take first column
                if smiles:
                    smiles_list.append(smiles)
    
    elif filepath.suffix.lower() == '.sdf':
        # SDF file
        supplier = SDMolSupplier(str(filepath))
        for mol in supplier:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
    
    elif filepath.suffix.lower() == '.csv':
        # CSV with SMILES column
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            smiles_col = None
            for col in reader.fieldnames:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break
            
            if smiles_col:
                for row in reader:
                    smiles = row[smiles_col].strip()
                    if smiles:
                        smiles_list.append(smiles)
            else:
                raise ValueError("No SMILES column found in CSV file")
    
    else:
        raise ValueError(f"Unsupported molecular file format: {filepath.suffix}")
    
    return smiles_list


def format_output(results: List[Dict[str, Any]], output_format: str) -> str:
    """Format results for output."""
    if output_format == 'json':
        return json.dumps(results, indent=2)
    
    elif output_format == 'csv':
        if not results:
            return ""
        
        output = []
        fieldnames = list(results[0].keys())
        
        # CSV header
        output.append(','.join(fieldnames))
        
        # CSV rows
        for result in results:
            row = []
            for field in fieldnames:
                value = result.get(field, '')
                if isinstance(value, float):
                    row.append(f"{value:.6f}")
                else:
                    row.append(str(value))
            output.append(','.join(row))
        
        return '\n'.join(output)
    
    elif output_format == 'table':
        if not results:
            return "No results to display"
        
        table = Table(title="3D Molecular Descriptors")
        
        # Add columns
        for key in results[0].keys():
            table.add_column(key, style="cyan")
        
        # Add rows
        for result in results:
            row = []
            for key in results[0].keys():
                value = result.get(key, '')
                if isinstance(value, float):
                    row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            table.add_row(*row)
        
        console = Console()
        with console.capture() as capture:
            console.print(table)
        return capture.get()
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def cli(verbose):
    """
    Molecular3DLengthDescriptors CLI
    
    Calculate 3D molecular shape descriptors from coordinates or SMILES.
    """
    if verbose:
        click.echo(f"Molecular3DLengthDescriptors v{__version__}")
        numpy_info = get_numpy_info()
        click.echo(f"NumPy v{numpy_info['version']} ({'2.x compatible' if numpy_info['is_numpy_2_plus'] else '1.x'})")
        click.echo(f"RDKit: {'Available' if RDKIT_AVAILABLE else 'Not available'}")


@cli.command()
@click.argument('smiles', type=str)
@click.option('--force-field', '-f', default='MMFF94', type=click.Choice(['MMFF94', 'UFF']), 
              help='Force field for 3D optimization')
@click.option('--output-format', '-o', default='table', type=click.Choice(['json', 'csv', 'table']),
              help='Output format')
def single(smiles, force_field, output_format):
    """Calculate descriptors for a single SMILES string."""
    if not RDKIT_AVAILABLE:
        console.print("[red]Error: RDKit is required for SMILES processing[/red]")
        return
    
    try:
        with console.status("[bold green]Calculating descriptors..."):
            calculator = MolecularDescriptorCalculator(force_field)
            result = calculator.calculate_from_smiles(smiles)
        
        output = format_output([result], output_format)
        console.print(output)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.File('w'), default='-', 
              help='Output file (default: stdout)')
@click.option('--output-format', '-f', default='csv', type=click.Choice(['json', 'csv', 'table']),
              help='Output format')
@click.option('--force-field', default='MMFF94', type=click.Choice(['MMFF94', 'UFF']),
              help='Force field for 3D optimization')
@click.option('--input-type', type=click.Choice(['auto', 'smiles', 'coordinates']),
              default='auto', help='Input file type')
def batch(input_file, output, output_format, force_field, input_type):
    """Process multiple molecules from a file."""
    input_path = Path(input_file)
    
    try:
        # Determine input type
        if input_type == 'auto':
            if input_path.suffix.lower() in ['.smi', '.sdf', '.csv'] and RDKIT_AVAILABLE:
                input_type = 'smiles'
            elif input_path.suffix.lower() in ['.csv', '.json', '.txt', '.dat']:
                input_type = 'coordinates'
            else:
                raise ValueError("Cannot determine input type automatically")
        
        if input_type == 'smiles' and not RDKIT_AVAILABLE:
            console.print("[red]Error: RDKit is required for SMILES processing[/red]")
            return
        
        calculator = MolecularDescriptorCalculator(force_field)
        results = []
        
        # Load input data
        if input_type == 'smiles':
            smiles_list = load_molecules_from_file(input_path)
            total = len(smiles_list)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing molecules...", total=total)
                
                for i, smiles in enumerate(smiles_list):
                    try:
                        result = calculator.calculate_from_smiles(smiles)
                        result['Index'] = i + 1
                        results.append(result)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to process SMILES {i+1}: {e}[/yellow]")
                    
                    progress.update(task, advance=1)
        
        else:  # coordinates
            coords_list = load_coordinates_from_file(input_path)
            total = len(coords_list)
            
            with Progress(
                SpinnerColumn(), 
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing coordinate sets...", total=total)
                
                for i, coords in enumerate(coords_list):
                    try:
                        result = calculator.calculate_from_coords(coords)
                        result['Index'] = i + 1
                        result['NumAtoms'] = len(coords)
                        results.append(result)
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to process coordinates {i+1}: {e}[/yellow]")
                    
                    progress.update(task, advance=1)
        
        # Output results
        if results:
            output_text = format_output(results, output_format)
            output.write(output_text)
            
            console.print(f"[green]Successfully processed {len(results)} molecules[/green]")
        else:
            console.print("[yellow]No molecules were successfully processed[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument('coords_file', type=click.Path(exists=True))
@click.option('--output-format', '-f', default='table', type=click.Choice(['json', 'csv', 'table']),
              help='Output format')
def coords(coords_file, output_format):
    """Calculate descriptors from coordinate file."""
    try:
        coords_list = load_coordinates_from_file(Path(coords_file))
        
        if not coords_list:
            console.print("[red]Error: No valid coordinates found in file[/red]")
            return
        
        calculator = MolecularDescriptorCalculator()
        results = []
        
        for i, coords in enumerate(coords_list):
            try:
                result = calculator.calculate_from_coords(coords)
                result['Index'] = i + 1
                result['NumAtoms'] = len(coords)
                results.append(result)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to process coordinates {i+1}: {e}[/yellow]")
        
        if results:
            output = format_output(results, output_format)
            console.print(output)
        else:
            console.print("[yellow]No coordinate sets were successfully processed[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
def info():
    """Display system and package information."""
    # Package info
    info_panel = Panel.fit(
        f"""[bold blue]Molecular3DLengthDescriptors v{__version__}[/bold blue]

[bold]3D Molecular Descriptors:[/bold]
• Flatness: Measure of molecular planarity (0-1)
• Cubeularity: Measure of cubic symmetry (0-1)  
• Plateularity: Measure of plate-like shape (0+)
• ShortOverLong: Short/long axis ratio (0-1)
• MediumOverLong: Medium/long axis ratio (0-1)
• Principal Lengths: L1, L2, L3 (sorted desc)

[bold]Supported Input Formats:[/bold]
• SMILES strings (.smi files)
• SDF molecular files (.sdf)
• CSV files (with SMILES or coordinates)
• JSON coordinate files (.json)
• Text coordinate files (.txt, .dat)

[bold]Features:[/bold]
• NumPy 2.x compatibility with fallback support
• Robust 3D conformer generation using RDKit
• MMFF94 and UFF force field optimization
• Batch processing with progress bars
• Multiple output formats (JSON, CSV, table)
""",
        title="Package Information",
        border_style="blue"
    )
    
    console.print(info_panel)
    
    # System info
    numpy_info = get_numpy_info()
    system_info = f"""[bold]System Information:[/bold]
• Python: {sys.version.split()[0]}
• NumPy: {numpy_info['version']} ({'2.x compatible' if numpy_info['is_numpy_2_plus'] else '1.x mode'})
• RDKit: {'Available' if RDKIT_AVAILABLE else 'Not available (install with: pip install rdkit)'}
• Platform: {sys.platform}"""
    
    system_panel = Panel.fit(system_info, title="System Status", border_style="green")
    console.print(system_panel)


@cli.command()
def examples():
    """Show usage examples."""
    examples_text = """[bold]Basic Usage Examples:[/bold]

[bold blue]1. Single SMILES calculation:[/bold blue]
mol3d-descriptors single "CCO"

[bold blue]2. Process SMILES file:[/bold blue]
mol3d-descriptors batch molecules.smi --output results.csv

[bold blue]3. Process coordinate file:[/bold blue]
mol3d-descriptors coords coordinates.csv --output-format json

[bold blue]4. Use different force field:[/bold blue]
mol3d-descriptors single "c1ccccc1" --force-field UFF

[bold blue]5. Batch processing with table output:[/bold blue]
mol3d-descriptors batch molecules.sdf --output-format table

[bold blue]6. Get package information:[/bold blue]
mol3d-descriptors info

[bold]Input File Formats:[/bold]

[bold green]SMILES files (.smi):[/bold green]
CCO
c1ccccc1
CN(C)C

[bold green]CSV with coordinates:[/bold green]
x,y,z
0.0,0.0,0.0
1.0,0.0,0.0
0.0,1.0,0.0

[bold green]JSON coordinates:[/bold green]
[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]"""
    
    examples_panel = Panel.fit(examples_text, title="Usage Examples", border_style="cyan")
    console.print(examples_panel)


if __name__ == '__main__':
    cli()