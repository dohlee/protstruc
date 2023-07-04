# protstruc

Protein structure handling in Python made easy.

## Installation
```bash
pip install protstruc
```

## Testing
``` bash
pytest
```

## Usage
```python
from protstruc import AntibodyFvStructure

# Load a structure from a PDB file
struc = AntibodyFvStructure('1ad9_HL.pdb')

# Get inter-residue geometries
# as a dictionary of 2D matrices (n_residue, n_residue)
g = struc.inter_residue_geometry()

# Pairwise distance between Ca atoms (Symmetric)
g['d_ca']
# Pairwise distance bewteen Cb atoms (Symmetric)
g['d_cb']
# Pairwise distance between N and O atoms (Non-symmetric)
g['d_no']
# Pairwise dihedral between Ca-Cb-Cb'-Ca' atoms (Symmetric)
g['omega']
# Pairwise dihedral between N-Ca-Cb-Cb' atoms (Non-symmetric)
g['theta']
# Pairwise planar angle between Ca-Cb-Cb' atoms (Non-symmetric)
g['phi']
```