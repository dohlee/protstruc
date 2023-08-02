# Welcome to ProtStruc documentation!

## What is ProtStruc?

ProtStruc is a Python package for handling protein structures, especially for deep learning applications.

## Examples


### Initialize a single protein structure from a PDB file
```python
import torch
import protstruc as ps

pdb_file = '1a0s.pdb'
batch = ps.StructureBatch.from_pdb(pdb_file)
```

### Initialize a batch of protein structures from a list of PDB files
```python
import torch
import protstruc as ps

pdb_files = ['1a0s.pdb', '1a1s.pdb', '1a2s.pdb', '1a3s.pdb', '1a4s.pdb']
batch = ps.StructureBatch.from_pdb(pdb_files)
```

### Initialize a batch of protein structures from backbone (or full atom) xyz coordinates
```python
import torch
import protstruc as ps

batch_size, max_n_residues = 32, 100
max_n_atoms_per_residue = 10

xyz = torch.randn(batch_size, max_n_residues, max_n_atoms_per_residue, 3)

batch = ps.StructureBatch.from_xyz(xyz)
```