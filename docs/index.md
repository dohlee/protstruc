# ProtStruc

[![PyPI version](https://badge.fury.io/py/protstruc.svg)](https://badge.fury.io/py/protstruc)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://dohlee.github.io/protstruc)

## What is ProtStruc?

ProtStruc is a Python package for handling protein structures, especially for deep learning applications,
through a simple, flexible, and efficient representation of protein structures.

There are many ways to represent protein structures in various deep learning applications:

|Year & Source|Category|Deep learning application|Protein structure representation|
|----|-------------------------|--------------------------------|---|
|2020, *PNAS*|Structure prediction|TrRoseTTA|**Inter-residue geometry**|
|2021, *Science*|Structure prediction|RoseTTAFold|**Inter-residue geometry**|
|2021, *Nature*|Structure prediction|AlphaFold2|**Orientation & translation** of residue frames centered at Ca's|
|2022, *Patterns*|Structure prediction|DeepAb|**Inter-residue geometry**|
|2023, *Nat. Commun.*|Antibody structure prediction|IgFold|**Oritentation & translation** of residue frames centered at Ca's|
|2022, *arXiv*|Structure generation|FoldingDiff|**Three backbone dihedrals and three bond angles**|
|2022, *NeurIPS*|Structure generation|DiffAb|**Orientation & translation** of residue frames centered at Ca's|
|2022, *Science*|Inverse-folding|ProteinMPNN|**k-nearest neighbor graph**|
|2022, *arXiv*|Inverse-folding|PiFold|**Inter-atomic/residue distance, backbone dihedrals and bond angles, orientation of residue frame, inter-residue orientations**|
|2023, *Science*|Structure prediction|[ESMFold](https://www.science.org/doi/10.1126/science.ade2574)|**Orientation & translation** of residue frames centered at Ca's|
|2023, *ICML*|Structure generation|FrameDiff|**Orientation & translation** of residue frames centered at Ca's, an additional torsion angle for oxygen atom|
||Structure generation|HERN||
|2022, *arXiv*|Sequence-structure co-design|Multi-channel Equivariant Attention Network (MEAN)||
|2023, *ICML*|Sequence-structure co-design|[Dynamic multi-channel Equivariant Attention Network (dyMEAN)](https://arxiv.org/abs/2302.00203)||