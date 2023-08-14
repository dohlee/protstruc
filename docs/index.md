# ProtStruc

[![PyPI version](https://badge.fury.io/py/protstruc.svg)](https://badge.fury.io/py/protstruc)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://dohlee.github.io/protstruc)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## What is ProtStruc?

ProtStruc is a Python package for handling protein structures, especially for deep learning applications,
through a simple, flexible, and efficient representation of protein structures.

There are many ways to represent protein structures in various deep learning applications:

|Year & Source|Category|Deep learning application|Protein structure representation|
|----|-------------------------|--------------------------------|---|
|2020, *PNAS*|Structure prediction|[TrRoseTTA](https://doi.org/10.1073/pnas.1914677117)|**Inter-residue geometry**|
|2021, *Science*|Structure prediction|[RoseTTAFold](https://doi.org/10.1126/science.abj8754)|**Inter-residue geometry**|
|2021, *Nature*|Structure prediction|[AlphaFold2](https://doi.org/10.1038/s41586-021-03819-2)|**Orientation & translation** of backbone frames centered at Ca's|
|2022, *Patterns*|Structure prediction|[DeepAb](https://www.sciencedirect.com/science/article/pii/S2666389921002804)|**Inter-residue geometry**|
|2023, *Nat. Commun.*|Antibody structure prediction|[IgFold](https://doi.org/10.1038/s41467-023-38063-x)|**Oritentation & translation** of backbone frames centered at Ca's|
|2022, *arXiv*|Structure generation|[FoldingDiff](https://arxiv.org/abs/2209.15611)|**Three backbone dihedrals and three bond angles**|
|2022, *NeurIPS*|Structure generation|[DiffAb](https://doi.org/10.1101/2022.07.10.499510)|**Orientation & translation** of backbone frames centered at Ca's|
|2022, *Science*|Inverse-folding|[ProteinMPNN](https://doi.org/10.1126/science.add2187)|**k-nearest neighbor graph**|
|2022, *arXiv*|Inverse-folding|[PiFold](https://arxiv.org/abs/2209.12643)|**Inter-atomic/residue distance, backbone dihedrals and bond angles, orientation of residue frame, inter-residue orientations**|
|2023, *Science*|Structure prediction|[ESMFold](https://www.science.org/doi/10.1126/science.ade2574)|**Orientation & translation** of backbone frames centered at Ca's|
|2023, *ICML*|Structure generation|[FrameDiff](https://arxiv.org/abs/2302.02277)|**Orientation & translation** of backbone frames centered at Ca's, an additional torsion angle for oxygen atom|
|2022, *ICML*|Structure generation|[Hierarchical Equivariant Refinement Network (HERN)](https://arxiv.org/abs/2207.06616)||
|2022, *arXiv*|Sequence-structure co-design|[Multi-channel Equivariant Attention Network (MEAN)](https://arxiv.org/abs/2208.06073)||
|2023, *ICML*|Sequence-structure co-design|[Dynamic multi-channel Equivariant Attention Network (dyMEAN)](https://arxiv.org/abs/2302.00203)||