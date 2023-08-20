# ProtStruc

[![PyPI version](https://badge.fury.io/py/protstruc.svg)](https://badge.fury.io/py/protstruc)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://dohlee.github.io/protstruc)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## What is ProtStruc?

ProtStruc is a Python package for handling protein structures, especially for deep learning applications,
through a simple, flexible, and efficient representation of protein structures.

There are many ways to represent protein structures in various deep learning applications:

| Year & Source        | Category                      | Deep learning application                                                                        | Protein structure representation                                                                                                |
| -------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| 2020, _PNAS_         | Structure prediction          | [TrRoseTTA](https://doi.org/10.1073/pnas.1914677117)                                             | **Inter-residue geometry**                                                                                                      |
| 2021, _Science_      | Structure prediction          | [RoseTTAFold](https://doi.org/10.1126/science.abj8754)                                           | **Inter-residue geometry**                                                                                                      |
| 2021, _Nature_       | Structure prediction          | [AlphaFold2](https://doi.org/10.1038/s41586-021-03819-2)                                         | **Orientation & translation** of backbone frames centered at Ca's                                                               |
| 2022, _Patterns_     | Structure prediction          | [DeepAb](https://www.sciencedirect.com/science/article/pii/S2666389921002804)                    | **Inter-residue geometry**                                                                                                      |
| 2023, _Nat. Commun._ | Antibody structure prediction | [IgFold](https://doi.org/10.1038/s41467-023-38063-x)                                             | **Oritentation & translation** of backbone frames centered at Ca's                                                              |
| 2022, _arXiv_        | Structure generation          | [FoldingDiff](https://arxiv.org/abs/2209.15611)                                                  | **Three backbone dihedrals and three bond angles**                                                                              |
| 2022, _NeurIPS_      | Structure generation          | [DiffAb](https://doi.org/10.1101/2022.07.10.499510)                                              | **Orientation & translation** of backbone frames centered at Ca's                                                               |
| 2022, _Science_      | Inverse-folding               | [ProteinMPNN](https://doi.org/10.1126/science.add2187)                                           | **k-nearest neighbor graph**                                                                                                    |
| 2022, _arXiv_        | Inverse-folding               | [PiFold](https://arxiv.org/abs/2209.12643)                                                       | **Inter-atomic/residue distance, backbone dihedrals and bond angles, orientation of residue frame, inter-residue orientations** |
| 2023, _Science_      | Structure prediction          | [ESMFold](https://www.science.org/doi/10.1126/science.ade2574)                                   | **Orientation & translation** of backbone frames centered at Ca's                                                               |
| 2023, _ICML_         | Structure generation          | [FrameDiff](https://arxiv.org/abs/2302.02277)                                                    | **Orientation & translation** of backbone frames centered at Ca's, an additional torsion angle for oxygen atom                  |
| 2022, _ICML_         | Structure generation          | [Hierarchical Equivariant Refinement Network (HERN)](https://arxiv.org/abs/2207.06616)           |                                                                                                                                 |
| 2022, _arXiv_        | Sequence-structure co-design  | [Multi-channel Equivariant Attention Network (MEAN)](https://arxiv.org/abs/2208.06073)           |                                                                                                                                 |
| 2023, _ICML_         | Sequence-structure co-design  | [Dynamic multi-channel Equivariant Attention Network (dyMEAN)](https://arxiv.org/abs/2302.00203) |                                                                                                                                 |
