import enum

# fmt: off
"""
This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2013 Stanford University and the Authors.
Authors: Peter Eastman
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
non_standard_residue_substitutions = {
    '2AS': 'ASP', '3AH': 'HIS', '5HP': 'GLU', 'ACL': 'ARG', 'AGM': 'ARG', 'AIB': 'ALA', 'ALM': 'ALA', 'ALO': 'THR', 'ALY': 'LYS', 'ARM': 'ARG',  # noqa: E501
    'ASA': 'ASP', 'ASB': 'ASP', 'ASK': 'ASP', 'ASL': 'ASP', 'ASQ': 'ASP', 'AYA': 'ALA', 'BCS': 'CYS', 'BHD': 'ASP', 'BMT': 'THR', 'BNN': 'ALA',  # noqa: E501
    'BUC': 'CYS', 'BUG': 'LEU', 'C5C': 'CYS', 'C6C': 'CYS', 'CAS': 'CYS', 'CCS': 'CYS', 'CEA': 'CYS', 'CGU': 'GLU', 'CHG': 'ALA', 'CLE': 'LEU', 'CME': 'CYS',  # noqa: E501
    'CSD': 'ALA', 'CSO': 'CYS', 'CSP': 'CYS', 'CSS': 'CYS', 'CSW': 'CYS', 'CSX': 'CYS', 'CXM': 'MET', 'CY1': 'CYS', 'CY3': 'CYS', 'CYG': 'CYS',  # noqa: E501
    'CYM': 'CYS', 'CYQ': 'CYS', 'DAH': 'PHE', 'DAL': 'ALA', 'DAR': 'ARG', 'DAS': 'ASP', 'DCY': 'CYS', 'DGL': 'GLU', 'DGN': 'GLN', 'DHA': 'ALA',  # noqa: E501
    'DHI': 'HIS', 'DIL': 'ILE', 'DIV': 'VAL', 'DLE': 'LEU', 'DLY': 'LYS', 'DNP': 'ALA', 'DPN': 'PHE', 'DPR': 'PRO', 'DSN': 'SER', 'DSP': 'ASP',  # noqa: E501
    'DTH': 'THR', 'DTR': 'TRP', 'DTY': 'TYR', 'DVA': 'VAL', 'EFC': 'CYS', 'FLA': 'ALA', 'FME': 'MET', 'GGL': 'GLU', 'GL3': 'GLY', 'GLZ': 'GLY',  # noqa: E501
    'GMA': 'GLU', 'GSC': 'GLY', 'HAC': 'ALA', 'HAR': 'ARG', 'HIC': 'HIS', 'HIP': 'HIS', 'HMR': 'ARG', 'HPQ': 'PHE', 'HTR': 'TRP', 'HYP': 'PRO',  # noqa: E501
    'IAS': 'ASP', 'IIL': 'ILE', 'IYR': 'TYR', 'KCX': 'LYS', 'LLP': 'LYS', 'LLY': 'LYS', 'LTR': 'TRP', 'LYM': 'LYS', 'LYZ': 'LYS', 'MAA': 'ALA', 'MEN': 'ASN',  # noqa: E501
    'MHS': 'HIS', 'MIS': 'SER', 'MLE': 'LEU', 'MPQ': 'GLY', 'MSA': 'GLY', 'MSE': 'MET', 'MVA': 'VAL', 'NEM': 'HIS', 'NEP': 'HIS', 'NLE': 'LEU',  # noqa: E501
    'NLN': 'LEU', 'NLP': 'LEU', 'NMC': 'GLY', 'OAS': 'SER', 'OCS': 'CYS', 'OMT': 'MET', 'PAQ': 'TYR', 'PCA': 'GLU', 'PEC': 'CYS', 'PHI': 'PHE',  # noqa: E501
    'PHL': 'PHE', 'PR3': 'CYS', 'PRR': 'ALA', 'PTR': 'TYR', 'PYX': 'CYS', 'SAC': 'SER', 'SAR': 'GLY', 'SCH': 'CYS', 'SCS': 'CYS', 'SCY': 'CYS',  # noqa: E501
    'SEL': 'SER', 'SEP': 'SER', 'SET': 'SER', 'SHC': 'CYS', 'SHR': 'LYS', 'SMC': 'CYS', 'SOC': 'CYS', 'STY': 'TYR', 'SVA': 'SER', 'TIH': 'ALA',  # noqa: E501
    'TPL': 'TRP', 'TPO': 'THR', 'TPQ': 'ALA', 'TRG': 'LYS', 'TRO': 'TRP', 'TYB': 'TYR', 'TYI': 'TYR', 'TYQ': 'TYR', 'TYS': 'TYR', 'TYY': 'TYR'   # noqa: E501
}

ressymb_to_resindex = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,
}
resindex_to_oneletter = {v: k for k, v in ressymb_to_resindex.items()}

class AA(enum.IntEnum):
    ALA = 0
    CYS = 1
    ASP = 2
    GLU = 3
    PHE = 4
    GLY = 5
    HIS = 6
    ILE = 7
    LYS = 8
    LEU = 9
    MET = 10
    ASN = 11
    PRO = 12
    GLN = 13
    ARG = 14
    SER = 15
    THR = 16
    VAL = 17
    TRP = 18
    TYR = 19
    UNK = 20

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and len(value) == 3:  # three representation
            if value in non_standard_residue_substitutions:
                value = non_standard_residue_substitutions[value]
            if value in cls._member_names_:
                return getattr(cls, value)
        elif isinstance(value, str) and len(value) == 1:  # one representation
            if value in ressymb_to_resindex:
                return cls(ressymb_to_resindex[value])

        return super()._missing_(value)

    def __str__(self):
        return self.name

    def oneletter(self):
        return resindex_to_oneletter[self.value]

    @classmethod
    def is_aa(cls, value):
        return (
            (value in ressymb_to_resindex)
            or (value in non_standard_residue_substitutions)
            or (value in cls._member_names_)
            or (value in cls._member_map_.values())
        )
    
standard_aa_names = [AA(i).name for i in range(20)]

# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
restype_to_heavyatom_names = {
    AA.ALA: ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.ARG: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.ASN: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.ASP: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.CYS: ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.GLN: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.GLU: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.GLY: ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.HIS: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.ILE: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.LEU: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.LYS: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.MET: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.PHE: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.PRO: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.SER: ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.THR: ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.TRP: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT'],  # noqa: E241, E501
    AA.TYR: ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    '', 'OXT'],  # noqa: E241, E501
    AA.VAL: ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],  # noqa: E241, E501
    AA.UNK: ['',  '',   '',  '',  '',   '',    '',    '',    '',    '',    '',    '',    '',    '',    ''],  # noqa: E241, E501
}

class ATOM(enum.IntEnum):
    N = 0
    CA = 1
    C = 2
    O = 3  # noqa: E741
    CB = 4

    def __str__(self):
        return self.name
