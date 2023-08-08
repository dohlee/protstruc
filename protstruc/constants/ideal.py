# ideal bond lengths
NA = 1.458  # N to Ca
AN = 1.458  # Ca to N
AC = 1.523  # Ca to C
CA = 1.523  # C to Ca
AB = 1.522  # Ca to Cb
BA = 1.522  # Cb to Ca
C_N = 1.329  # C to N of the next residue
NB = 2.447  # N to Cb
BN = 2.447  # Cb to N
CB = 2.499  # C to Cb
BC = 2.499  # Cb to C
NC = 2.460  # N to C
CN = 2.460  # C to N

CO = 1.231
OC = 1.231

# ideal planar angles
ANC = 0.615  # Ca-N-C
NAB = 1.927  # N-Ca-Cb
BAN = 1.927  # Cb-Ca-N
NAC = 1.937  # N-Ca-C
CAN = 1.937  # C-Ca-N
ACO = 2.108
OCA = 2.108

# ideal dihedral angles
BANC = -2.143  # Cb-Ca-N-C
NACO = -3.142  # note the planarity of a peptide bond

as_dict = {
    "NA": NA,
    "AN": AN,
    "AC": AC,
    "CA": CA,
    "AB": AB,
    "BA": BA,
    "C_N": C_N,
    "NB": NB,
    "BN": BN,
    "CB": CB,
    "BC": BC,
    "NC": NC,
    "CN": CN,
    "ANC": ANC,
    "NAB": NAB,
    "BAN": BAN,
    "BANC": BANC,
}
