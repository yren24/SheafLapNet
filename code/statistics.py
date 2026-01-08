import sys, os
import numpy as np
from PPIstructure import get_structure
from PPIprotein import protein
from PPIprotein import construct_feature_aux
from PPIprotein import construct_features_PH0
from PPIprotein import construct_features_PH12

# 1DVF AB CD D D 52 A 7.0
#          arguement    example
PDBid    = sys.argv[1] # 1KBH
Chains   = sys.argv[2] # A
Chain    = sys.argv[3] # A
resWT    = sys.argv[4] # L
resID    = sys.argv[5] # 37
resMT    = sys.argv[6] # W
pH       = sys.argv[7] # 7.0

s = get_structure(PDBid, Chains, Chain, resWT, resID, resMT, pH)
if os.path.exists(PDBid+'_distance_distribution.txt'):
    os.system('rm '+PDBid+'_distance_distribution.txt')
cutoffs = ['10.', '11.', '12.', '13.', '14.', '15.', '16.', '17.']
for cutoff in cutoffs:
    s.cutoff_PDB_file(cutoff)
