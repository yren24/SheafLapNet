#!/mnt/home/chenj159/anaconda3/bin/python
import sys, os
import numpy as np
from structure import get_structure
from protein import protein
from protein import construct_feature_aux
from protein import construct_features_PH0
from protein import construct_features_PH12

# 1DVF AB CD D D 52 A 7.0
#          arguement    example
PDBid    = sys.argv[1] # 1KBH
Chains   = sys.argv[2] # A
Chain    = sys.argv[3] # A
resWT    = sys.argv[4] # L
resID    = sys.argv[5] # 37
resMT    = sys.argv[6] # W
pH       = sys.argv[7] # 7.0

s = get_structure(PDBid, Chains, Chain, resWT, resID, resMT, pH=pH)
s.generateMutedPDBs()
s.generateMutedPQRs()
s.readFASTA()
s.writeFASTA()
#########################################################################################
p_WT = protein(s, 'WT')
p_WT.construct_feature_seq()
#----------------------------------------------------------------------------------------
p_MT = protein(s, 'MT')
p_MT.construct_feature_seq()
