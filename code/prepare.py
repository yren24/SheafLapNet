#!/mnt/home/chenj159/anaconda3/bin/python
import sys, os
import numpy as np
from structure import get_structure
from protein import protein

# 1DVF ABCD D D 52 A 7.0
#          arguement    example
PDBid    = sys.argv[1] # 1KBH
Chains   = sys.argv[2] # A
Chain    = sys.argv[3] # A
resWT    = sys.argv[4] # L
resID    = sys.argv[5] # 37
resMT    = sys.argv[6] # W
pH       = sys.argv[7] # 7.0

s = get_structure(PDBid, Chains, Chain, resWT, resID, resMT, pH=pH, onlyBLAST=True)
s.readFASTA()
s.writeFASTA()
#########################################################################################
p_WT = protein(s, 'WT', onlyBLAST=True)
p_WT.runBLAST()

#----------------------------------------------------------------------------------------
p_MT = protein(s, 'MT', onlyBLAST=True)
p_MT.runBLAST()
