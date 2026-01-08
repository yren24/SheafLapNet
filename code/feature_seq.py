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

flag_BLAST = True
flag_MIBPB = True

s = get_structure(PDBid, Chains, Chain, resWT, resID, resMT, pH=pH)
s.generateMutedPDBs()
s.generateMutedPQRs()
s.readFASTA()
if flag_BLAST:
    s.writeFASTA()

#########################################################################################
sequence_representations = s.transformer()
seq_partner1_WT = sequence_representations[0]
seq_partner1_MT = sequence_representations[1]

seq_diff_WT = seq_partner1_MT - seq_partner1_WT
seq_diff_MT = seq_partner1_WT - seq_partner1_MT
##----------------------------------------------------------------------------------------
feature_seq = np.concatenate((seq_partner1_MT, seq_partner1_WT), axis=0)
feature_seq = np.concatenate((feature_seq, seq_diff_WT), axis=0)
##----------------------------------------------------------------------------------------
feature_seq_inv = np.concatenate((seq_partner1_WT, seq_partner1_MT), axis=0)
feature_seq_inv = np.concatenate((feature_seq_inv, seq_diff_MT), axis=0)
##----------------------------------------------------------------------------------------
print('SEQ feature size:       ', feature_seq.shape)
filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_seq.npy', 'wb')
np.save(OutFile, feature_seq)
OutFile.close()
filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_seq.npy', 'wb')
np.save(OutFile, feature_seq_inv)
OutFile.close()
