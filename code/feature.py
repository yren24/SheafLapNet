#!/mnt/home/chenj159/anaconda3/bin/python
import sys, os
import numpy as np
from structure import get_structure
from protein import protein
from protein import construct_feature_aux
from protein import construct_features_PH0
from protein import construct_features_PH12
from protein import construct_features_FRI

import os
# print("DEBUG CHECK:")
# print(f"Current Working Dir: {os.getcwd()}")
# print(f"Files in here: {os.listdir('.')}")
# print(f"Can I see the PDB? {os.path.exists('190610225.pdb')}")
# 1DVF ABCD D D 52 A 7.0
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
p_WT = protein(s, 'WT')
p_WT.construct_feature_global()
p_WT.construct_feature_env()
if flag_MIBPB:
    p_WT.construct_feature_MIBPB(h=0.6)
if flag_BLAST:
    p_WT.construct_feature_seq()
#----------------------------------------------------------------------------------------
p_MT = protein(s, 'MT')
p_MT.construct_feature_global()
p_MT.construct_feature_env()
if flag_MIBPB:
    p_MT.construct_feature_MIBPB(h=0.6)
if flag_BLAST:
    p_MT.construct_feature_seq()
feature_aux     = construct_feature_aux(p_WT, p_MT, flag_MIBPB=flag_MIBPB, flag_BLAST=flag_BLAST)
feature_aux_inv = construct_feature_aux(p_MT, p_WT, flag_MIBPB=flag_MIBPB, flag_BLAST=flag_BLAST)
#----------------------------------------------------------------------------------------
print('auxiliary feature size: ', feature_aux.shape)


def check_nan(obj, name):
    arr = getattr(obj, name, None)
    print(arr)
    if arr is None:
        print(f"{obj.typeFlag} {name}: None")
        return
    if np.isnan(arr).any():
        print(f"⚠️  {obj.typeFlag} {name} has NaN at indices {np.where(np.isnan(arr))}")
    else:
        print(f"✓ {obj.typeFlag} {name}: OK (no NaN)")

# -------------------------------------------------------------
# Check WT features
# -------------------------------------------------------------
check_nan(p_WT, "FeatureMIBPB")
check_nan(p_WT, "FeatureGLB")
check_nan(p_WT, "FeatureMIBPBglb")
check_nan(p_WT, "FeatureGLBother")

# -------------------------------------------------------------
# Check MT features
# -------------------------------------------------------------
check_nan(p_MT, "FeatureMIBPB")
check_nan(p_MT, "FeatureGLB")
check_nan(p_MT, "FeatureMIBPBglb")
check_nan(p_MT, "FeatureGLBother")


filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_aux.npy', 'wb')
np.save(OutFile, feature_aux)
OutFile.close()

filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_aux.npy', 'wb')
np.save(OutFile, feature_aux_inv)
OutFile.close()
#########################################################################################
c_WT_b_r_dth, c_WT_b_r_bar = p_WT.rips_complex()
c_WT_b_a, c_WT_b_a_all = p_WT.alpha_complex()
c_WT_dist_b   = p_WT.FRI_dists()
#----------------------------------------------------------------------------------------
c_MT_b_r_dth, c_MT_b_r_bar = p_MT.rips_complex()
c_MT_b_a, c_MT_b_a_all = p_MT.alpha_complex()
c_MT_dist_b   = p_MT.FRI_dists()
#----------------------------------------------------------------------------------------
feature_PH0  = construct_features_PH0(c_WT_b_r_dth, c_WT_b_r_bar, c_MT_b_r_dth, c_MT_b_r_bar)
feature_PH12 = construct_features_PH12(c_WT_b_a, c_WT_b_a_all, c_MT_b_a, c_MT_b_a_all)
feature_FRI  = construct_features_FRI(p_WT,          p_MT,
                                      c_WT_dist_b,   c_MT_dist_b)
#----------------------------------------------------------------------------------------
feature_PH0_inv  = construct_features_PH0(c_MT_b_r_dth, c_MT_b_r_bar, c_WT_b_r_dth, c_WT_b_r_bar)
feature_PH12_inv = construct_features_PH12(c_MT_b_a, c_MT_b_a_all, c_WT_b_a, c_WT_b_a_all)
feature_FRI_inv  = construct_features_FRI(p_MT,          p_WT,
                                      c_MT_dist_b,   c_WT_dist_b)
#----------------------------------------------------------------------------------------
print('PH0 feature size:       ', feature_PH0.shape)
filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_PH0.npy', 'wb')
np.save(OutFile, feature_PH0)
OutFile.close()

filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_PH0.npy', 'wb')
np.save(OutFile, feature_PH0_inv)
OutFile.close()

print('PH12 feature size:      ', feature_PH12.shape)
filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_PH12.npy', 'wb')
np.save(OutFile, feature_PH12)
OutFile.close()

filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_PH12.npy', 'wb')
np.save(OutFile, feature_PH12_inv)
OutFile.close()

print('FRI feature size:       ', feature_FRI.shape)
filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_FRI.npy', 'wb')
np.save(OutFile, feature_FRI)
OutFile.close()

filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_FRI.npy', 'wb')
np.save(OutFile, feature_FRI_inv)
OutFile.close()
#########################################################################################
#sequence_representations = s.transformer()
#seq_partner1_WT = sequence_representations[0]
#seq_partner1_MT = sequence_representations[1]
#seq_partner2    = sequence_representations[2]
#
#seq_diff_WT = seq_partner1_MT - seq_partner1_WT
#seq_diff_MT = seq_partner1_WT - seq_partner1_MT
##----------------------------------------------------------------------------------------
#feature_seq = np.concatenate((seq_partner1_MT, seq_partner1_WT), axis=0)
#feature_seq = np.concatenate((feature_seq, seq_partner2), axis=0)
#feature_seq = np.concatenate((feature_seq, seq_diff_WT), axis=0)
##----------------------------------------------------------------------------------------
#feature_seq_inv = np.concatenate((seq_partner1_WT, seq_partner1_MT), axis=0)
#feature_seq_inv = np.concatenate((feature_seq_inv, seq_partner2), axis=0)
#feature_seq_inv = np.concatenate((feature_seq_inv, seq_diff_MT), axis=0)
##----------------------------------------------------------------------------------------
#print('SEQ feature size:       ', feature_seq.shape)
#OutFile = open(filename+'_seq.npy', 'wb')
#np.save(OutFile, feature_seq)
#OutFile.close()
#OutFile = open(filename_inv+'_seq.npy', 'wb')
#np.save(OutFile, feature_seq_inv)
#OutFile.close()
