#!/mnt/home/chenj159/anaconda3/bin/python
import sys, os
import numpy as np
from structure import get_structure
from protein import protein
from protein import construct_features_PH0
from protein import construct_features_PH12
from protein import construct_feature_aux

# 1DVF AB CD D D 52 A 7.0
#          arguement    example
PDBid    = sys.argv[1] # 1KBH
Chains   = sys.argv[2] # A
Chain    = sys.argv[3] # A
resWT    = sys.argv[4] # L
resID    = sys.argv[5] # 37
resMT    = sys.argv[6] # W
pH       = sys.argv[7] # 7.0

flag_BLAST = False
flag_MIBPB = False

s = get_structure(PDBid, Chains, Chain, resWT, resID, resMT, pH=pH)
s.generateMutedPDBs()
s.generateMutedPQRs()
s.readFASTA()

#########################################################################################
c_WT_en = protein(s, 'WT', mode='en')
feats_en_rips = c_WT_en.rips_complex_sheaf_spectra()
feats_en_alpha = c_WT_en.alpha_complex_sheaf_spectra()
feats_en_WT = np.concatenate([feats_en_rips, feats_en_alpha])

c_WT_charge = protein(s, 'WT', mode='charge')
feats_ch_rips = c_WT_charge.rips_complex_sheaf_spectra()
feats_ch_alpha = c_WT_charge.alpha_complex_sheaf_spectra()
feats_ch_WT = np.concatenate([feats_ch_rips, feats_ch_alpha])

c_WT_Lap_sheaf=feats_ch_WT #np.concatenate([feats_en_WT, feats_ch_WT])
# c_WT_Lap_b = c_WT.rips_complex_spectra()
# c_WT_Lap_sheaf0 = c_WT.rips_complex_sheaf_spectra()
# c_WT_Lap_sheaf1 = c_WT.alpha_complex_sheaf_spectra()
# c_WT_Lap_sheaf = np.concatenate([c_WT_Lap_sheaf0, c_WT_Lap_sheaf1])
#c_WT_Lap_m = c_WT.rips_complex_spectra(c_WT.atoms_m_m, c_WT.atoms_m_o)
#c_WT_Lap_s = c_WT.rips_complex_spectra(c_WT.atoms_m_m, c_WT.atoms_m_s)
#----------------------------------------------------------------------------------------
# c_MT = protein(s, 'MT')
# # c_MT_Lap_b = c_MT.rips_complex_spectra()
# c_MT_Lap_sheaf0 = c_MT.rips_complex_sheaf_spectra()
# c_MT_Lap_sheaf1 = c_MT.alpha_complex_sheaf_spectra()
# c_MT_Lap_sheaf = np.concatenate([c_MT_Lap_sheaf0, c_MT_Lap_sheaf1])



c_MT_en = protein(s, 'MT', mode='en')
feats_en_rips = c_MT_en.rips_complex_sheaf_spectra()
feats_en_alpha = c_MT_en.alpha_complex_sheaf_spectra()
feats_en_MT = np.concatenate([feats_en_rips, feats_en_alpha])

c_MT_charge = protein(s, 'MT', mode='charge')
feats_ch_rips = c_MT_charge.rips_complex_sheaf_spectra()
feats_ch_alpha = c_MT_charge.alpha_complex_sheaf_spectra()
feats_ch_MT = np.concatenate([feats_ch_rips, feats_ch_alpha])

c_MT_Lap_sheaf=feats_ch_MT#np.concatenate([feats_en_MT, feats_ch_MT])
#c_MT_Lap_m = c_MT.rips_complex_spectra(c_MT.atoms_m_m, c_MT.atoms_m_o)
#c_MT_Lap_s = c_MT.rips_complex_spectra(c_MT.atoms_m_m, c_MT.atoms_m_s)
#----------------------------------------------------------------------------------------
# feature_Lap_b = np.concatenate((c_MT_Lap_b, c_WT_Lap_b), axis=0)
# feature_Lap_b = np.concatenate((feature_Lap_b, c_MT_Lap_b-c_WT_Lap_b), axis=0)
feature_Lap_sheaf = np.concatenate((c_MT_Lap_sheaf, c_WT_Lap_sheaf), axis=0)
feature_Lap_sheaf = np.concatenate((feature_Lap_sheaf, c_MT_Lap_sheaf-c_WT_Lap_sheaf), axis=0)
#feature_Lap_m = np.concatenate((c_MT_Lap_m, c_WT_Lap_m), axis=0)
#feature_Lap_m = np.concatenate((feature_Lap_m, c_MT_Lap_m-c_WT_Lap_m), axis=0)
#feature_Lap_s = np.concatenate((c_MT_Lap_s, c_WT_Lap_s), axis=0)
#feature_Lap_s = np.concatenate((feature_Lap_s, c_MT_Lap_s-c_WT_Lap_s), axis=0)
#feature_Lap   = np.concatenate((feature_Lap_b, feature_Lap_m), axis=0)
#feature_Lap   = np.concatenate((feature_Lap,   feature_Lap_s), axis=0)
#----------------------------------------------------------------------------------------
feature_Lap_sheaf_inv = np.concatenate((c_WT_Lap_sheaf, c_MT_Lap_sheaf), axis=0)
feature_Lap_sheaf_inv = np.concatenate((feature_Lap_sheaf_inv, c_WT_Lap_sheaf-c_MT_Lap_sheaf), axis=0)
#feature_Lap_m_inv = np.concatenate((c_WT_Lap_m, c_MT_Lap_m), axis=0)
#feature_Lap_m_inv = np.concatenate((feature_Lap_m_inv, c_WT_Lap_m-c_MT_Lap_m), axis=0)
#feature_Lap_s_inv = np.concatenate((c_WT_Lap_s, c_MT_Lap_s), axis=0)
#feature_Lap_s_inv = np.concatenate((feature_Lap_s_inv, c_WT_Lap_s-c_MT_Lap_s), axis=0)
#feature_Lap_inv   = np.concatenate((feature_Lap_b_inv, feature_Lap_m_inv), axis=0)
#feature_Lap_inv   = np.concatenate((feature_Lap_inv,   feature_Lap_s_inv), axis=0)
#########################################################################################
print('Lap feature size: ', feature_Lap_sheaf.shape)

filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
OutFile = open(filename+'_Lap_sheaf.npy', 'wb')
np.save(OutFile, feature_Lap_sheaf)
OutFile.close()

filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
OutFile = open(filename_inv+'_Lap_sheaf.npy', 'wb')
np.save(OutFile, feature_Lap_sheaf_inv)
OutFile.close()
#########################################################################################
