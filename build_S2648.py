import numpy as np 
import os 
import re
import time
import multiprocessing as mp
import sys
import argparse

def dataset_list(filename):
    dataset = []
    fp = open(filename)
    for line in fp:
        line_split = re.split(',|\n', line)
        dataset.append(line_split)
    fp.close()
    return dataset

parser = argparse.ArgumentParser(description='build X and y values of data')
parser.add_argument('--data', type=str, default='S2648', help='input data name')
args = parser.parse_args()

data = dataset_list(f'./{args.data}/{args.data}.txt')

os.chdir(f'./S2648/')

feat_aux = []
feat_ph0 = []
feat_ph12 = []
feat_fri = []
feat_lap = []
feat_esm = []

y_ = []

seq_list = []
for i in range(len(data)):
    
    ilist = data[i]
    PDBid, Antibody, Chain, resWT, resID, resMT, pH, ddG = ilist[0], ilist[1], ilist[2], ilist[3], ilist[4], ilist[5], ilist[6], float(ilist[7])

    curr_dir = "features/{}_{}_{}_{}_{}/".format(PDBid, Chain, resWT, resID, resMT)
    os.chdir("features/{}_{}_{}_{}_{}".format(PDBid, Chain, resWT, resID, resMT))

    filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
    filename_inv = PDBid+'_'+Chain+'_'+resMT+'_'+resID+'_'+resWT
    
    try:
        # aux = np.load(filename+'_aux.npy', allow_pickle=True)
        # aux_inv = np.load(filename_inv+'_aux.npy', allow_pickle=True)

        # fri = np.load(filename+'_FRI.npy', allow_pickle=True)
        # fri_inv = np.load(filename_inv+'_FRI.npy', allow_pickle=True)

        # #ph0 = np.load(filename+'_PH0.npy', allow_pickle=True)
        # #ph0_inv = np.load(filename_inv+'_PH0.npy', allow_pickle=True)

        # ph12 = np.load(filename+'_PH12.npy', allow_pickle=True)
        # ph12_inv = np.load(filename_inv+'_PH12.npy', allow_pickle=True)

 

        lap = np.load(filename+'_Lap_sheaf.npy', allow_pickle=True)
        lap_inv = np.load(filename_inv+'_Lap_sheaf.npy', allow_pickle=True)

        # esm = np.load(filename+'_seq.npy', allow_pickle=True)
        # esm_inv = np.load(filename_inv+'_seq.npy', allow_pickle=True)

        os.chdir("../../")
        # feat_aux.append(aux)
        # feat_aux.append(aux_inv)
        # #feat_ph0.append(ph0)
        # #feat_ph0.append(ph0_inv)
        # feat_fri.append(fri)
        # feat_fri.append(fri_inv)
        # feat_ph12.append(ph12)
        # feat_ph12.append(ph12_inv)
        # feat_esm.append(esm)
        # feat_esm.append(esm_inv)
        feat_lap.append(lap)
        feat_lap.append(lap_inv)


        

        y_.append(ddG)
        y_.append(-ddG)
        
    except:
        print(PDBid, Antibody, Chain, resWT, resID, resMT)
        os.chdir("../../")
        continue

# feat_aux = np.array(feat_aux)
# feat_fri = np.array(feat_fri)
#feat_ph0 = np.array(feat_ph0)
# feat_ph12 = np.array(feat_ph12)
feat_lap = np.array(feat_lap)
# feat_esm = np.array(feat_esm)

y_ = np.array(y_)
y_ = np.round(y_, 3)

# print(np.shape(feat_aux))
# print(np.shape(feat_fri))
# #print(np.shape(feat_ph0))
# #print(np.shape(feat_ph12))
# print(np.shape(feat_lap))
# print(np.shape(feat_esm))

# print(np.shape(y_))

# np.save(f'X_{args.data}_aux.npy', feat_aux)
# np.save(f'X_{args.data}_FRI.npy', feat_fri)
#np.save(f'X_{args.data}_PH0.npy', feat_ph0)
# np.save(f'X_{args.data}_PH12.npy', feat_ph12)
np.save(f'X_{args.data}_Lap_sheaf.npy', feat_lap)
# np.save(f'X_{args.data}_ESM.npy', feat_esm)

# np.save(f'Y_{args.data}.npy', y_)

#X_val1 = np.load(f'X_{args.data}_aux.npy')
#X_val2 = np.load(f'X_{args.data}_FRI.npy')
#X_val3 = np.load(f'X_{args.data}_PH0.npy')
#X_val4 = np.load(f'X_{args.data}_PH12.npy')
#X_val5 = np.load(f'X_{args.data}_ESM.npy')
#X_val6 = np.load(f'X_{args.data}_Lap_b.npy')
#X_val = np.concatenate((X_val1, X_val2), axis=1)
#X_val = np.concatenate((X_val,  X_val3), axis=1)
#X_val = np.concatenate((X_val,  X_val4), axis=1)
#np.save(f'X_val_{args.data}.npy', X_val)

#X_val = np.concatenate((X_val5,  X_val6), axis=1)
#np.save(f'X_val_{args.data}_Lap_ESM.npy', X_val)
