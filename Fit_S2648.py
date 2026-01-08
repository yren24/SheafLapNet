import numpy as np 
import os 
import re
from structure import get_structure
import time
import multiprocessing as mp



def feat_job(list_, folder1, folder2):
    numCPU = 10
    os.chdir("./S2648/")
    for ilist in list_:
        folder_dir = f'{ilist[0]}_{ilist[2]}_{ilist[3]}_{ilist[4]}_{ilist[5]}'
        dst = f'{folder1}/{folder_dir}'
        fp = open(f'{folder2}/{folder_dir}.job', 'w')
        fp.write('#!/bin/bash\n')
        fp.write('#SBATCH --time=04:00:00\n')
        fp.write('#SBATCH --nodes=1\n')
        fp.write('#SBATCH -A guowei-search\n')
        fp.write('#SBATCH --ntasks-per-node=1\n')
        fp.write('#SBATCH --cpus-per-task=1\n')
        fp.write('#SBATCH --constraint=\"amd20\"\n')
        #fp.write('#SBATCH --reservation=centos_compute\n')
        fp.write('#SBATCH --mem=%dG\n'%(numCPU*4))
        fp.write(f'#SBATCH --job-name {folder_dir}\n')
        fp.write(f'#SBATCH --output=./feat_out/{folder_dir}.out\n')
        fp.write(f'cd /mnt/home/renyimi2/SheafLapNet/S2648/{dst}\n')
        # fp.write(f'cd /mnt/research/guowei-search.4/junjie-proteng/S2648/{dst}\n')
        fp.write('module purge\n')
        #fp.write('module load GCC/11.2.0 OpenMPI/4.0.3\n')
        fp.write('module load GCC/12.3.0 OpenMPI/4.1.5-GCC-12.3.0\n')
        fp.write('source ~/.bashrc\n')            # Load conda setup
        fp.write('conda activate esm_env\n')      # Activate environment
        # fp.write('python feature.py '+' '.join(ilist)+'\n')
        # fp.write('python feature_Lap.py '+' '.join(ilist)+'\n')
        fp.write('python feature_Lap.py '+' '.join(ilist)+'\n')
        fp.close()
    os.chdir("../")
    return

def run_feat(list_, folder1, folder2):
    os.chdir("./S2648/")
    # for i in range(len(list_)):
    # for i in range(len(list_)):
    for i in range(len(list_)):
        ilist = list_[i]
        folder_dir = f'{ilist[0]}_{ilist[2]}_{ilist[3]}_{ilist[4]}_{ilist[5]}'
        dst = f'{folder1}/{folder_dir}'
        filename = f'{ilist[0]}_{ilist[2]}'
        
        opp_folder_dir = f'{ilist[0]}_{ilist[2]}_{ilist[5]}_{ilist[4]}_{ilist[3]}'
        # if not os.path.exists(f'{dst}/{folder_dir}_FRI.npy') or not os.path.exists(f'{dst}/{folder_dir}_PH0.npy') \
        #     or not os.path.exists(f'{dst}/{folder_dir}_PH12.npy')  \
        #     or not os.path.exists(f'{dst}/{opp_folder_dir}_FRI.npy') or not os.path.exists(f'{dst}/{opp_folder_dir}_PH0.npy') \
        #     or not os.path.exists(f'{dst}/{opp_folder_dir}_PH12.npy') \
        #     or not os.path.exists(f'{dst}/{folder_dir}_Lap_b.npy') or not os.path.exists(f'{dst}/{opp_folder_dir}_Lap_b.npy'):
            #or os.stat(f'{dst}/{folder_dir}_Lap_b.npy').st_size == 0 or os.stat(f'{dst}/{opp_folder_dir}_Lap_b.npy').st_size == 0:
        #tmp = np.load(f'{dst}/{folder_dir}_Lap_b.npy', allow_pickle=True)
        #if np.shape(tmp)[0] != 1296:
        PDBid, Antibody, Chain, resWT, resID, resMT, *_ = ilist
        # if (PDBid, Chain, resWT, resID, resMT) == ("1KDX", "A", "Y", "640", "F"):
        # if not os.path.exists(f'{dst}/{folder_dir}_Lap_sheaf.npy'):
        os.system(f'sbatch {folder2}/{folder_dir}.job')
    os.chdir("../")
    return 

def create_blastjob(list_, folder1, folder2):
    numCPU = 10
    os.chdir("./S2648/")
    for ilist in list_:
        folder_dir = f'{ilist[0]}_{ilist[2]}_{ilist[3]}_{ilist[4]}_{ilist[5]}'
        dst = f'{folder1}/{folder_dir}'
        fp = open(f'{folder2}/{folder_dir}.job', 'w')
        fp.write('#!/bin/bash\n')
        fp.write('#SBATCH --time=8:00:00\n')
        fp.write('#SBATCH --nodes=1\n')
        fp.write('#SBATCH -A guowei-search\n')
        fp.write('#SBATCH --ntasks-per-node=1\n')
        fp.write('#SBATCH --cpus-per-task=1\n')
        fp.write('#SBATCH --constraint=\"amd20\"\n')
        fp.write('#SBATCH --mem=%dG\n'%(numCPU*4))
        fp.write(f'#SBATCH --job-name {folder_dir}\n')
        fp.write(f'#SBATCH --output=./blast_out/{folder_dir}.out\n')
        fp.write(f'cd /mnt/home/renyimi2/SheafLapNet/S2648/{dst}\n')
        # fp.write(f'cd /mnt/research/guowei-search.4/junjie-proteng/S2648/{dst}\n')
        #fp.write('source activate pytorch\n')
        fp.write('module purge\n')
        #fp.write('module load GCC/9.3.0 OpenMPI/4.0.3\n')
        #fp.write('module load BLAST+/2.10.1\n')
        fp.write('module load BLAST+/2.14.1-gompi-2023a\n')
        fp.write('source ~/.bashrc\n')            # Load conda setup
        fp.write('conda activate esm_env\n')      # Activate environment
        fp.write('python prepare.py '+' '.join(ilist)+'\n')
        #fp.write('python feature_Lap.py '+' '.join(ilist[:-1])+'\n')
        fp.close()
    os.chdir("../")
    return

def blast_jobs(list_, folder1, folder2):
    os.chdir("./S2648/")
    # for i in range(len(list_)):
    for i in range(1800,len(list_)):
        ilist = list_[i]
        folder_dir = f'{ilist[0]}_{ilist[2]}_{ilist[3]}_{ilist[4]}_{ilist[5]}'
        dst = f'{folder1}/{folder_dir}'
        filename = f'{ilist[0]}_{ilist[2]}'
        if not os.path.exists(f'{dst}/{filename}_WT.pssm') or not os.path.exists(f'{dst}/{filename}_MT.pssm'):
            os.system(f'sbatch {folder2}/{folder_dir}.job')
    os.chdir("../")
    return 

def seq_job(list_, folder1, folder2):
    numCPU = 10
    os.chdir("./S2648/")
    for ilist in list_:
        folder_dir = f'{ilist[0]}_{ilist[2]}_{ilist[3]}_{ilist[4]}_{ilist[5]}'
        dst = f'{folder1}/{folder_dir}'
        fp = open(f'{folder2}/{folder_dir}.job', 'w')
        fp.write('#!/bin/bash\n')
        fp.write('#SBATCH --time=03:00:00\n')
        fp.write('#SBATCH --nodes=1\n')
        fp.write('#SBATCH -A guowei-search\n')
        fp.write('#SBATCH --ntasks-per-node=1\n')
        fp.write('#SBATCH --cpus-per-task=1\n')
        fp.write('#SBATCH --constraint=\"amd20\"\n')
        fp.write('#SBATCH --mem=%dG\n'%(numCPU*10))
        fp.write(f'#SBATCH --job-name {folder_dir}\n')
        fp.write(f'#SBATCH --output=./seq_out/{folder_dir}.out\n')
        fp.write('module purge\n') ###yiming
        fp.write('source ~/.bashrc\n')            # Load conda setup
        fp.write('conda activate esm_env\n')      # Activate environment
        fp.write(f'cd /mnt/home/renyimi2/SheafLapNet/S2648/{dst}\n')
        # fp.write(f'cd /mnt/research/guowei-search.4/junjie-proteng/S2648/{dst}\n')
        #fp.write('source activate pytorch\n')
        fp.write('python feature_seq.py '+' '.join(ilist)+'\n')
        fp.close()
    os.chdir("../")
    return

def run_seq(list_, folder1, folder2):
    os.chdir("./S2648/")
    # for i in range(len(list_)):
    for i in range(1800,len(list_)):
        ilist = list_[i]
        folder_dir = f'{ilist[0]}_{ilist[2]}_{ilist[3]}_{ilist[4]}_{ilist[5]}'
        dst = f'{folder1}/{folder_dir}'
        filename = f'{ilist[0]}_{ilist[2]}'

        opp_folder_dir = f'{ilist[0]}_{ilist[2]}_{ilist[5]}_{ilist[4]}_{ilist[3]}'
        if not os.path.exists(f'{dst}/{folder_dir}_seq.npy') or not os.path.exists(f'{dst}/{opp_folder_dir}_seq.npy'):
            os.system(f'sbatch {folder2}/{folder_dir}.job')
    os.chdir("../")
    return 

def gen_PDBs(PDBid, Chains, Chain, resWT, resID, resMT, pH):
    print(PDBid, Chains, Chain, resWT, resID, resMT)
    if not os.path.exists("features/{}_{}_{}_{}_{}".format(PDBid, Chain, resWT, resID, resMT)):
        os.system("mkdir features/{}_{}_{}_{}_{}".format(PDBid, Chain, resWT, resID, resMT))
    
    curr_dir = "features/{}_{}_{}_{}_{}/".format(PDBid, Chain, resWT, resID, resMT)
    os.chdir("features/{}_{}_{}_{}_{}".format(PDBid, Chain, resWT, resID, resMT))
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/bin/jackal.dir jackal.dir")
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/bin/profix profix")
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/bin/scap scap")
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/code/structure.py structure.py")
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/code/protein.py protein.py")
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/code/feature.py feature.py")
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/code/feature_dssp.py feature_dssp.py")
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/code/feature_Lap.py feature_Lap.py")
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/code/feature_seq.py feature_seq.py")
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/code/prepare.py prepare.py")
    os.system("ln -s  /mnt/home/renyimi2/SheafLapNet/code/src src")

    #if not os.path.exists(PDBid+'.pdb'):
        #os.system('wget https://files.rcsb.org/download/'+PDBid+'.pdb')

    #print(PDBid, Chains, Chain, resWT, resID, resMT)
    s = get_structure(PDBid, Chains, Chain, resWT, resID, resMT, pH=pH)
    s.generateMutedPDBs()
    s.generateMutedPQRs()
    s.readFASTA()
    s.writeFASTA()
    #filename = PDBid+'_'+Chain+'_'+resWT+'_'+resID+'_'+resMT
    #if not os.path.exists(PDBid+'_'+Chain+'.englist'):
    #print(PDBid, Antibody, Antigen, Chain, resWT, resID, resMT)
    #os.system("python PPIprepare.py {} {} {} {} {} {} {} 7.0".format(PDBid, Antibody, Antigen, Chain, resWT, resID, resMT))
    #os.system("python PPIfeature_Lap.py {} {} {} {} {} {} {} 7.0".format(PDBid, Antibody, Antigen, Chain, resWT, resID, resMT))
    os.chdir("../../")

def mp_gen_PDBs(seq_list):

    os.chdir("./S2648/")
    no_threads = mp.cpu_count()
    p = mp.Pool(processes = no_threads)
    results = p.starmap(gen_PDBs, seq_list)
    p.close()
    p.join()
    os.chdir("../")

def check_pssm(list_, folder1, folder2):
    os.chdir("./S2648/")
    # for i in range(len(list_)):
    for i in range(900):
        ilist = list_[i]
        folder_dir = f'{ilist[0]}_{ilist[2]}_{ilist[3]}_{ilist[4]}_{ilist[5]}'
        dst = f'{folder1}/{folder_dir}'
        filename = f'{ilist[0]}_{ilist[2]}'
        #if not os.path.exists(f'{dst}/{filename}_WT.pssm') or not os.path.exists(f'{dst}/{filename}_MT.pssm'):
        if not os.path.exists(f'{dst}/{filename}_WT.pssm'):
            print(f'{dst}/{filename}_WT.pssm')
            #os.system(f'cp {folder1}/HK3O_A_A_344_C/{filename}_WT.pssm {dst}/{filename}_WT.pssm')
        #file1 = open(f'{dst}/{filename}_WT.pssm')
        #c1 = file1.readlines()

        if not os.path.exists(f'{dst}/{filename}_MT.pssm'):
            print(f'{dst}/{filename}_MT.pssm')
            # os.system(f'sbatch {folder2}/{folder_dir}.job')

        #file2 = open(f'{dst}/{filename}_MT.pssm')
        #c2 = file2.readlines()
        #if c1[-1][:3]!="PSI" or c2[-1][:3]!="PSI":
            #print(folder_dir)
            #os.system(f'sbatch {folder2}/{folder_dir}.job')
    os.chdir("../")
    return 

def check_MIBPB(list_, folder1):
    os.chdir("./S2648/")
    for i in range(len(list_)):
        ilist = list_[i]
        folder_dir = f'{ilist[0]}_{ilist[2]}_{ilist[3]}_{ilist[4]}_{ilist[5]}'
        dst = f'{folder1}/{folder_dir}'
        filename = f'{ilist[0]}_{ilist[2]}'
        #if not os.path.exists(f'{dst}/{filename}_WT.pssm') or not os.path.exists(f'{dst}/{filename}_MT.pssm'):
        if not os.path.exists(f'{dst}/{filename}_WT.eng') or not os.path.exists(f'{dst}/{filename}_WT.englist') \
            or not os.path.exists(f'{dst}/{filename}_WT.arealist') or not os.path.exists(f'{dst}/{filename}_WT.areavolume'):
            os.system(f'cp {folder1}/HK3O_A_A_344_C/{filename}_WT.eng {dst}/{filename}_WT.eng')
            os.system(f'cp {folder1}/HK3O_A_A_344_C/{filename}_WT.englist {dst}/{filename}_WT.englist')
            os.system(f'cp {folder1}/HK3O_A_A_344_C/{filename}_WT.arealist {dst}/{filename}_WT.arealist')
            os.system(f'cp {folder1}/HK3O_A_A_344_C/{filename}_WT.areavolume {dst}/{filename}_WT.areavolume')
        #file1 = open(f'{dst}/{filename}_WT.pssm')
        #c1 = file1.readlines()

    os.chdir("../")
    return 

def dataset_list(filename):
    dataset = []
    fp = open(filename)
    for line in fp:
        line_split = re.split(',|\n', line)
        dataset.append([line_split[i] for i in [0, 1, 2, 3, 4, 5, 6]])
    fp.close()
    return dataset

def run_dssp(PDBid, Chains, Chain, resWT, resID, resMT, pH):
    print(PDBid, Chains, Chain, resWT, resID, resMT)
    if not os.path.exists("features/{}_{}_{}_{}_{}".format(PDBid, Chain, resWT, resID, resMT)):
        os.system("mkdir features/{}_{}_{}_{}_{}".format(PDBid, Chain, resWT, resID, resMT))
    
    curr_dir = "features/{}_{}_{}_{}_{}/".format(PDBid, Chain, resWT, resID, resMT)
    os.chdir("features/{}_{}_{}_{}_{}".format(PDBid, Chain, resWT, resID, resMT))
    os.system("ln -s ../../../code/TopLapFit/code/feature_dssp.py feature_dssp.py")

    os.system("python feature_dssp.py {} {} {} {} {} {} {} {}".format(PDBid, Chains, Chain, resWT, resID, resMT, pH))
    os.chdir("../../")

if not os.path.exists("./S2648/features/"):
    os.system("mkdir ./S2648/features/")

seq_list = dataset_list("./S2648/S2648.txt")
#print(seq_list)
# mp_gen_PDBs(seq_list)
#os.chdir("./S2648/")
#PDBid, Chains, Chain, resWT, resID, resMT, pH = seq_list[0][0], seq_list[0][1], seq_list[0][2], seq_list[0][3], seq_list[0][4], seq_list[0][5], seq_list[0][6]
#gen_PDBs(PDBid, Chains, Chain, resWT, resID, resMT, pH)
#os.chdir("../")

if not os.path.exists("./S2648/blast_jobs/"):
    os.mkdir("./S2648/blast_jobs/")
if not os.path.exists("./S2648/blast_out/"):
    os.mkdir("./S2648/blast_out/")

#create_blastjob(seq_list, "features", "blast_jobs")
#blast_jobs(seq_list, "features", "blast_jobs")
#check_pssm(seq_list, "features", "blast_jobs")

#os.chdir("./S2648/")
#no_threads = mp.cpu_count()
#p = mp.Pool(processes = no_threads)
#results = p.starmap(run_dssp, seq_list)
#p.close()
#p.join()
#for i in range(len(seq_list)):
    #ilist = seq_list[i]
    #PDBid, Antibody, Antigen, Chain, resWT, resID, resMT, pH = ilist[0], ilist[1], ilist[2], ilist[3], ilist[4], ilist[5], ilist[6], ilist[7]
    #run_dssp(PDBid, Antibody, Antigen, Chain, resWT, resID, resMT, pH)
#os.chdir("../")

if not os.path.exists("./S2648/feat_jobs/"):
    os.mkdir("./S2648/feat_jobs/")

if not os.path.exists("./S2648/feat_out/"):
    os.mkdir("./S2648/feat_out/")

#check_MIBPB(seq_list, "features")
#feat_job(seq_list, "features", "feat_jobs")
run_feat(seq_list, "features", "feat_jobs")

if not os.path.exists("./S2648/seq_jobs/"):
    os.mkdir("./S2648/seq_jobs/")

if not os.path.exists("./S2648/seq_out/"):
    os.mkdir("./S2648/seq_out/")

#seq_job(seq_list, "features", "seq_jobs")
#run_seq(seq_list, "features", "seq_jobs")


#print(seq_list)
os.chdir("./S2648/")
# for i in range(len(seq_list)):
#     ilist = seq_list[i]
#     print(ilist)
#     PDBid, Chains, Chain, resWT, resID, resMT, pH = ilist[0], ilist[1], ilist[2], ilist[3], ilist[4], ilist[5], ilist[6]
#     link_SR(PDBid, Chains, Chain, resWT, resID, resMT, pH)
os.chdir("../")
