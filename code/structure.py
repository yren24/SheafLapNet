import os, re, sys, warnings, time
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Residue import Residue
warnings.filterwarnings('ignore')

AminoA = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS',
          'SEF', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR',
          'TRP']
# Non-canonical or non natural
NC_AminoA = {'LLP': 'LYS', 'M3P': 'LYS', 'MSE': 'MET', 'F2F': 'PHE', 'CGU': 'GLU',
        'MYL': 'LYS', 'TPO': 'THR', 'HSE': 'HIS'}
default_cutoff = 16.
ElementList = ['C', 'N', 'O']

def use_profix_scap(filename, muteChain, resID, resMT):
    # the profix eliminates the insertion code!!!
    # adjust atoms position by profix
    #if not os.path.exists('profix'):
    #    os.system('cp ../../bin/profix .')
    #    os.system('cp ../../bin/jackal.dir .')
    os.system('profix -fix 0 '+filename+'_WT.pdb')
    os.system('cp '+filename+'_WT_fix.pdb '+filename+'_WT.pdb')
 
    # generate mutant PDB_file
    #os.system('cp ../../bin/scap .')
    scap_file = open('tmp_scap.list', 'w')
    scap_file.write(','.join([muteChain, str(resID), resMT]))
    scap_file.close()
    os.system('scap -ini 20 -min 4 '+filename+'_WT.pdb tmp_scap.list')
    os.system('cp '+filename+'_WT_scap.pdb '+filename+'_MT.pdb')
    #os.system('rm -f tmp_scap.list')
    return

def removeChains_insertionCode(structure, Chains, resID, muteChain):
    '''
        This function exclude chains that are not in 'Chains'
        structure: the structure from PDBs.pdb
        Chains: target chains are included
        resID: mutation residue ID
        muteChain: mutation chain
    '''
    # the write-down PDBs are for MIBPB calculation
    # the SolvEng has a (1.421162, 0.704903) might be important with more juice
    # however, insertion code should be removed since profix will remove it anyway
    # the new residue ID for mutation site should be updated
    structure_clean = structure.copy()
    ichain_ids_to_remove = []
    for ichain in structure_clean[0]:
        if ichain.id not in Chains:
            ichain_ids_to_remove.append(ichain.id)
        else: # remove the insertion code
    # when change residue ID with insertion code
    # ValueError: Cannot change id from `(' ', 83, ' ')` to `(' ', 86, ' ')`. 
    # The id `(' ', 86, ' ')` is already used for a sibling of this entity.
            for idx, iresidue in enumerate(structure_clean[0][ichain.id]):
                #print(iresidue.id, resID, ichain.id, muteChain)
                if iresidue.id == resID and ichain.id == muteChain:
                    resID_structure = idx+1
                iresidue_id = list(iresidue.id)
                iresidue_id[0] = 'Old'
                iresidue.id = tuple(iresidue_id)
            for idx, iresidue in enumerate(structure_clean[0][ichain.id]):
                iresidue_id = list(iresidue.id)
                iresidue_id[0], iresidue_id[1], iresidue_id[2] = ' ', idx+1, ' '
                iresidue.id = tuple(iresidue_id)
                #print(ichain.id, iresidue.id)
    for ichain_id in ichain_ids_to_remove:
        structure_clean[0].detach_child(ichain_id)
    return structure_clean, resID_structure

class get_structure:
    def __init__(self, PDBid, Chains, muteChain, resWT, resID, resMT, pH='7.0', cutoff=default_cutoff, onlyBLAST=False):
        self.PDBid = PDBid

        self.pH = pH
        self.cutoff = default_cutoff

        self.muteChain = muteChain
        self.resWT = resWT
        self.resMT = resMT
        self.fasta = {}

        self.Chains = re.split('', Chains)
        self.Chains = [i for i in self.Chains if i]

        # other chains
        self.OtherChains = self.Chains.copy()
        self.OtherChains.pop(self.Chains.index(self.muteChain))

        # filename
        self.fileComplex  = PDBid+'_'+Chains

        # deal with insertion code of PDB
        resIDchr = re.search('[a-zA-Z]', resID)
        if resIDchr!=None:
            resIDchr = resIDchr.group(0).upper()
            resIDidx = int(resID[:-1])
        else:
            resIDchr = ' '
            resIDidx = int(resID)
        self.resID_ori = (' ', resIDidx, resIDchr)

        # get PDB_File and fasta_File
        if not os.path.exists(self.PDBid+'.pdb'):
            os.system('wget https://files.rcsb.org/download/'+self.PDBid+'.pdb')

        if not onlyBLAST:
            # change self.resID_ori (self.resID) to number residue
            # use Biopython to load PDB_file for the starting residue ID of target chain
            parser = PDBParser(PERMISSIVE=1)
            s = parser.get_structure(self.PDBid, self.PDBid+'.pdb')
            # three things are done in the following
            # 1. remove ions and waters; 2. replace non-canonical residues to their parental residues
            for ichain in self.Chains:
                iresidue_ids_to_remove = []
                for iresidue in s[0][ichain]:
                    if iresidue.resname in NC_AminoA:
                        iresidue_id = list(iresidue.id)
                        iresidue_id[0] = ' '
                        iresidue.id = tuple(iresidue_id)
                        iresidue.resname = NC_AminoA[iresidue.resname]
                    elif iresidue.resname not in NC_AminoA and iresidue.resname not in AminoA:
                        iresidue_ids_to_remove.append(iresidue.id)
                for iresidue_id in iresidue_ids_to_remove:
                    s[0][ichain].detach_child(iresidue_id)
            # 3. remove other chains
            ichain_ids_to_remove = []
            for ichain in s[0]:
                if ichain.id not in self.Chains:
                    ichain_ids_to_remove.append(ichain.id)
            for ichain_id in ichain_ids_to_remove:
                s[0].detach_child(ichain_id)
            # save files and profix
            s, resID_temp = removeChains_insertionCode(s, self.Chains, self.resID_ori, self.muteChain)
            if not os.path.exists(f'{self.fileComplex}.pdb'):
                io_Complex = PDBIO()
                io_Complex.set_structure(s)
                io_Complex.save(self.fileComplex+'.pdb')
                os.system(f'profix -fix 0 {self.fileComplex}.pdb')
                os.system(f'mv {self.fileComplex}_fix.pdb {self.fileComplex}.pdb')
            parser = PDBParser(PERMISSIVE=1)
            self.s = parser.get_structure(self.PDBid, f'{self.fileComplex}.pdb')
            self.resID = (' ', resID_temp, ' ')

            # distance_mutation_binding
            for iresidue in self.s[0][muteChain]:
                if iresidue.id == self.resID: 
                    if not three_to_one(iresidue.resname) == self.resWT:
                        print(iresidue.id, self.resID); sys.exit('After first profix, mutant residue not match')
                    self.muteResidue = iresidue.copy()
            self.distance_muteResidue_bindingSite = 100
            for ichain in self.Chains:
                for iresidue in self.s[0][ichain]:
                    ##print(ichain, iresidue.id, iresidue.resname)
                    #dist = iresidue['CA'] - self.muteResidue['CA']
                    #if dist < self.distance_muteResidue_bindingSite:
                    #    self.distance_muteResidue_bindingSite = dist
                    for iatom in iresidue:
                        dist = iatom - self.muteResidue['CA']
                        if dist < self.distance_muteResidue_bindingSite:
                            self.distance_muteResidue_bindingSite = dist
            # compare with the cutoff
            #if self.distance_muteResidue_bindingSite > self.cutoff:
            #    sys.exit('the distance of mute residue to binding site is longer than cutoff')

    def generateMutedPDBs(self, flag_use_binary = True):
        self.s_MutedPartner_WT, self.resID_MutedPartner = \
                removeChains_insertionCode(self.s, self.Chains, self.resID, self.muteChain)

        if flag_use_binary and (not os.path.exists(self.PDBid+'_WT.pdb') \
                            or  not os.path.exists(self.PDBid+'_MT.pdb')):
            io_MutedPartner_WT = PDBIO()
            io_MutedPartner_WT.set_structure(self.s_MutedPartner_WT)
            io_MutedPartner_WT.save(self.PDBid+'_WT.pdb')
            use_profix_scap(self.PDBid, self.muteChain, self.resID_MutedPartner, self.resMT)
        print('generate files:', self.PDBid+'_WT.pdb', self.PDBid+'_MT.pdb')

        parser = PDBParser(PERMISSIVE=1)
        self.s_MutedPartner_MT = parser.get_structure(self.PDBid, self.PDBid+'_MT.pdb')
        return # generateMutedPartnerPDBs

    def generateMutedPQRs(self):
        # generated PQR_file
        if not os.path.exists(self.PDBid+'_WT.pqr'):
            os.system('pdb2pqr --ff=amber --ph-calc-method=propka --chain --with-ph='+self.pH+
                    ' '+self.PDBid+'_WT.pdb '+self.PDBid+'_WT.pqr')
        if not os.path.exists(self.PDBid+'_MT.pqr'):
            os.system('pdb2pqr --ff=charmm --ph-calc-method=propka --chain --with-ph='+self.pH+
                    ' '+self.PDBid+'_MT.pdb '+self.PDBid+'_MT.pqr')
        return # generateMutedPartnerPQRs

    def readFASTA(self):
        # local structure only used for fasta. use original pdb, not the fixed one
        parser = PDBParser(PERMISSIVE=1)
        s = parser.get_structure(self.PDBid, self.PDBid+'.pdb')
        #s = parser.get_structure(self.PDBid, self.PDBid+'.pdb1') # To read structure from original pdb and not the one after alphafold or after scap
        # initialize variables
        self.non_canonical = []

        # filename
        self.fileMuteChain = self.PDBid+'_'+self.muteChain
        #fp = open(self.PDBid+'.pdb')
        # check missing residue and record it to AB_MISS_RES and AG_MISS_RES
        fp = open(self.fileMuteChain+'.pdb') # Modify to suit the bug
        # this MISS_RES only record the target chain 'self.muteChain'
        array_RES = {}
        flagMISSRES = False; marker = ''
        for line in fp:
            words = re.split(' |\n', line)
            if len(words) > 4:
                if words[2] == 'MISSING' and words[3] == 'RESIDUES':
                    marker = words[1]
                    flagMISSRES = True
                    break
        if flagMISSRES:
            for _ in range(5): # skip 5 lines
                fp.readline()
            line = re.split(' |\n', fp.readline())
            line = [i for i in line if i]
            while line[1] == marker:
                if line[3] in self.muteChain:
                    if line[2] in AminoA: 
                        if not line[4][-1].isalpha():
                            array_RES[int(line[4])] = three_to_one(line[2])
                    #elif line[2] != 'HOH':
                    #    self.non_canonical.append(line[2])
                line = re.split(' |\n', fp.readline())
                line = [i for i in line if i]
        fp.close()

        # if array_RES is empty, then jump is not allowed or
        # the largest index of MISSING RESIDUE is less than starter residue
        residue = next(iter(s[0][self.muteChain])) # what's this? # gives the first residue
        start_idx_PDB  = residue.id[1]
        if len(array_RES) == 0:
            flagMISSRES = False
        else:
            array_RES_idx = list(array_RES.keys()) # find the last index
            last_idx_MISS = array_RES_idx[-1]
            if last_idx_MISS < start_idx_PDB:
                flagMISSRES = False
        # if flagMISSRES = False, then jump index is not allowed

        shift = 0;# last_idx 
        resID_mute_in_ = 0
        for idx, iresidue in enumerate(s[0][self.muteChain]):
            if iresidue.resname in AminoA:
                if iresidue.id[2] != ' ': # Here is the insertion code appearing
                    shift += 1
                if flagMISSRES:
                    if iresidue.id == self.resID_ori:
                        resID_mute_in_ = iresidue.id[1]+shift
                        if self.resWT != three_to_one(iresidue.resname):
                            sys.exit('Check the PDB residues. Wrong residue name for input!!! 1')
                    array_RES[iresidue.id[1]+shift] = three_to_one(iresidue.resname)
                else:
                    if iresidue.id == self.resID_ori:
                        # this might be OK for no MISSING RESIDUE, what about the other?
                        # should be OK, MISSING RESIDUE is in the front
                        resID_mute_in_ = start_idx_PDB+idx
                        if self.resWT != three_to_one(iresidue.resname):
                            sys.exit('Check the PDB residues. Wrong residue name for input!!! 2')
                    array_RES[start_idx_PDB+idx] = three_to_one(iresidue.resname)
            elif iresidue.resname in NC_AminoA:
                iresidue_resname = NC_AminoA[iresidue.resname]
                if iresidue.id[2] != ' ': # Here is the insertion code appearing
                    shift += 1
                if flagMISSRES:
                    if iresidue.id == self.resID_ori:
                        resID_mute_in_ = iresidue.id[1]+shift
                        if self.resWT != three_to_one(iresidue_resname):
                            sys.exit('Check the PDB residues. Wrong residue name for input!!! 3')
                    array_RES[iresidue.id[1]+shift] = three_to_one(iresidue_resname)
                else:
                    if iresidue.id == self.resID_ori:
                        # this might be OK for no MISSING RESIDUE, what about the other?
                        # should be OK, MISSING RESIDUE is in the front
                        resID_mute_in_ = start_idx_PDB+idx
                        if self.resWT != three_to_one(iresidue_resname):
                            sys.exit('Check the PDB residues. Wrong residue name for input!!! 4')
                    array_RES[start_idx_PDB+idx] = three_to_one(iresidue_resname)
            #elif iresidue.resname != 'HOH' and start_idx_PDB+idx == iresidue.id[1]:
            #    self.non_canonical.append(iresidue.resname)
            #if iresidue.resname != 'HOH':
            #    print(start_idx_PDB+idx, iresidue.id, iresidue.resname)

        # sort array_RES by index
        array_RES_sorted = {}
        for idx in sorted(array_RES):
            array_RES_sorted[idx] = array_RES[idx]
        #print(array_RES_sorted.keys())

        # get the start and end index, real mutation ID
        array_RES_idx = list(array_RES_sorted.keys())
        idx_s = array_RES_idx[0]; idx_e = array_RES_idx[-1]
        # 0 in array_RES_idx, when flagMISSRES == True
        if 0 not in array_RES_idx and idx_s < 0:
            self.resID_fasta = resID_mute_in_ - idx_s - 1
            seqlength = idx_e-idx_s
        else:
            self.resID_fasta = resID_mute_in_ - idx_s
            seqlength = idx_e-idx_s + 1
        #print(self.resID_fasta, resID_mute_in_, idx_s, seqlength)

        flag_fastaWT_fastaMT = False
        #print(len(array_RES_idx))
        if seqlength == len(array_RES_idx):
            #print('This is a consecutive FASTA sequence!!!')
            self.fastaWT = ''.join(list(array_RES_sorted.values()))
            array_RES_sorted[resID_mute_in_] = self.resMT
            self.fastaMT = ''.join(list(array_RES_sorted.values()))
            flag_fastaWT_fastaMT = True

        #print('Cannot have a consecutive FASTA sequence from this PDB file')
        #print(idx_s, idx_e, idx_e-idx_s+1, len(array_RES_idx))
        #print('Use SEQRES info from PDB instead')

        if not flag_fastaWT_fastaMT:
            print('WARNING: flag_fastaWT_fastaMT true!')
            # use SEQRES in PDB for fasta file
            array_RES = {}; idx = idx_s
            fp = open(self.PDBid+'.pdb')
            for line in fp:
                if line[:6] == 'SEQRES':
                    words = [i for i in re.split(' |\n', line) if i]
                    if words[2] == self.muteChain:
                        for resname in words[4:]:
                            if resname in AminoA:
                                array_RES[idx] = three_to_one(resname)
                            #elif resname != 'HOH':
                            #    self.non_canonical.append(resname)
                            idx += 1 # !!! 
            fp.close();
            if array_RES[resID_mute_in_] != self.resWT:
                # use the position info from BioPython to check whether it is the one of array_RES
                array_RES_index = list(array_RES_sorted.keys())
                resID_mute_in_pos = array_RES_index.index(resID_mute_in_)
                self.resID_fasta = resID_mute_in_pos # this is dangerous
                array_RES_residue = list(array_RES.values())
                if array_RES_residue[resID_mute_in_pos] == self.resWT:
                    self.fastaWT = ''.join(list(array_RES_residue))
                    array_RES_residue[resID_mute_in_pos] = self.resMT
                    self.fastaMT = ''.join(list(array_RES_residue))
                    flag_fastaWT_fastaMT = True
                else:
                    sys.exit('Need manually check or fasta file')
            else:
                self.fastaWT = ''.join(list(array_RES.values()))
                array_RES[resID_mute_in_] = self.resMT
                self.fastaMT = ''.join(list(array_RES.values()))
                flag_fastaWT_fastaMT = True

        if not flag_fastaWT_fastaMT:
            return False

        # gather all the fasta for other chains and other partner
        self.fasta[self.muteChain] = self.fastaWT
        for ChainID in self.OtherChains:
            self.fasta[ChainID] = self.readOtherFASTA(ChainID, s)
            (flag_fasta, fasta_) = self.compareSeqFasta(ChainID)
            if not flag_fasta:
                self.fasta[ChainID] = fasta_
                #self.compareSeqFasta(ChainID)
            else:
                if len(fasta_) != len(self.fasta[ChainID]):
                    self.fasta[ChainID] = fasta_
        self.fasta['WT'] = self.fastaWT
        self.fasta['MT'] = self.fastaMT
        return True # readFASTA()
    

    # def readFASTA(self):
    #     # local structure only used for fasta. use original pdb, not the fixed one
    #     parser = PDBParser(PERMISSIVE=1)
    #     s = parser.get_structure(self.PDBid, self.PDBid+'.pdb')
    #     #s = parser.get_structure(self.PDBid, self.PDBid+'.pdb1') # To read structure from original pdb and not the one after alphafold or after scap
    #     # initialize variables
    #     self.non_canonical = []

    #     # filename
    #     self.fileMuteChain = self.PDBid+'_'+self.muteChain

    #     # check missing residue and record it to AB_MISS_RES and AG_MISS_RES
    #     #fp = open(self.fileMuteChain+'.pdb') # Modify to suit the bug
    #     fp = open(self.PDBid+'.pdb')
    #     # this MISS_RES only record the target chain 'self.muteChain'
    #     array_RES = {}
    #     flagMISSRES = False; marker = ''
    #     for line in fp:
    #         words = re.split(' |\n', line)
    #         if len(words) > 4:
    #             if words[2] == 'MISSING' and words[3] == 'RESIDUES':
    #                 marker = words[1]
    #                 flagMISSRES = True
    #                 break
    #     if flagMISSRES:
    #         for _ in range(5): # skip 5 lines
    #             fp.readline()
    #         line = re.split(' |\n', fp.readline())
    #         line = [i for i in line if i]
    #         while line[1] == marker:
    #             if line[3] in self.muteChain:
    #                 if line[2] in AminoA: 
    #                     if not line[4][-1].isalpha():
    #                         array_RES[int(line[4])] = three_to_one(line[2])
    #                 #elif line[2] != 'HOH':
    #                 #    self.non_canonical.append(line[2])
    #             line = re.split(' |\n', fp.readline())
    #             line = [i for i in line if i]
    #     fp.close()

    #     # if array_RES is empty, then jump is not allowed or
    #     # the largest index of MISSING RESIDUE is less than starter residue
    #     residue = next(iter(s[0][self.muteChain])) # what's this? # gives the first residue
    #     start_idx_PDB  = residue.id[1]
    #     if len(array_RES) == 0:
    #         flagMISSRES = False
    #     else:
    #         array_RES_idx = list(array_RES.keys()) # find the last index
    #         last_idx_MISS = array_RES_idx[-1]
    #         if last_idx_MISS < start_idx_PDB:
    #             flagMISSRES = False
    #     # if flagMISSRES = False, then jump index is not allowed

    #     shift = 0;# last_idx 
    #     resID_mute_in_ = 0
    #     for idx, iresidue in enumerate(s[0][self.muteChain]):
    #         if iresidue.resname in AminoA:
    #             if iresidue.id[2] != ' ': # Here is the insertion code appearing
    #                 shift += 1
    #             if flagMISSRES:
    #                 if iresidue.id == self.resID_ori:
    #                     resID_mute_in_ = iresidue.id[1]+shift
    #                     if self.resWT != three_to_one(iresidue.resname):
    #                         sys.exit('Check the PDB residues. Wrong residue name for input!!! 1')
    #                 array_RES[iresidue.id[1]+shift] = three_to_one(iresidue.resname)
    #             else:
    #                 if iresidue.id == self.resID_ori:
    #                     # this might be OK for no MISSING RESIDUE, what about the other?
    #                     # should be OK, MISSING RESIDUE is in the front
    #                     resID_mute_in_ = start_idx_PDB+idx
    #                     if self.resWT != three_to_one(iresidue.resname):
    #                         sys.exit('Check the PDB residues. Wrong residue name for input!!! 2')
    #                 array_RES[start_idx_PDB+idx] = three_to_one(iresidue.resname)
    #         elif iresidue.resname in NC_AminoA:
    #             iresidue_resname = NC_AminoA[iresidue.resname]
    #             if iresidue.id[2] != ' ': # Here is the insertion code appearing
    #                 shift += 1
    #             if flagMISSRES:
    #                 if iresidue.id == self.resID_ori:
    #                     resID_mute_in_ = iresidue.id[1]+shift
    #                     if self.resWT != three_to_one(iresidue_resname):
    #                         sys.exit('Check the PDB residues. Wrong residue name for input!!! 3')
    #                 array_RES[iresidue.id[1]+shift] = three_to_one(iresidue_resname)
    #             else:
    #                 if iresidue.id == self.resID_ori:
    #                     # this might be OK for no MISSING RESIDUE, what about the other?
    #                     # should be OK, MISSING RESIDUE is in the front
    #                     resID_mute_in_ = start_idx_PDB+idx
    #                     if self.resWT != three_to_one(iresidue_resname):
    #                         sys.exit('Check the PDB residues. Wrong residue name for input!!! 4')
    #                 array_RES[start_idx_PDB+idx] = three_to_one(iresidue_resname)
    #         #elif iresidue.resname != 'HOH' and start_idx_PDB+idx == iresidue.id[1]:
    #         #    self.non_canonical.append(iresidue.resname)
    #         #if iresidue.resname != 'HOH':
    #         #    print(start_idx_PDB+idx, iresidue.id, iresidue.resname)

    #     # # sort array_RES by index
    #     # array_RES_sorted = {}
    #     # for idx in sorted(array_RES):
    #     #     array_RES_sorted[idx] = array_RES[idx]
    #     # #print(array_RES_sorted.keys())

    #     # # get the start and end index, real mutation ID
    #     # array_RES_idx = list(array_RES_sorted.keys())
    #     # idx_s = array_RES_idx[0]; idx_e = array_RES_idx[-1]
    #     # # 0 in array_RES_idx, when flagMISSRES == True
    #     # if 0 not in array_RES_idx and idx_s < 0:
    #     #     self.resID_fasta = resID_mute_in_ - idx_s - 1
    #     #     seqlength = idx_e-idx_s
    #     # else:
    #     #     self.resID_fasta = resID_mute_in_ - idx_s
    #     #     seqlength = idx_e-idx_s + 1
    #     # #print(self.resID_fasta, resID_mute_in_, idx_s, seqlength)

    #     # flag_fastaWT_fastaMT = False
    #     # #print(len(array_RES_idx))
    #     # if seqlength == len(array_RES_idx):
    #     #     #print('This is a consecutive FASTA sequence!!!')
    #     #     self.fastaWT = ''.join(list(array_RES_sorted.values()))
    #     #     array_RES_sorted[resID_mute_in_] = self.resMT
    #     #     self.fastaMT = ''.join(list(array_RES_sorted.values()))
    #     #     flag_fastaWT_fastaMT = True

    #     # #print('Cannot have a consecutive FASTA sequence from this PDB file')
    #     # #print(idx_s, idx_e, idx_e-idx_s+1, len(array_RES_idx))
    #     # #print('Use SEQRES info from PDB instead')

    #     # if not flag_fastaWT_fastaMT:
    #     #     print('WARNING: flag_fastaWT_fastaMT true!')
    #     #     # use SEQRES in PDB for fasta file
    #     #     array_RES = {}; idx = idx_s
    #     #     fp = open(self.PDBid+'.pdb')
    #     #     for line in fp:
    #     #         if line[:6] == 'SEQRES':
    #     #             words = [i for i in re.split(' |\n', line) if i]
    #     #             if words[2] == self.muteChain:
    #     #                 for resname in words[4:]:
    #     #                     if resname in AminoA:
    #     #                         array_RES[idx] = three_to_one(resname)
    #     #                     #elif resname != 'HOH':
    #     #                     #    self.non_canonical.append(resname)
    #     #                     idx += 1 # !!! 
    #     #     fp.close();
    #     #     # print(array_RES)
    #     #     if array_RES[resID_mute_in_] != self.resWT:
    #     #         # use the position info from BioPython to check whether it is the one of array_RES
    #     #         array_RES_index = list(array_RES_sorted.keys())
    #     #         resID_mute_in_pos = array_RES_index.index(resID_mute_in_)
    #     #         self.resID_fasta = resID_mute_in_pos # this is dangerous
    #     #         array_RES_residue = list(array_RES.values())
    #     #         if array_RES_residue[resID_mute_in_pos] == self.resWT:
    #     #             self.fastaWT = ''.join(list(array_RES_residue))
    #     #             array_RES_residue[resID_mute_in_pos] = self.resMT
    #     #             self.fastaMT = ''.join(list(array_RES_residue))
    #     #             flag_fastaWT_fastaMT = True
    #     #         else:
    #     #             sys.exit('Need manually check or fasta file')
    #     #     else:
    #     #         self.fastaWT = ''.join(list(array_RES.values()))
    #     #         array_RES[resID_mute_in_] = self.resMT
    #     #         self.fastaMT = ''.join(list(array_RES.values()))
    #     #         flag_fastaWT_fastaMT = True

    #     # if not flag_fastaWT_fastaMT:
    #     #     return False
    #     # print('self.resID_fasta', self.resID_fasta)
    #     # print('self.fastaWT', self.fastaWT)

    #     # --- ORDER-ONLY FASTA FROM ATOM (no contiguity test, no SEQRES fallback) ---

    #     # 1) Sort ATOM keys and build the ordered 1-letter sequence
    #     array_RES_sorted = {k: array_RES[k] for k in sorted(array_RES)}
    #     klist = sorted(array_RES_sorted.keys())                # e.g., [23,24,25, ..., 75,77,...]
    #     vlist = [array_RES_sorted[k] for k in klist]           # 1-letter codes in that order

    #     # 2) Find mutation position by PDB number in the ATOM order
    #     try:
    #         pos = klist.index(resID_mute_in_)                  # 0-based FASTA index
    #     except ValueError:
    #         print(f"[DIAG] Mutation PDB index {resID_mute_in_} not found "
    #             f"(first={klist[0]}, last={klist[-1]}, n={len(klist)})")
    #         sys.exit("Mutation residue not in ATOM residues")

    #     # 3) Normalize WT/MT to 1-letter uppercase (in case inputs are 'GLY'/'g'/'gly')
    #     self.resWT = self.resWT[0].upper()
    #     self.resMT = self.resMT[0].upper()

    #     # Optional: warn (don’t hard-fail) if the observed ATOM letter differs from WT
    #     obs_atom = vlist[pos]
    #     if obs_atom != self.resWT:
    #         print(f"[WARN] WT mismatch at {self.muteChain} {resID_mute_in_}: "
    #             f"ATOM={obs_atom} vs input WT={self.resWT}. Using ATOM order.")

    #     # 4) Build WT/MT FASTA strings by position
    #     self.resID_fasta = pos
    #     self.fastaWT = ''.join(vlist)
    #     vlist_mut = vlist.copy()
    #     vlist_mut[pos] = self.resMT
    #     self.fastaMT = ''.join(vlist_mut)
    #     flag_fastaWT_fastaMT = True

    #     # gather all the fasta for other chains and other partner
    #     self.fasta[self.muteChain] = self.fastaWT
    #     for ChainID in self.OtherChains:
    #         self.fasta[ChainID] = self.readOtherFASTA(ChainID, s)
    #         (flag_fasta, fasta_) = self.compareSeqFasta(ChainID)
    #         if not flag_fasta:
    #             self.fasta[ChainID] = fasta_
    #             #self.compareSeqFasta(ChainID)
    #         else:
    #             if len(fasta_) != len(self.fasta[ChainID]):
    #                 self.fasta[ChainID] = fasta_
    #     self.fasta['WT'] = self.fastaWT
    #     self.fasta['MT'] = self.fastaMT
    #     return True # readFASTA()

    def writeFASTA(self):
        seqlength = len(self.fasta['WT'])
        # check if self.fasta = {}, then run readFASTA
        if len(self.fasta) == 0:
            self.readFASTA()
        seqfile_WT = open(self.fileMuteChain+'_WT.fasta', 'w')
        seqfile_MT = open(self.fileMuteChain+'_MT.fasta', 'w')
        for idx in range(seqlength):
            seqfile_WT.write(self.fasta['WT'][idx])
            seqfile_MT.write(self.fasta['MT'][idx])
            if (idx+1)%80 == 0:
                seqfile_WT.write('\n')
                seqfile_MT.write('\n')
        seqfile_WT.close()
        seqfile_MT.close()
        if len(self.fasta['WT']) < 15:
            seqfile_WT = open(self.fileMuteChain+'_WT.fasta', 'w')
            seqfile_MT = open(self.fileMuteChain+'_MT.fasta', 'w')
            for idx in range(seqlength*20):
                seqfile_WT.write(self.fasta['WT'][idx%seqlength])
                seqfile_MT.write(self.fasta['MT'][idx%seqlength])
                if (idx+1)%80 == 0:
                    break
            seqfile_WT.close()
            seqfile_MT.close()
        return # writeFASTA()

    def transformer(self):
        import torch, esm
        # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        data = []
        for ChainID in self.OtherChains:
            ifasta = self.fasta[ChainID]
            length_fasta = len(ifasta)
            if length_fasta > 1022:
                idx = int((length_fasta-1022)/2)
                ifasta = ifasta[idx:idx+1022]
            data.append((ChainID, ifasta))
        fastaWT, fastaMT = self.fasta['WT'], self.fasta['MT']
        if len(fastaWT) > 1022:
            mutation_pos = self.resID_fasta
            if abs(mutation_pos-1022) < 500:
                fastaWT = fastaWT[-1022:]
                fastaMT = fastaMT[-1022:]
            else:
                fastaWT = fastaWT[:1022]
                fastaMT = fastaMT[:1022]
        data.append(('WT', fastaWT))
        data.append(('MT', fastaMT))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results['representations'][33].numpy()

        partner1_WT = []
        partner1_MT = []
        len_seq = len(fastaWT)
        partner1_WT.append(token_representations[-2, 1:len_seq+1].mean(0))
        partner1_WT = np.array(partner1_WT).mean(0)
        len_seq = len(fastaMT)
        partner1_MT.append(token_representations[-1, 1:len_seq+1].mean(0))
        partner1_MT = np.array(partner1_MT).mean(0)

        sequence_representations = []
        sequence_representations.append(partner1_WT)
        sequence_representations.append(partner1_MT)
        sequence_representations = np.array(sequence_representations)

        return sequence_representations

    def readOtherFASTA(self, ChainID, s):
        fp = open(self.PDBid+'.pdb')
        # this MISS_RES only record the target chain 'ChainID'
        array_RES = {}
        flagMISSRES = False; marker = ''
        for line in fp:
            words = re.split(' |\n', line)
            if len(words) > 4:
                if words[2] == 'MISSING' and words[3] == 'RESIDUES':
                    marker = words[1]
                    flagMISSRES = True
                    break
        if flagMISSRES:
            for _ in range(5): # skip 5 lines
                fp.readline()
            line = re.split(' |\n', fp.readline())
            line = [i for i in line if i]
            while line[1] == marker:
                if line[3] in ChainID:
                    if line[2] in AminoA:
                        array_RES[int(line[4])] = three_to_one(line[2])
                    #elif line[2] != 'HOH':
                    #    self.non_canonical.append(line[2])
                line = re.split(' |\n', fp.readline())
                line = [i for i in line if i]
        fp.close()

        # if array_RES is empty, then jump is not allowed or
        # the largest index of MISSING RESIDUE is less than starter residue
        residue = next(iter(s[0][ChainID])) # what's this? # gives the first residue
        start_idx_PDB  = residue.id[1]
        if len(array_RES) == 0:
            flagMISSRES = False
        else:
            array_RES_idx = list(array_RES.keys()) # find the last index
            last_idx_MISS = array_RES_idx[-1]
            if last_idx_MISS < start_idx_PDB:
                flagMISSRES = False
        # if flagMISSRES = False, then jump index is not allowed

        shift = 0;# last_idx 
        resID_mute_in_ = 0
        for idx, iresidue in enumerate(s[0][ChainID]):
            if iresidue.resname in AminoA:
                if iresidue.id[2] != ' ': # Here is the insertion code appearing
                    shift += 1
                if flagMISSRES:
                    array_RES[iresidue.id[1]+shift] = three_to_one(iresidue.resname)
                else:
                    array_RES[start_idx_PDB+idx] = three_to_one(iresidue.resname)
            elif iresidue.resname in NC_AminoA:
                iresidue_resname = NC_AminoA[iresidue.resname]
                if iresidue.id[2] != ' ': # Here is the insertion code appearing
                    shift += 1
                if flagMISSRES:
                    array_RES[iresidue.id[1]+shift] = three_to_one(iresidue_resname)
                else:
                    array_RES[start_idx_PDB+idx] = three_to_one(iresidue_resname)

        # sort array_RES by index
        array_RES_sorted = {}
        for idx in sorted(array_RES):
            array_RES_sorted[idx] = array_RES[idx]
        #print(array_RES_sorted.keys())

        # get the start and end index, real mutation ID
        array_RES_idx = list(array_RES_sorted.keys())
        idx_s = array_RES_idx[0]; idx_e = array_RES_idx[-1]
        # 0 in array_RES_idx, when flagMISSRES == True
        if 0 not in array_RES_idx and idx_s < 0:
            seqlength = idx_e-idx_s
        else:
            seqlength = idx_e-idx_s + 1

        if seqlength == len(array_RES_idx):
            fasta = ''.join(list(array_RES_sorted.values()))
            return fasta

        # use SEQRES in PDB for fasta file
        array_RES = {}; idx = idx_s
        fp = open(self.PDBid+'.pdb')
        for line in fp:
            if line[:6] == 'SEQRES':
                words = [i for i in re.split(' |\n', line) if i]
                if words[2] == ChainID:
                    for resname in words[4:]:
                        if resname in AminoA:
                            array_RES[idx] = three_to_one(resname)
                            idx += 1
        fp.close()
        fasta = ''.join(list(array_RES.values()))
        return fasta

    def compareSeqFasta(self, ChainID):
        from Bio.SeqIO.FastaIO import SimpleFastaParser
        # if the PDB cannot contains a consecutive FASTA sequence
        self.fasta_PDB = ''

        if not os.path.exists(self.PDBid+'.fasta'):
            os.system('wget https://www.rcsb.org/fasta/entry/'+self.PDBid)
            os.system('mv '+self.PDBid+' '+self.PDBid+'.fasta')

        # values is a tuple (info, fasta)
        # exampel: 1E50_1|Chains A,C,E,G,Q,R[auth I]|CORE-BINDING FACTOR ALPHA SUBUNIT|HOMO SAPIENS
        # want to get ['A', 'C', 'E', 'G', 'Q', 'R'] and 'I'
        # !!! PDBid|Chains A[auth B]|
        # !!! PDBid|Chains B[auth A]| needs to tell the difference
        # from Bio.SeqIO.FastaIO import SimpleFastaParser to read the fasta file
        ## from BioPython structure, import chains
        #for chain in s[0]:
        #    print(chain.id)
        fasta_file = SimpleFastaParser(open(self.PDBid+'.fasta'))
        fasta = {}
        for values in fasta_file:
            # two ways: 1. only save the fasta that we need; 2. save the fasta for all chains
            #ichain = [i for i in self.Chains if i in fasta_chains][0]
            #self.fasta[ichain] = values[1]
            chain_info = values[0].split('|')[1]
            fasta_chains_temp = re.split(' |\[|\]', chain_info)
            fasta_chains = fasta_chains_temp[1].split(',')
            for i_chain in fasta_chains_temp[2:]:
                fasta_chains += i_chain.split(',')
            for ichain in fasta_chains:
                fasta[ichain] = values[1]
        fasta_target = fasta[ChainID]
        #print(fasta_target)
        #print(self.fasta[ChainID])
        if fasta_target == self.fasta[ChainID]:
            return (True, fasta_target)
        else:
            if len(self.fasta[ChainID]) == len(fasta_target):
                print(self.PDBid, ChainID, 'length same but AA different')
                return (False, fasta_targe)
            elif len(self.fasta[ChainID]) < len(fasta_target):
                for i in range(len(fasta_target)-len(self.fasta[ChainID])+1):
                    if self.fasta[ChainID] == fasta_target[i:i+len(self.fasta[ChainID])]:
                        return (True, fasta_target)
                print(self.PDBid, ChainID, 'self.fasta length less')
                return (False, fasta_target)
            else:
                print(self.PDBid, ChainID, 'self.fasta length longer')
                return (False, fasta_target)

    def readFASTA_(self):
        # Check https://biopython.org/docs/1.75/api/Bio.PDB.Polypeptide.html
        # MISSING RESIDUES are still problem
        from Bio.PDB.Polypeptide import PPBuilder
        ppb = PPBuilder()
        for pp in ppb.build_peptides(self.s, aa_only=False):
            print(pp.get_sequence())
        return # readFASTA_

        #print(len(self.fasta[ChainID]), len(fasta_target))
        #print(fasta_target==self.fasta[ChainID])
        #return

        ## for fasta reading
            #if 'auth' in fasta_chains_temp:
            #    index = fasta_chains_temp.index('auth')+1
            #    fasta_chains = fasta_chains_temp[index].split(',')
            #else:
            #    fasta_chains = fasta_chains_temp[1].split(',')
            #    for i_chain in fasta_chains_temp[2:]:
            #        fasta_chains += i_chain.split(',')
            #if 'auth' not in fasta_chains_temp:
            #    fasta_chains = fasta_chains_temp[1].split(',')
            #else:
            #    fasta_chains = fasta_chains_temp[1].split(',')
            #    fasta_chains += fasta_chains_temp[3].split(',')
            #print(fasta_chains)

        ##print(self.PDBid, self.muteChain, self.resID, len(self.fasta[self.muteChain]))
        ## , self.fasta[self.muteChain][self.resID], self.resWT)
        #if len(self.fasta[self.muteChain]) <= self.resID:
        #    return False
        #if self.fasta[self.muteChain][self.resID]!=self.resWT:
        #    #sys.exit('Wrong residue name after loading fasta file')
        #    print(self.fasta[self.muteChain])
        #    print(self.resID, self.fasta[self.muteChain][self.resID-1], self.resWT)
        #    print('Cannot find the right residue ID between PDB file and fasta file')
        #    print('Using PDB file to generate the fasta file')
        #    #os.system('rm -rf *.pssm')
        #    return False

        ## write up fasta_file for wild type and mutant
        #seqfile_WT = open(self.fileMuteChain+'_WT.fasta', 'w')
        #seqfile_MT = open(self.fileMuteChain+'_MT.fasta', 'w')
        #self.fasta_WT = ''
        #self.fasta_MT = ''
        #for idx, c in enumerate(fasta_target):
        #    seqfile_WT.write(c)
        #    self.fasta_WT += c
        #    if idx==self.resID:
        #        seqfile_MT.write(self.resMT)
        #        self.fasta_MT += self.resMT
        #    else:
        #        seqfile_MT.write(c)
        #        self.fasta_MT += c
        #    if (idx+1)%80==0:
        #        seqfile_WT.write('\n')
        #        seqfile_MT.write('\n')
        #seqfile_WT.close()
        #seqfile_MT.close()
        #if len(self.fasta_MT) < 15:
        #    seqfile_WT = open(self.fileMuteChain+'_WT.fasta', 'w')
        #    seqfile_MT = open(self.fileMuteChain+'_MT.fasta', 'w')
        #    for i in range(len(self.fasta_WT)*20):
        #        seqfile_WT.write(self.fasta_WT[i%len(self.fasta_WT)])
        #        seqfile_MT.write(self.fasta_MT[i%len(self.fasta_MT)])
        #        if (i+1)%80==0:
        #            break
        #    seqfile_WT.close()
        #    seqfile_MT.close()
        #return True

if __name__ == '__main__':
    #s = get_structure('1DVF', 'CD', 'D', 'R', '100b', 'A', '7.0')
    #s = get_structure('1E50', 'AB', 'B', 'P', '100', 'A', '7.0')
    s = get_structure(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    #s.generateMutedPDBs()
    s.readFASTA()
    print(s.fasta)
    print(s.muteChain, s.Chains, s.OtherChains)
    #s.readFASTA_()
    #s.compareSeqFasta(sys.argv[4])
    #print(s.compareSeqFasta())
    #print((s.fastaWT))
    #print((s.fastaMT))
    #s.clean_files()
