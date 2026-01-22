import os, pickle, operator, sys, time
import gudhi, re
import numpy as np
import scipy as sp
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB.DSSP import DSSP
import subprocess#Yiming
from Bio.PDB.DSSP import make_dssp_dict#yiming
from Bio.Blast.Applications import NcbipsiblastCommandline
from scipy.spatial import cKDTree
from vr_facet import vr_facet
from alpha_facet import alpha_facet
from vr_fvector import vr_fvector
from alpha_fvector import alpha_fvector
from math import sqrt

FRIDefault = [['Lorentz', 0.5,  5],
              ['Lorentz', 1.0,  5],
              ['Lorentz', 2.0,  5],
              ['Exp',     1.0, 15],
              ['Exp',     2.0, 15]]
ElementList = ['C', 'N', 'O']
ElementTau = np.array([6., 1.12, 1.1])
EleLength = len(ElementList)
ele2index = {'C':0, 'N':1, 'O':2, 'S':3, 'H':4}
ss2index = {'H':1, 'E':2, 'G':3, 'S':4, 'B':5, 'T':6, 'I':7, 'P':8, '-':0}
Hydro = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
PolarAll = ['S','T','N','Q','R','H','K','D','E']
PolarUncharged = ['S','T','N','Q']
PolarPosCharged = ['R','H','K']
PolarNegCharged = ['D','E']
SpecialCase = ['C','U','G','P']
AAvolume = {'A': 88.6, 'R':173.4, 'D':111.1, 'N':114.1, 'C':108.5, \
            'E':138.4, 'Q':143.8, 'G': 60.1, 'H':153.2, 'I':166.7, \
            'L':166.7, 'K':168.6, 'M':162.9, 'F':189.9, 'P':112.7, \
            'S': 89.0, 'T':116.1, 'W':227.8, 'Y':193.6, 'V':140.0}
AAhydropathy = {'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5, \
                'E':-3.5, 'Q':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5, \
                'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6, \
                'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2}
AAarea = {'A':115., 'R':225., 'D':150., 'N':160., 'C':135., \
          'E':190., 'Q':180., 'G': 75., 'H':195., 'I':175., \
          'L':170., 'K':200., 'M':185., 'F':210., 'P':145., \
          'S':115., 'T':140., 'W':255., 'Y':230., 'V':155.}
AAweight = {'A': 89.094, 'R':174.203, 'N':132.119, 'D':133.104, 'C':121.154, \
            'E':147.131, 'Q':146.146, 'G': 75.067, 'H':155.156, 'I':131.175, \
            'L':131.175, 'K':146.189, 'M':149.208, 'F':165.192, 'P':115.132, \
            'S':105.093, 'T':119.12 , 'W':204.228, 'Y':181.191, 'V':117.148}
AApharma = {'A':[0,1,3,1,1,1],'R':[0,3,3,2,1,1],'N':[0,2,4,1,1,0],'D':[0,1,5,1,2,0],\
            'C':[0,2,3,1,1,0],'E':[0,1,5,1,2,0],'Q':[0,2,4,1,1,0],'G':[0,1,3,1,1,0],\
            'H':[0,3,5,3,1,0],'I':[0,1,3,1,1,2],'L':[0,1,3,1,1,1],'K':[0,2,4,2,1,2],\
            'M':[0,1,3,1,1,2],'F':[1,1,3,1,1,1],'P':[0,1,3,1,1,1],'S':[0,2,4,1,1,0],\
            'T':[0,2,4,1,1,1],'W':[2,2,3,1,1,2],'Y':[1,2,4,1,1,1],'V':[0,1,3,1,1,1]}
Groups = [Hydro, PolarAll, PolarUncharged, PolarPosCharged, PolarNegCharged, SpecialCase]


electronegativity = {
            'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58,\
            'H': 2.20, 'P': 2.19, 'F': 3.98, 'CL': 3.16,\
            'BR': 2.96, 'I': 2.66}

def get_atom_electronegativity(atom_type):
    """Retrieves Pauling electronegativity for a given atom type."""
    clean_type = atom_type.upper()
    if not clean_type: return 0.0
    if clean_type in electronegativity:
        return electronegativity[clean_type]
    elif clean_type[0] in electronegativity:
        return electronegativity[clean_type[0]]
    return 0.0
def atmtyp_to_ele( st ):
    if len(st.strip()) == 1:
        return st.strip()
    elif st[0] == 'H':
        return 'H'
    elif st == "CA":
        return "CA"
    elif st == "CL":
        return "CL"
    elif st == "BR":
        return "BR"
    else:
        print(st, 'Not in dictionary')
        return

def AAcharge(AA):
    if AA in ['D','E']:
        return -1.
    elif AA in ['R','H','K']:
        return 1.
    else:
        return 0.

class atom:
    def __init__(self, AType, AVType, Charge, Chain, ResName, ResID, Radii):
        self.pos         = None
        self.atype       = AType.replace(' ', '')
        self.verboseType = AVType
        self.Charge      = Charge
        self.ResName     = ResName
        self.ResID       = ResID
        self.R           = Radii
        self.Chain       = Chain
        self.Area        = 0.
        self.SolvEng     = 0.
    def position(self, pos):
        self.pos = pos
    def calcharge(self, charg):
        self.Charge = charg

class protein:
    def __init__(self, structure, typeFlag, onlyBLAST=False, mode='charge'):
        self.PDBid    = structure.PDBid
        self.Chain    = structure.muteChain
        self.ResIDSeq = structure.resID_fasta # PSSM will use this
        self.typeFlag = typeFlag
        self.mode = mode.lower()
        if typeFlag == 'WT':
            self.ResName  = structure.resWT
            self.Sequence = structure.fastaWT
        elif typeFlag == 'MT':
            self.ResName  = structure.resMT
            self.Sequence = structure.fastaMT
        else:
            sys.exit('wrong typeFlag for PPIprotein.py')

        self.filename = structure.PDBid+'_'+self.typeFlag
        self.filename_single = '_'.join([self.PDBid, self.Chain, self.typeFlag])

        if not onlyBLAST:
            self.ResID = structure.resID_MutedPartner
            if os.path.exists(self.filename+'.pqr'):
                self.loadPQRFile()
            if os.path.exists(self.filename+'.propka'):
                self.get_pka_info()
            self.IndexList = self.construct_index_list()
            self.setup_pairwise_interaction()

            self.SeqLength = len(structure.fasta['WT'])

    def loadPQRFile(self):
        print('load PQR file >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        ## Atom position from PQR file
        self.AtomPos = []
        self.Atoms   = []
        self.Charge  = []

        PQRFile = open(self.filename+'.pqr')
        for line in PQRFile:
            if line[0:4] == 'ATOM':
                resname = line[17:20]
                #if resname=='HSE': 
                #    resname='HIS'
                self.AtomPos.append([float(line[26:38]), float(line[38:46]), float(line[46:54])])
                Atom = atom(atmtyp_to_ele(line[12:14]), line[11:17], float(line[54:62]),
                            line[21], three_to_one(resname), int(line[22:26]), float(line[62:69]))
                self.Atoms.append(Atom)
                self.Charge.append(float(line[54:62]))
        PQRFile.close()
        self.AtomNum = len(self.Atoms)
        self.AtomPos = np.array(self.AtomPos)
        self.Charge = np.array(self.Charge, float)
        for idx, iPos in enumerate(self.AtomPos):
            self.Atoms[idx].position(iPos)
        for idx, icharge in enumerate(self.Charge):
            self.Atoms[idx].calcharge(icharge)
        return

    def get_pka_info(self):
        print('get pKa information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        pKaFile = open(self.filename+'.propka')
        self.pKaSite = 0.0
        self.pKa = []; self.pKaName = []
        self.pKaCt = 0.; self.PKaNt = 0.
        for line in pKaFile:
            if len(line)<24:
                continue
            if line[23]=='%':
                if line[0:2]=='C-': self.pKaCt = float(line[11:16])
                if line[0:2]=='N+': self.pKaNt = float(line[11:16])
                resid = int(line[3:7])
                if resid!=self.ResID:
                    self.pKa.append(float(line[11:16]))
                    self.pKaName.append(line[0:3])
                else:
                    self.pKaSite = float(line[11:16])
        pKaFile.close()
        return

    def construct_index_list(self, CutNear=10.):
        """ Lists that contains atom index
            first index: 0 mutsite, 1 other near, 2 all
            seconde index: 0 C, 1 N, 2 O, 3 S, 4 H, 5 heavy, 6 all
        """
        print('constructing index list >>>>>>>>>>>>>>>>>>>>>>>>>>>')
        heavy = ['C', 'N', 'O', 'S']
        IndexList = [[[] for i in range(7)] for i in range(3)]

        # get the mutation residue index on atoms
        IndexMutSite = []; PosMutSite = []
        checkFlag = False
        for idx, iAtom in enumerate(self.Atoms):
            #print(iAtom.ResID, iAtom.Chain, iAtom.ResName)
            if iAtom.ResID==self.ResID and iAtom.Chain==self.Chain and iAtom.ResName==self.ResName:
                IndexMutSite.append(idx)
                PosMutSite.append(iAtom.pos)
                checkFlag = True
        if not checkFlag:
            print(iAtom.ResID, iAtom.Chain, iAtom.ResName)
            print(self.ResID,  self.Chain,  self.ResName)
            #os.system('rm '+self.filename+'*')
            sys.exit('Wrong residue ID and name in construct_index_list() '+self.typeFlag)
        PosMutSite = np.array(PosMutSite)

        # collect Near Residue ID. Not just the near atoms, but the atoms of near residue
        NearRes = []
        for idx, iAtom in enumerate(self.Atoms):
            ChainResID = iAtom.Chain+str(iAtom.ResID)
            if ChainResID not in NearRes and ChainResID != self.Chain+str(self.ResID):
                if np.min(np.linalg.norm(iAtom.pos-PosMutSite, axis=1)) < CutNear:
                    NearRes.append(ChainResID)
        NearAtom = []
        for idx, iAtom in enumerate(self.Atoms):
            if iAtom.Chain+str(iAtom.ResID) in NearRes:
                NearAtom.append(idx)

        # Index of mutation site
        for idx in IndexMutSite:
            IndexList[0][ele2index[self.Atoms[idx].atype]].append(idx)
            if self.Atoms[idx].atype in heavy:
                IndexList[0][5].append(idx)
            IndexList[0][6].append(idx)
        # Index of near atoms
        for idx in NearAtom:
            IndexList[1][ele2index[self.Atoms[idx].atype]].append(idx)
            if self.Atoms[idx].atype in heavy:
                IndexList[1][5].append(idx)
            IndexList[1][6].append(idx)
        # Index of all atoms
        for idx, iAtom in enumerate(self.Atoms):
            IndexList[2][ele2index[self.Atoms[idx].atype]].append(idx)
            if iAtom.atype in heavy:
                IndexList[2][5].append(idx)
            IndexList[2][6].append(idx)

        for i in range(3):
            for j in range(7):
                IndexList[i][j]=np.array(IndexList[i][j], int)
        return IndexList # construct_index_list()

    def setup_pairwise_interaction(self, sCut=10., lCut=40):
        print('setup pairwise interaction >>>>>>>>>>>>>>>>>>>>>>>>')
        from src import PyProtein
        PyProt = PyProtein(self.PDBid.encode('utf-8'))
        if not PyProt.loadPQRFile((self.filename+'.pqr').encode('utf-8')):
            sys.exit('PyProtein reads PQR file filed')
        PyProt.atomwise_interaction(sCut, lCut)
        self.CLB = PyProt.feature_CLB()
        self.VDW = PyProt.feature_VDW()
        self.FRI1, self.FRI2, self.FRI3, self.FRI4, self.FRI5 = PyProt.feature_FRIs()
        PyProt.Deallocate()
        return

    def rips_complex(self, cutoff=16, deathcut=11):
        elecomb = ['C', 'N', 'O']
        #Bins = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
        BinIdx = int((deathcut-2))
        Bins = np.linspace(2, deathcut, BinIdx+1)
        def BinID(x):
            for i in range(BinIdx):
                if Bins[i] <= x <= Bins[i+1]:
                    return i
            return BinIdx

        Atoms = self.Atoms
        for iAtom in Atoms:
            if iAtom.verboseType.replace(' ', '')=="CA" and iAtom.ResID==self.ResID:
                AtomCA = iAtom
        res_num = self.ResID

        rips_dth = np.zeros([3,3,12], int)
        rips_bar = np.zeros([3,3,12], int)
        dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

        for Ip, ele1 in enumerate(elecomb):
            for Is, ele2 in enumerate(elecomb):
                pts = []; ptid = [] # ptid to indicate if atom belongs to mutated residue id
                for iAtom in Atoms:
                    if (iAtom.atype.replace(' ','')==ele1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==ele2 and iAtom.ResID==res_num):
                        dis = np.linalg.norm(iAtom.pos-AtomCA.pos)
                        if dis <= cutoff:
                            pts.append(iAtom.pos)
                            if iAtom.ResID==res_num:
                                ptid.append(0)
                            else:
                                ptid.append(1)
                matrixA = np.ones((len(pts), len(pts)))*100.
                for ii in range(len(pts)):
                    matrixA[ii, ii] = 0.
                    for jj in range(ii+1, len(pts)):
                        if ptid[ii]+ptid[jj]==1:
                            dis = np.linalg.norm(pts[ii]-pts[jj])
                            matrixA[ii, jj] = dis
                            matrixA[jj, ii] = dis
        
                rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=deathcut)
                PH = rips_complex.create_simplex_tree().persistence()

                tmpbars = np.zeros(len(pts), dtype=dt)
                cnt = 0
                for simplex in PH:
                    dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
                    if d-b < 0.1: continue
                    tmpbars[cnt]['dim']   = dim
                    tmpbars[cnt]['birth'] = b
                    tmpbars[cnt]['death'] = d
                    cnt += 1
                bars = tmpbars[0:cnt]
                for bar in bars:
                    death = bar['death']
                    if death >= deathcut: continue
                    Did = BinID(death)
                    rips_dth[Ip, Is,  Did] += 1
                    rips_bar[Ip, Is, :Did] += 1

        return np.array(rips_dth), np.array(rips_bar)

    def rips_complex_sr(self, cutoff=16, deathcut=11):

        print('Calculating SR-Rips >>>>>>>>>>>>>>>>>>>>>>>>>>>')
        elecomb = ['C', 'N', 'O']
        #Bins = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
        BinIdx = int((deathcut-2))

        Atoms = self.Atoms
        for iAtom in Atoms:
            if iAtom.verboseType.replace(' ', '')=="CA" and iAtom.ResID==self.ResID:
                AtomCA = iAtom
        res_num = self.ResID

        rips_curves = np.zeros([3,3,30], int)
        rips_rates = np.zeros([3,3,30], float)

        for Ip, ele1 in enumerate(elecomb):
            for Is, ele2 in enumerate(elecomb):
                pts = []; ptid = [] # ptid to indicate if atom belongs to mutated residue id
                for iAtom in Atoms:
                    if (iAtom.atype.replace(' ','')==ele1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==ele2 and iAtom.ResID==res_num):
                        dis = np.linalg.norm(iAtom.pos-AtomCA.pos)
                        if dis <= cutoff:
                            pts.append(iAtom.pos)
                            if iAtom.ResID==res_num:
                                ptid.append(0)
                            else:
                                ptid.append(1)
                matrixA = np.ones((len(pts), len(pts)))*100.
                for ii in range(len(pts)):
                    matrixA[ii, ii] = 0.
                    for jj in range(ii+1, len(pts)):
                        if ptid[ii]+ptid[jj]==1:
                            dis = np.linalg.norm(pts[ii]-pts[jj])
                            matrixA[ii, jj] = dis
                            matrixA[jj, ii] = dis

                model = vr_facet(matrixA, max_dim = 2, min_edge_length = 2, max_edge_length = deathcut, num_samples=BinIdx+1)
                #tmp = np.hstack(model.facet_curves())
                #print(np.shape(tmp))
                rips_curves[Ip, Is] = np.hstack(model.facet_curves())
                rips_rates[Ip, Is] = np.hstack(model.facet_rates())

        return np.array(rips_curves), np.array(rips_rates)

    def rips_complex_fvector(self, cutoff=16, deathcut=11):

        print('Calculating SR-Rips >>>>>>>>>>>>>>>>>>>>>>>>>>>')
        elecomb = ['C', 'N', 'O']
        #Bins = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
        BinIdx = int((deathcut-2))

        Atoms = self.Atoms
        for iAtom in Atoms:
            if iAtom.verboseType.replace(' ', '')=="CA" and iAtom.ResID==self.ResID:
                AtomCA = iAtom
        res_num = self.ResID

        rips_curves = np.zeros([3,3,30], int)
        rips_rates = np.zeros([3,3,30], float)

        for Ip, ele1 in enumerate(elecomb):
            for Is, ele2 in enumerate(elecomb):
                pts = []; ptid = [] # ptid to indicate if atom belongs to mutated residue id
                for iAtom in Atoms:
                    if (iAtom.atype.replace(' ','')==ele1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==ele2 and iAtom.ResID==res_num):
                        dis = np.linalg.norm(iAtom.pos-AtomCA.pos)
                        if dis <= cutoff:
                            pts.append(iAtom.pos)
                            if iAtom.ResID==res_num:
                                ptid.append(0)
                            else:
                                ptid.append(1)
                matrixA = np.ones((len(pts), len(pts)))*100.
                for ii in range(len(pts)):
                    matrixA[ii, ii] = 0.
                    for jj in range(ii+1, len(pts)):
                        if ptid[ii]+ptid[jj]==1:
                            dis = np.linalg.norm(pts[ii]-pts[jj])
                            matrixA[ii, jj] = dis
                            matrixA[jj, ii] = dis

                model = vr_fvector(matrixA, max_dim = 2, min_edge_length = 2, max_edge_length = deathcut, num_instances=BinIdx+1)
                #tmp = np.hstack(model.facet_curves())
                #print(np.shape(tmp))
                rips_curves[Ip, Is] = np.hstack(list(model.compute_f_vector_curves().values())) # the f-vector curves components stacked
                rips_rates[Ip, Is] = np.hstack(list(model.compute_rate_curves().values())) # the average of the radius f/t

        return np.array(rips_curves), np.array(rips_rates)

    def rips_complex_spectra(self, cutoff=16):
        elecomb = ['C', 'N', 'O']
        #Bins = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
        #Bins = [2, 3, 4, 5, 6, 7, 9, 11]
        Bins = [3, 4, 5, 6, 7, 9]
        features = np.zeros((len(Bins)*len(elecomb)*len(elecomb), 8), float)
        def BinID(x, B):
            for i in range(len(B)-1):
                if B[i] <= x <= B[i+1]:
                    y = i
            return y

        Atoms = self.Atoms
        for iAtom in Atoms:
            if iAtom.verboseType.replace(' ', '')=="CA" and iAtom.ResID==self.ResID:
                AtomCA = iAtom
        res_num = self.ResID

        for Ip, ele1 in enumerate(elecomb):
            for Is, ele2 in enumerate(elecomb):
                pts = []; ptid = [] # ptid to indicate if atom belongs to mutated residue id
                for iAtom in Atoms:
                    if (iAtom.atype.replace(' ','')==ele1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==ele2 and iAtom.ResID==res_num):
                        dis = np.linalg.norm(iAtom.pos-AtomCA.pos)
                        if dis <= cutoff:
                            pts.append(iAtom.pos)
                            if iAtom.ResID==res_num:
                                ptid.append(0)
                            else:
                                ptid.append(1)
                matrixA = np.ones((len(pts), len(pts)))*100.
                for ii in range(len(pts)):
                    matrixA[ii, ii] = 0.
                    for jj in range(ii+1, len(pts)):
                        if ptid[ii]+ptid[jj]==1:
                            dis = np.linalg.norm(pts[ii]-pts[jj])
                            matrixA[ii, jj] = dis
                            matrixA[jj, ii] = dis

                for idx_cut, cut in enumerate(Bins):
                    Laplacian = np.zeros((len(pts), len(pts)), int)
                    Laplacian[matrixA < cut] = -1
                    Laplacian += np.diagflat(-np.sum(Laplacian, axis=0))
                    eigens = np.sort(np.linalg.eigvalsh(Laplacian))

                    idx_feat = idx_cut * 3 * 3 + Ip * 3 + Is
                    eigens = eigens[eigens > 10 ** -8]
                    if len(eigens) > 0:
                        # sum, min, max, mean, std, var,
                        features[idx_feat][0] = eigens.sum()
                        features[idx_feat][1] = eigens.min()
                        features[idx_feat][2] = eigens.max()
                        features[idx_feat][3] = eigens.mean()
                        features[idx_feat][4] = eigens.std()
                        features[idx_feat][5] = eigens.var()
                        features[idx_feat][6] = np.dot(eigens, eigens)
                        features[idx_feat][7] = len(eigens[eigens > 10 ** -8])

        return features.flatten() #rips_complex_spectra

   
    def get_atom_value(self, atom):
        """Returns the scalar value (Charge or EN) based on initialized mode."""
        if self.mode == 'charge':
            return getattr(atom, 'Charge', 0.0)
        else:
            raw_en = get_atom_electronegativity(atom.atype)
            return raw_en / 4.0

    def compute_manual_laplacian(self, st, cutoff, values, num_pts, pt_group):
        """
        Manually builds the Laplacian matrix L0 = D.T @ D.
        This bypasses petls indexing issues by strictly using the SimplexTree indices.
        
        Args:
            st: GUDHI SimplexTree
            cutoff: Distance threshold
            charges: List of charges for all points
            num_pts: Total number of points (vertices)
            pt_group: List indicating group (Mut=0, Env=1) for bipartite check
        """
        
        active_edges = []
        
 
        for simplex, filt in st.get_filtration():
            if len(simplex) == 2 and filt <= cutoff:
                u, v = sorted(simplex)
                
                if pt_group[u] != pt_group[v]:
                    active_edges.append((u, v, filt))
        
        if not active_edges:
            return None 

        num_edges = len(active_edges)
        
    
        D = np.zeros((num_edges, num_pts))
        
        for row_idx, (u, v, dist) in enumerate(active_edges):
            q_u = values[u]
            q_v = values[v]
            
            d = dist if dist > 1e-9 else 1e-9
            rho_u = q_v / d
            rho_v = q_u / d
            
            D[row_idx, u] = -rho_u
            D[row_idx, v] = rho_v
            
        L_matrix = D.T @ D
        return L_matrix
   
    def rips_complex_sheaf_spectra(self, cutoff=16):
        elecomb = ['C', 'N', 'O']
        Bins = [3, 4, 5, 6, 7, 8, 9]
        
       
        features = np.zeros((len(Bins) * len(elecomb) * len(elecomb), 9), float)

     
        AtomCA = None
        for iAtom in self.Atoms:
            v_type = iAtom.verboseType.strip() if hasattr(iAtom, 'verboseType') else ""
            if v_type == "CA" and iAtom.ResID == self.ResID:
                AtomCA = iAtom
                break
        
        if AtomCA is None:
            return features.flatten()

        res_num = self.ResID

        for Ip, ele1 in enumerate(elecomb):
            for Is, ele2 in enumerate(elecomb):
                
                pts = []     # [x, y, z]
                sheaf_values = []
                pt_group = [] 
                
              
                for iAtom in self.Atoms:
                    if iAtom.pos is None: continue 
                    clean_type = iAtom.atype.replace(' ', '')
                    is_ele1_env = (clean_type == ele1 and iAtom.ResID != res_num)
                    is_ele2_mut = (clean_type == ele2 and iAtom.ResID == res_num)
                    
                    if is_ele1_env or is_ele2_mut:
                        dis = np.linalg.norm(iAtom.pos - AtomCA.pos)
                        if dis <= cutoff:
                            pts.append(list(iAtom.pos))
                            val = self.get_atom_value(iAtom)
                            sheaf_values.append(val)
                            if is_ele2_mut: pt_group.append(0)
                            else: pt_group.append(1)

                num_pts = len(pts)
                if num_pts < 2:
                    continue

                max_r = max(Bins) + 0.1

                dist_matrix = np.full((num_pts, num_pts), 10000.0, dtype=float)
                
                np.fill_diagonal(dist_matrix, 0.0)
                
                for i in range(num_pts):
                    for j in range(i + 1, num_pts):
                        if pt_group[i] + pt_group[j] == 1:
                            d = sqrt((pts[i][0]-pts[j][0])**2 + 
                                     (pts[i][1]-pts[j][1])**2 + 
                                     (pts[i][2]-pts[j][2])**2)
                            dist_matrix[i][j] = d
                            dist_matrix[j][i] = d

                #rips = gudhi.RipsComplex(distance_matrix=dist_matrix, max_edge_length=max_r)
                rips = gudhi.RipsComplex(points=pts, max_edge_length=max_r)
                st = rips.create_simplex_tree(max_dimension=1)

                for idx_cut, cut in enumerate(Bins):
                    idx_feat = idx_cut * 9 + Ip * 3 + Is
                    
                    try:
                        L_matrix = self.compute_manual_laplacian(st, float(cut), sheaf_values, num_pts, pt_group)
                        
                        if L_matrix is not None and L_matrix.shape[0] > 0:
                            eigens = np.linalg.eigvalsh(L_matrix)
                            valid_eigens = eigens[eigens > 1e-8]
                            num_zeros = len(eigens) - len(valid_eigens) 
                            
                            if len(valid_eigens) > 0:
                                if self.mode =='en':
                                    features[idx_feat][0] = np.log1p(valid_eigens.sum())
                                    features[idx_feat][1] = valid_eigens.min()
                                    features[idx_feat][2] = np.log1p(valid_eigens.max())
                                    features[idx_feat][3] = np.log1p(valid_eigens.mean())
                                    features[idx_feat][4] = np.log1p(valid_eigens.std())
                                    features[idx_feat][5] = np.log1p(valid_eigens.var())
                                    features[idx_feat][6] = np.log1p(np.dot(valid_eigens, valid_eigens))
                                    features[idx_feat][7] = len(valid_eigens) 
                                    features[idx_feat][8] = num_zeros 
                                else:
                                    features[idx_feat][0] = valid_eigens.sum()
                                    features[idx_feat][1] = valid_eigens.min()
                                    features[idx_feat][2] = valid_eigens.max()
                                    features[idx_feat][3] = valid_eigens.mean()
                                    features[idx_feat][4] = valid_eigens.std()
                                    features[idx_feat][5] = valid_eigens.var()
                                    features[idx_feat][6] = np.dot(valid_eigens, valid_eigens)
                                    features[idx_feat][7] = len(valid_eigens)
                                    features[idx_feat][8] = num_zeros

                            else:
                                features[idx_feat][8] = len(eigens)

                    except Exception:
                        continue

        return features.flatten()

    def compute_alpha_L1_matrix(self, st, cutoff_sq, values, pts, num_pts):
        """
        Builds the 1-Dimensional Laplacian (Edges vs Triangles) for the Alpha Complex.
        L1 = D0 @ D0.T + D1.T @ D1
        
        Note: Alpha Complex filtration values are squared distances.
        """
        
        active_edges = []
        active_triangles = []
        edge_to_idx = {}
        
        for simplex, filt in st.get_filtration():
            if filt > cutoff_sq:
                continue
                
            if len(simplex) == 2:
                u, v = sorted(simplex)
                edge_to_idx[(u, v)] = len(active_edges)
                active_edges.append((u, v, sqrt(filt)))
                
            elif len(simplex) == 3:
                u, v, w = sorted(simplex)
                active_triangles.append((u, v, w))

        num_edges = len(active_edges)
        num_tri = len(active_triangles)
        
        if num_edges == 0:
            return None
            
        D0 = np.zeros((num_edges, num_pts))
        for row_idx, (u, v, d) in enumerate(active_edges):
            q_u = values[u]
            q_v = values[v]
            dist = d if d > 1e-9 else 1e-9
            
            D0[row_idx, u] = -q_v / dist
            D0[row_idx, v] = q_u / dist
            
        D1 = np.zeros((num_tri, num_edges))
        
        for tri_idx, (u, v, w) in enumerate(active_triangles):
            e_uv = (u, v)
            e_uw = (u, w)
            e_vw = (v, w)
            
            def get_dist(i, j):
                return np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))

            d_uv = get_dist(u, v)
            d_uw = get_dist(u, w)
            d_vw = get_dist(v, w)
            
            # Ensure edges exist in active set (Alpha complex ensures this property)
            if e_uv in edge_to_idx and e_uw in edge_to_idx and e_vw in edge_to_idx:
                idx_uv = edge_to_idx[e_uv]
                idx_uw = edge_to_idx[e_uw]
                idx_vw = edge_to_idx[e_vw]
                
                # Charges
                q_u = values[u]
                q_v = values[v]
                q_w = values[w]
                
                # Formula based on user snippet:
                # Edge opposite w (uv): +q_w / (d_uw * d_vw)
                val_uv = q_w / (d_uw * d_vw)
                
                # Edge opposite v (uw): -q_v / (d_uv * d_vw)
                val_uw = -q_v / (d_uv * d_vw)
                
                # Edge opposite u (vw): +q_u / (d_uv * d_uw)
                val_vw = q_u / (d_uv * d_uw)
                
                D1[tri_idx, idx_uv] = val_uv
                D1[tri_idx, idx_uw] = val_uw
                D1[tri_idx, idx_vw] = val_vw

        L1 = (D0 @ D0.T) + (D1.T @ D1)
        return L1

    # def alpha_complex_sheaf_spectra(self):
    #     """
    #     Computes 1-Dim Sheaf Laplacian features using Alpha Complex.
    #     Can be concatenated with Rips features.
    #     """
    #     elecomb = ['C', 'N', 'O']
    #     Bins = [3, 4, 5, 6, 7, 8, 9]
    #     #Bins = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    #     #Bins = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0]
    #     features = np.zeros((len(Bins) * len(elecomb) * len(elecomb), 8), float)

    #     AtomCA = None
    #     for iAtom in self.Atoms:
    #         v_type = iAtom.verboseType.strip() if hasattr(iAtom, 'verboseType') else ""
    #         if v_type == "CA" and iAtom.ResID == self.ResID:
    #             AtomCA = iAtom
    #             break
        
    #     if AtomCA is None: return features.flatten()
    #     res_num = self.ResID

    #     for Ip, ele1 in enumerate(elecomb):
    #         for Is, ele2 in enumerate(elecomb):
                
    #             pts = []
    #             charges = []
    #             sheaf_values = []
    #             # Note: We do NOT use pt_group/bipartite logic for Alpha complex geometry
                
    #             for iAtom in self.Atoms:
    #                 if iAtom.pos is None: continue 
    #                 clean_type = iAtom.atype.replace(' ', '')
    #                 is_ele1_env = (clean_type == ele1 and iAtom.ResID != res_num)
    #                 is_ele2_mut = (clean_type == ele2 and iAtom.ResID == res_num)
                    
    #                 if is_ele1_env or is_ele2_mut:
    #                     # We use largest bin for cutoff to build complex
    #                     dis = np.linalg.norm(iAtom.pos - AtomCA.pos)
    #                     if dis <= max(Bins): 
    #                         pts.append(list(iAtom.pos))
    #                         charges.append(iAtom.Charge)
                            
    #                         # Flexible value assignment
    #                         val = self.get_atom_value(iAtom)
    #                         sheaf_values.append(val)

    #             num_pts = len(pts)
    #             if num_pts < 3: continue # Need at least 3 points for triangles

    #             # Build Alpha Complex
    #             # AlphaComplex filtration values are alpha^2
    #             ac = gudhi.AlphaComplex(points=pts)
    #             st = ac.create_simplex_tree()
                
    #             for idx_cut, cut in enumerate(Bins):
    #                 idx_feat = idx_cut * 9 + Ip * 3 + Is
                    
    #                 try:
    #                     # Alpha filtration is squared
    #                     L1_matrix = self.compute_alpha_L1_matrix(st, cut**2, sheaf_values, pts, num_pts)
                        
    #                     if L1_matrix is not None and L1_matrix.shape[0] > 0:
    #                         eigens = np.linalg.eigvalsh(L1_matrix)
    #                         valid_eigens = eigens[eigens > 1e-8]
                            
    #                         if len(valid_eigens) > 0:
    #                             if self.mode =='en':
    #                                 features[idx_feat][0] = np.log1p(valid_eigens.sum())#valid_eigens.sum()
    #                                 features[idx_feat][1] = valid_eigens.min()#valid_eigens.min()
    #                                 features[idx_feat][2] = np.log1p(valid_eigens.max())#valid_eigens.max()
    #                                 features[idx_feat][3] = np.log1p(valid_eigens.mean())#valid_eigens.mean()
    #                                 features[idx_feat][4] = np.log1p(valid_eigens.std())#valid_eigens.std()
    #                                 features[idx_feat][5] = np.log1p(valid_eigens.var())#valid_eigens.var()
    #                                 features[idx_feat][6] = np.log1p(np.dot(valid_eigens, valid_eigens))#np.dot(valid_eigens, valid_eigens)
    #                                 features[idx_feat][7] = len(valid_eigens)
    #                             else:
    #                                 features[idx_feat][0] = valid_eigens.sum()
    #                                 features[idx_feat][1] = valid_eigens.min()
    #                                 features[idx_feat][2] = valid_eigens.max()
    #                                 features[idx_feat][3] = valid_eigens.mean()
    #                                 features[idx_feat][4] = valid_eigens.std()
    #                                 features[idx_feat][5] = valid_eigens.var()
    #                                 features[idx_feat][6] = np.dot(valid_eigens, valid_eigens)#np.dot(valid_eigens, valid_eigens)
    #                                 features[idx_feat][7] = len(valid_eigens)
    #                 except Exception:
    #                     pass
        
    #     return features.flatten()
    def alpha_complex_sheaf_spectra(self):
        """
        Computes 1-Dim Sheaf Laplacian features using Alpha Complex.
        Can be concatenated with Rips features.
        """
        elecomb = ['C', 'N', 'O']
        Bins = [3, 4, 5, 6, 7, 8, 9]
        
        # CHANGE 1: Increase feature dimension from 8 to 9
        # 9 combinations * number of bins * 9 statistical features
        features = np.zeros((len(Bins) * len(elecomb) * len(elecomb), 9), float)

        AtomCA = None
        for iAtom in self.Atoms:
            v_type = iAtom.verboseType.strip() if hasattr(iAtom, 'verboseType') else ""
            if v_type == "CA" and iAtom.ResID == self.ResID:
                AtomCA = iAtom
                break
        
        if AtomCA is None: return features.flatten()
        res_num = self.ResID

        for Ip, ele1 in enumerate(elecomb):
            for Is, ele2 in enumerate(elecomb):
                
                pts = []
                charges = []
                sheaf_values = []
                
                for iAtom in self.Atoms:
                    if iAtom.pos is None: continue 
                    clean_type = iAtom.atype.replace(' ', '')
                    is_ele1_env = (clean_type == ele1 and iAtom.ResID != res_num)
                    is_ele2_mut = (clean_type == ele2 and iAtom.ResID == res_num)
                    
                    if is_ele1_env or is_ele2_mut:
                        # We use largest bin for cutoff to build complex
                        dis = np.linalg.norm(iAtom.pos - AtomCA.pos)
                        if dis <= max(Bins): 
                            pts.append(list(iAtom.pos))
                            charges.append(iAtom.Charge)
                            val = self.get_atom_value(iAtom)
                            sheaf_values.append(val)

                num_pts = len(pts)
                if num_pts < 3: continue 

                # Build Alpha Complex
                ac = gudhi.AlphaComplex(points=pts)
                st = ac.create_simplex_tree()
                
                for idx_cut, cut in enumerate(Bins):
                    idx_feat = idx_cut * 9 + Ip * 3 + Is
                    
                    try:
                        # Alpha filtration is squared
                        L1_matrix = self.compute_alpha_L1_matrix(st, cut**2, sheaf_values, pts, num_pts)
                        
                        if L1_matrix is not None and L1_matrix.shape[0] > 0:
                            eigens = np.linalg.eigvalsh(L1_matrix)
                            
                            # CHANGE 2: Calculate harmonic (zero) count
                            valid_eigens = eigens[eigens > 1e-8]
                            num_zeros = len(eigens) - len(valid_eigens)
                            
                            if len(valid_eigens) > 0:
                                if self.mode =='en':
                                    features[idx_feat][0] = np.log1p(valid_eigens.sum())
                                    features[idx_feat][1] = valid_eigens.min()
                                    features[idx_feat][2] = np.log1p(valid_eigens.max())
                                    features[idx_feat][3] = np.log1p(valid_eigens.mean())
                                    features[idx_feat][4] = np.log1p(valid_eigens.std())
                                    features[idx_feat][5] = np.log1p(valid_eigens.var())
                                    features[idx_feat][6] = np.log1p(np.dot(valid_eigens, valid_eigens))
                                    features[idx_feat][7] = len(valid_eigens)
                                    
                                    # CHANGE 3: Store harmonic count
                                    features[idx_feat][8] = num_zeros
                                else:
                                    features[idx_feat][0] = valid_eigens.sum()
                                    features[idx_feat][1] = valid_eigens.min()
                                    features[idx_feat][2] = valid_eigens.max()
                                    features[idx_feat][3] = valid_eigens.mean()
                                    features[idx_feat][4] = valid_eigens.std()
                                    features[idx_feat][5] = valid_eigens.var()
                                    features[idx_feat][6] = np.dot(valid_eigens, valid_eigens)
                                    features[idx_feat][7] = len(valid_eigens)
                                    
                                    # CHANGE 3: Store harmonic count
                                    features[idx_feat][8] = num_zeros
                            else:
                                # Case where ALL eigenvalues are zero
                                features[idx_feat][8] = len(eigens)
                    except Exception:
                        pass
        
        return features.flatten()
    def alpha_complex(self):
        ElementList = ['C', 'N', 'O']
        res_num = self.ResID

        alpha_PH12 = np.zeros([3, 3, 14])
        dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

        for Ip, e1 in enumerate(ElementList):
            for Is, e2 in enumerate(ElementList):
                points = []
                for iAtom in self.Atoms:
                    if (iAtom.atype.replace(' ','')==e1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==e2 and iAtom.ResID==res_num):
                        points.append(iAtom.pos)


                alpha_complex = gudhi.AlphaComplex(points=points)
                PH = alpha_complex.create_simplex_tree().persistence()

                tmpbars = np.zeros(len(PH), dtype=dt)
                cnt = 0
                for simplex in PH:
                    dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
                    if d-b < 0.1: continue
                    tmpbars[cnt]['dim']   = dim
                    tmpbars[cnt]['birth'] = b
                    tmpbars[cnt]['death'] = d
                    cnt += 1
                bars = tmpbars[0:cnt]
                if len(bars[bars['dim']==1]['death']) > 0:
                    alpha_PH12[Ip, Is, 0] = np.sum(bars[bars['dim']==1]['death'] - \
                                                        bars[bars['dim']==1]['birth'])
                    alpha_PH12[Ip, Is, 1] = np.max(bars[bars['dim']==1]['death'] - \
                                                        bars[bars['dim']==1]['birth'])
                    alpha_PH12[Ip, Is, 2] = np.mean(bars[bars['dim']==1]['death'] - \
                                                         bars[bars['dim']==1]['birth'])
                    alpha_PH12[Ip, Is, 3] = np.min(bars[bars['dim']==1]['birth'])
                    alpha_PH12[Ip, Is, 4] = np.max(bars[bars['dim']==1]['birth'])
                    alpha_PH12[Ip, Is, 5] = np.min(bars[bars['dim']==1]['death'])
                    alpha_PH12[Ip, Is, 6] = np.max(bars[bars['dim']==1]['death'])
                if len(bars[bars['dim']==2]['death']) > 0:
                    alpha_PH12[Ip, Is, 7]  = np.sum(bars[bars['dim']==2]['death'] - \
                                                         bars[bars['dim']==2]['birth'])
                    alpha_PH12[Ip, Is, 8]  = np.max(bars[bars['dim']==2]['death'] - \
                                                         bars[bars['dim']==2]['birth'])
                    alpha_PH12[Ip, Is, 9]  = np.mean(bars[bars['dim']==2]['death'] - \
                                                          bars[bars['dim']==2]['birth'])
                    alpha_PH12[Ip, Is, 10] = np.min(bars[bars['dim']==2]['birth'])
                    alpha_PH12[Ip, Is, 11] = np.max(bars[bars['dim']==2]['birth'])
                    alpha_PH12[Ip, Is, 12] = np.min(bars[bars['dim']==2]['death'])
                    alpha_PH12[Ip, Is, 13] = np.max(bars[bars['dim']==2]['death'])

        alpha_PH12_all = np.zeros([14])
        points = []
        for iAtom in self.Atoms:
            if iAtom.atype.replace(' ', '') != 'H':
                points.append(iAtom.pos)
        alpha_complex = gudhi.AlphaComplex(points=points)
        PH = alpha_complex.create_simplex_tree().persistence()

        tmpbars = np.zeros(len(PH), dtype=dt)
        cnt = 0
        for simplex in PH:
            dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
            if d-b < 0.1: continue
            tmpbars[cnt]['dim']   = dim
            tmpbars[cnt]['birth'] = b
            tmpbars[cnt]['death'] = d
            cnt += 1
        bars = tmpbars[0:cnt]
        if len(bars[bars['dim']==1]['death']) > 0:
            alpha_PH12_all[0] = np.sum(bars[bars['dim']==1]['death'] - \
                                            bars[bars['dim']==1]['birth'])
            alpha_PH12_all[1] = np.max(bars[bars['dim']==1]['death'] - \
                                            bars[bars['dim']==1]['birth'])
            alpha_PH12_all[2] = np.mean(bars[bars['dim']==1]['death'] - \
                                             bars[bars['dim']==1]['birth'])
            alpha_PH12_all[3] = np.min(bars[bars['dim']==1]['birth'])
            alpha_PH12_all[4] = np.max(bars[bars['dim']==1]['birth'])
            alpha_PH12_all[5] = np.min(bars[bars['dim']==1]['death'])
            alpha_PH12_all[6] = np.max(bars[bars['dim']==1]['death'])
        if len(bars[bars['dim']==2]['death']) > 0:
            alpha_PH12_all[7]  = np.sum(bars[bars['dim']==2]['death'] - \
                                             bars[bars['dim']==2]['birth'])
            alpha_PH12_all[8]  = np.max(bars[bars['dim']==2]['death'] - \
                                             bars[bars['dim']==2]['birth'])
            alpha_PH12_all[9]  = np.mean(bars[bars['dim']==2]['death'] - \
                                              bars[bars['dim']==2]['birth'])
            alpha_PH12_all[10] = np.min(bars[bars['dim']==2]['birth'])
            alpha_PH12_all[11] = np.max(bars[bars['dim']==2]['birth'])
            alpha_PH12_all[12] = np.min(bars[bars['dim']==2]['death'])
            alpha_PH12_all[13] = np.max(bars[bars['dim']==2]['death'])

        return np.array(alpha_PH12), np.array(alpha_PH12_all)

    def alpha_complex_sr(self):

        print('Calculating SR-Alpha >>>>>>>>>>>>>>>>>>>>>>>>>>>')
        ElementList = ['C', 'N', 'O']
        res_num = self.ResID

        alpha_SR12_curves = np.zeros([3, 3, 20])
        alpha_SR12_rates = np.zeros([3, 3, 20])
        dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

        for Ip, e1 in enumerate(ElementList):
            for Is, e2 in enumerate(ElementList):
                points = []
                for iAtom in self.Atoms:
                    if (iAtom.atype.replace(' ','')==e1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==e2 and iAtom.ResID==res_num):
                        points.append(iAtom.pos)

                model = alpha_facet(points, max_dim=1, min_edge_length=1, max_edge_length=10, num_samples=10)
                alpha_SR12_curves[Ip, Is] = np.hstack(model.facet_curves())
                alpha_SR12_rates[Ip, Is] = np.hstack(model.facet_rates())

        points = []
        for iAtom in self.Atoms:
            if iAtom.atype.replace(' ', '') != 'H':
                points.append(iAtom.pos)
        
        model = alpha_facet(points, max_dim=1, min_edge_length=1, max_edge_length=10, num_samples=10)
        alpha_SR12all_curves = np.hstack(model.facet_curves())
        alpha_SR12all_rates = np.hstack(model.facet_rates())

        alpha_SR_curves = np.concatenate((alpha_SR12_curves.flatten(), alpha_SR12all_curves), axis=0)
        alpha_SR_rates = np.concatenate((alpha_SR12_rates.flatten(), alpha_SR12all_rates), axis=0)
        return np.array(alpha_SR_curves), np.array(alpha_SR_rates)

    def alpha_complex_fvector(self):

        print('Calculating SR-Alpha >>>>>>>>>>>>>>>>>>>>>>>>>>>')
        ElementList = ['C', 'N', 'O']
        res_num = self.ResID

        alpha_SR12_curves = np.zeros([3, 3, 20])
        alpha_SR12_rates = np.zeros([3, 3, 20])
        dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

        for Ip, e1 in enumerate(ElementList):
            for Is, e2 in enumerate(ElementList):
                points = []
                for iAtom in self.Atoms:
                    if (iAtom.atype.replace(' ','')==e1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==e2 and iAtom.ResID==res_num):
                        points.append(iAtom.pos)

                model = alpha_fvector(points, max_dim=1, min_edge_length=1, max_edge_length=10, num_instances=10)
                alpha_SR12_curves[Ip, Is] = np.hstack(list(model.compute_f_vector_curves().values())) # the f-vector curves components stacked
                alpha_SR12_rates[Ip, Is] = np.hstack(list(model.compute_rate_curves().values())) # the average of the radius f/t

        points = []
        for iAtom in self.Atoms:
            if iAtom.atype.replace(' ', '') != 'H':
                points.append(iAtom.pos)
        
        model = alpha_fvector(points, max_dim=1, min_edge_length=1, max_edge_length=10, num_instances=10)
        alpha_SR12all_curves = np.hstack(list(model.compute_f_vector_curves().values()))
        alpha_SR12all_rates = np.hstack(list(model.compute_rate_curves().values()))

        alpha_SR_curves = np.concatenate((alpha_SR12_curves.flatten(), alpha_SR12all_curves), axis=0)
        alpha_SR_rates = np.concatenate((alpha_SR12_rates.flatten(), alpha_SR12all_rates), axis=0)
        return np.array(alpha_SR_curves), np.array(alpha_SR_rates)

    def FRI_dists(self):
        # get the list of all element-specific interactions distance
        Atoms = self.Atoms
        for iAtom in Atoms:
            if iAtom.verboseType.replace(' ', '')=="CA" and iAtom.ResID==self.ResID:
                AtomCA = iAtom
        res_num = self.ResID

        FRI_dists = []
        for Ip, ele1 in enumerate(ElementList):
            for Is, ele2 in enumerate(ElementList):
                pts = []; ptid = [] # ptid to indicate if atom belongs to mutated residue id
                for iAtom in Atoms:
                    if (iAtom.atype.replace(' ','')==ele1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==ele2 and iAtom.ResID==res_num):
                        dis = np.linalg.norm(iAtom.pos-AtomCA.pos)
                        if dis <= 16:
                            pts.append(iAtom.pos)
                            if iAtom.ResID==res_num:
                                ptid.append(0)
                            else:
                                ptid.append(1)
                dists = []
                for ii in range(len(pts)):
                    for jj in range(len(pts)):
                        if ptid[ii]+ptid[jj]==1:
                            dists.append(np.linalg.norm(pts[ii]-pts[jj]))
                dists = np.array(dists)
                FRI_dists.append(dists)
        return FRI_dists

    def FRI_Lorentz(self, dists, kappa, eta):
        Feature_FRI = np.zeros([EleLength, EleLength])
        for idx_m, e_m in enumerate(ElementList):
            for idx_o, e_o in enumerate(ElementList):
                if len(dists[idx_m*EleLength+idx_o]) > 0:
                    eta_i = eta[idx_m] + eta[idx_o]
                    Lorentz_array = np.true_divide(dists[idx_m*EleLength+idx_o], eta_i)
                    Lorentz_array = 1./(1.+np.power(Lorentz_array, kappa))
                    Feature_FRI[idx_m, idx_o] = np.sum(Lorentz_array)
                else:
                    Feature_FRI[idx_m, idx_o] = 0
        return np.around(Feature_FRI, 8).flatten()

    def FRI_Exp(self, dists, kappa, eta):
        Feature_FRI = np.zeros([EleLength, EleLength])
        for idx_m, e_m in enumerate(ElementList):
            for idx_o, e_o in enumerate(ElementList):
                if len(dists[idx_m*EleLength+idx_o]) > 0:
                    eta_i = eta[idx_m] + eta[idx_o]
                    Exp_array = np.true_divide(dists[idx_m*EleLength+idx_o], eta_i)
                    Exp_array = np.exp(-np.power(Exp_array, kappa))
                    Feature_FRI[idx_m, idx_o] = np.sum(Exp_array)
                else:
                    Feature_FRI[idx_m, idx_o] = 0
        return np.around(Feature_FRI, 8).flatten()

    def FRI_Lorentz_(self, dists, kappa, eta):
        Feature_FRI = np.zeros([EleLength, EleLength, 4])
        for idx_m, e_m in enumerate(ElementList):
            for idx_o, e_o in enumerate(ElementList):
                if len(dists[idx_m*EleLength+idx_o]) > 0:
                    eta_i = eta[idx_m] + eta[idx_o]
                    Lorentz_array = np.true_divide(dists[idx_m*EleLength+idx_o], eta_i)
                    Lorentz_array = 1./(1.+np.power(Lorentz_array, kappa))
                    Feature_FRI[idx_m, idx_o, 0] = np.sum(Lorentz_array)
                    Feature_FRI[idx_m, idx_o, 1] = np.min(Lorentz_array)
                    Feature_FRI[idx_m, idx_o, 2] = np.max(Lorentz_array)
                    Feature_FRI[idx_m, idx_o, 3] = np.mean(Lorentz_array)
                else:
                    Feature_FRI[idx_m, idx_o, 0] = 0
                    Feature_FRI[idx_m, idx_o, 1] = 0
                    Feature_FRI[idx_m, idx_o, 2] = 0
                    Feature_FRI[idx_m, idx_o, 3] = 0
        return np.around(Feature_FRI, 8).flatten()

    def FRI_Exp_(self, dists, kappa, eta):
        Feature_FRI = np.zeros([EleLength, EleLength, 4])
        for idx_m, e_m in enumerate(ElementList):
            for idx_o, e_o in enumerate(ElementList):
                if len(dists[idx_m*EleLength+idx_o]) > 0:
                    eta_i = eta[idx_m] + eta[idx_o]
                    Exp_array = np.true_divide(dists[idx_m*EleLength+idx_o], eta_i)
                    Exp_array = np.exp(-np.power(Exp_array, kappa))
                    Feature_FRI[idx_m, idx_o, 0] = np.sum(Exp_array)
                    Feature_FRI[idx_m, idx_o, 1] = np.min(Exp_array)
                    Feature_FRI[idx_m, idx_o, 2] = np.max(Exp_array)
                    Feature_FRI[idx_m, idx_o, 3] = np.mean(Exp_array)
                else:
                    Feature_FRI[idx_m, idx_o, 0] = 0
                    Feature_FRI[idx_m, idx_o, 1] = 0
                    Feature_FRI[idx_m, idx_o, 2] = 0
                    Feature_FRI[idx_m, idx_o, 3] = 0
        return np.around(Feature_FRI, 8).flatten()

    def flexibility_rigidy_index(self):
        Feature_FRI = np.zeros([EleLength, EleLength, 6])
        Atoms = self.Atoms
        for iAtom in Atoms:
            if iAtom.verboseType.replace(' ', '')=="CA" and iAtom.ResID==self.ResID:
                AtomCA = iAtom
        res_num = self.ResID

        for Ip, ele1 in enumerate(elecomb):
            for Is, ele2 in enumerate(elecomb):
                pts = []; ptid = [] # ptid to indicate if atom belongs to mutated residue id
                for iAtom in Atoms:
                    if (iAtom.atype.replace(' ','')==ele1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==ele2 and iAtom.ResID==res_num):
                        dis = np.linalg.norm(iAtom.pos-AtomCA.pos)
                        if dis <= cutoff:
                            pts.append(iAtom.pos)
                            if iAtom.ResID==res_num:
                                ptid.append(0)
                            else:
                                ptid.append(1)
                dists = []
                for ii in range(len(pts)):
                    for jj in range(len(pts)):
                        if ptid[ii]+ptid[jj]==1:
                            dists.append(np.linalg.norm(pts[ii]-pts[jj]))
                dists  = np.array(dists)
                dists0 = np.true_divide(dists, (ElementTau[idx_m]+ElementTau[idx_o])*0.5)
                dists1 = np.true_divide(dists, (ElementTau[idx_m]+ElementTau[idx_o]))
                dists2 = np.true_divide(dists, (ElementTau[idx_m]+ElementTau[idx_o])*2)
                # kernel exponential 0
                dists_exp0 = np.exp(-np.power(dists1, 2))
                Feature_FRI[idx_m, idx_o, 0] = np.sum(dists_exp0)
                # kernel exponential 1
                dists_exp1 = np.exp(-np.power(dists1, 15))
                Feature_FRI[idx_m, idx_o, 1] = np.sum(dists_exp1)
                # kernel exponential 2
                dists_exp2 = np.exp(-np.power(dists2, 15))
                Feature_FRI[idx_m, idx_o, 2] = np.sum(dists_exp2)
                # kernel lorentz 0
                dists_Lor0 = 1./(1.+np.power(dists0, 5))
                Feature_FRI[idx_m, idx_o, 3] = np.sum(dists_Lor0)
                # kernel lorentz 1
                dists_Lor1 = 1./(1.+np.power(dists1, 5))
                Feature_FRI[idx_m, idx_o, 4] = np.sum(dists_Lor1)
                # kernel lorentz 2
                dists_Lor2 = 1./(1.+np.power(dists2, 5))
                Feature_FRI[idx_m, idx_o, 5] = np.sum(dists_Lor2)

        return np.around(Feature_FRI, 8)

    def construct_feature_global(self):
        print('construct features >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        IndexArray = [np.array([0], int),
                      np.array([1], int),
                      np.array([2], int),
                      np.array([3], int),
                      np.array([4], int),
                      np.array([0,1,2,3], int),
                      np.array([0,1,2,3,4], int)]
        FeatureGLB = []
        # Charge
        for i in range(3):
            for j in range(7):
                FeatureGLB.append(np.sum(self.Charge[self.IndexList[i][j]]))
                FeatureGLB.append(np.sum(np.abs(self.Charge[self.IndexList[i][j]])))
        # RIG
        for i in range(3):
            for j in [0,1,2,3,5]:
                FeatureGLB.append(np.sum(self.FRI1[self.IndexList[i][j],:][:,IndexArray[5]]))
        for i in range(3):
            for j in [0,1,2,3,5]:
                FeatureGLB.append(np.sum(self.FRI2[self.IndexList[i][j],:][:,IndexArray[5]]))
        for i in range(3):
            for j in [0,1,2,3,5]:
                FeatureGLB.append(np.sum(self.FRI3[self.IndexList[i][j],:][:,IndexArray[5]]))
        for i in range(3):
            for j in [0,1,2,3,5]:
                FeatureGLB.append(np.sum(self.FRI4[self.IndexList[i][j],:][:,IndexArray[5]]))
        for i in range(3):
            for j in [0,1,2,3,5]:
                FeatureGLB.append(np.sum(self.FRI5[self.IndexList[i][j],:][:,IndexArray[5]]))
        # VDW
        for i in range(3):
            for j in [0,1,2,3,5]:
                FeatureGLB.append(np.sum(self.VDW[self.IndexList[i][j],:][:,IndexArray[5]]))
        # CLB
        for i in range(3):
            for j in [0,1,2,3,5]:
                FeatureGLB.append(np.sum(self.CLB[self.IndexList[i][j]][:,IndexArray[6]]))
                FeatureGLB.append(np.sum(np.abs(self.CLB[self.IndexList[i][j]][:,IndexArray[6]])))
        self.FeatureGLB = FeatureGLB
        FeatureGLBother = []
        # Other
        AA = self.ResName
        for Group in Groups:
            if AA in Group:
                FeatureGLBother.append(1.0)
            else:
                FeatureGLBother.append(0.0)
        FeatureGLBother.append(AAvolume[AA])
        FeatureGLBother.append(AAhydropathy[AA])
        FeatureGLBother.append(AAarea[AA])
        FeatureGLBother.append(AAweight[AA])
        FeatureGLBother.append(AAcharge(AA))
        FeatureGLBother.extend(AApharma[AA])
        self.FeatureGLBother = FeatureGLBother

        return FeatureGLB, FeatureGLBother # construct_feature_global()

    def construct_feature_env(self):
        print('construct environment feature >>>>>>>>>>>>>>>>>>>>>')
        FeatureEnv = []
        NearSeq = []
        CurResID = -1000
        for i in self.IndexList[1][6]:
            ResID = self.Atoms[i].ResID
            if self.Atoms[i].ResID!=CurResID:
                CurResID = ResID
                NearSeq.append(self.Atoms[i].ResName)
        for Group in Groups:
            cnt = 0.
            for AA in NearSeq:
                if AA in Group:
                    cnt += 1.
            FeatureEnv.append(cnt)
            FeatureEnv.append(cnt/max(1., float(len(NearSeq))))
        Vol = []; Hyd = []; Area = []; Wgt = []; Chg = []
        phara = [0, 0, 0, 0, 0, 0]
        for AA in NearSeq:
            Vol.append(AAvolume[AA])
            Hyd.append(AAhydropathy[AA])
            Area.append(AAarea[AA])
            Wgt.append(AAweight[AA])
            Chg.append(AAcharge(AA))
            for i in range(6):
                phara[i] += AApharma[AA][i]
        Vol = np.asarray(Vol)
        Hyd = np.asarray(Hyd)
        Area = np.asarray(Area)
        Wgt = np.asarray(Wgt)

        if len(NearSeq) == 0:
            FeatureEnv.extend([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        else:
            FeatureEnv.extend([np.sum(Vol), np.sum(Vol)/float(len(NearSeq)), np.var(Vol)])
            FeatureEnv.extend([np.sum(Hyd), np.sum(Hyd)/float(len(NearSeq)), np.var(Hyd)])
            FeatureEnv.extend([np.sum(Area), np.sum(Area)/float(len(NearSeq)), np.var(Area)])
            FeatureEnv.extend([np.sum(Wgt), np.sum(Wgt)/float(len(NearSeq)), np.var(Wgt)])
        FeatureEnv.append(sum(Chg))
        FeatureEnv.extend(phara)

        self.FeatureEnv = FeatureEnv
        return FeatureEnv

    def construct_feature_MIBPB(self, h=0.5):
        Area = []
        SolvEng = []
        print('run MIBPB >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        if not os.path.exists(self.filename+'.englist') or \
           not os.path.exists(self.filename+'.eng') or \
           not os.path.exists(self.filename+'.arealist') or \
           not os.path.exists(self.filename+'.areavolume'):
            os.system('mibpb5 '+self.filename+' h=%f'%(h))
            os.system('mv partition_area.txt '+self.filename+'.arealist')
            os.system('mv area_volume.dat '+self.filename+'.areavolume')
        os.system('rm -f bounding_box.txt')
        os.system('rm -f grid_info.txt')
        os.system('rm -f intersection_info.txt')
        os.system('rm -f '+self.filename+'.dx')
        # Info from arealist file
        AreaListFile = open(self.filename+'.arealist')
        for idx, line in enumerate(AreaListFile):
            a, b = line.split()
            Area.append(float(b))
        AreaListFile.close()
        # Info from englist file
        EngListFile = open(self.filename+'.englist')
        for idx, line in enumerate(EngListFile):
            SolvEng.append(float(line))
        EngListFile.close()
        Area = np.array(Area, float)
        SolvEng = np.array(SolvEng, float)

        # Info from areavolume file
        AreaVolumeFile = open(self.filename+'.areavolume')
        TotalArea = float(AreaVolumeFile.readline())
        TotalVolume = float(AreaVolumeFile.readline())
        AreaVolumeFile.close()
        # Info from eng file
        EngFile = open(self.filename+'.eng')
        EngFile.readline()
        TotalSolvEng = float(EngFile.readline())
        EngFile.close()

        FeatureMIBPB = []
        # SolvEng from MIBPB
        for i in range(3):
            for j in range(7):
                FeatureMIBPB.append(np.sum(SolvEng[self.IndexList[i][j]]))
        # Area from MIBPB
        for i in range(3):
            for j in range(7):
                FeatureMIBPB.append(np.sum(Area[self.IndexList[i][j]]))
        self.FeatureMIBPB = FeatureMIBPB

        FeatureMIBPBglb = []
        # Global
        FeatureMIBPBglb.append(TotalSolvEng)
        FeatureMIBPBglb.append(TotalArea)
        FeatureMIBPBglb.append(TotalVolume)
        self.FeatureMIBPBglb = FeatureMIBPBglb

        return FeatureMIBPB, FeatureMIBPBglb # construct_feature_MIBPB

    def runMIBPB(self, h=0.5):
        print('run MIBPB >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        if not os.path.exists(self.filename+'.englist') or \
           not os.path.exists(self.filename+'.eng') or \
           not os.path.exists(self.filename+'.arealist') or \
           not os.path.exists(self.filename+'.areavolume'):
            os.system('mibpb5 '+self.filename+' h=%f'%(h))
            os.system('mv partition_area.txt '+self.filename+'.arealist')
            os.system('mv area_volume.dat '+self.filename+'.areavolume')
        return



    def runBLAST(self):
        print('run BLAST >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        cline = NcbipsiblastCommandline(query=self.filename_single+'.fasta',
                                        # db='/mnt/research/common-data/Bio/blastdb/nr',
                                        # db='/mnt/home/renyimi2/blastdb/uniref50/uniref50',
                                        # db='/mnt/research/common-data/Bio/blast_databases/blastdb_current/nr',
                                        db='/mnt/scratch/renyimi2/blastdb/uniref50',
                                        num_iterations=3,
                                        evalue=5000,
                                        out=self.filename_single+'.out',
                                        out_ascii_pssm=self.filename_single+'.pssm')

        if not os.path.exists(self.filename_single+'.pssm'):
            print('running '+self.filename_single+'.pssm')
            stdout, stderr = cline()
            print(stdout, stderr)
        else:
            flag = True
            fp = open(self.filename_single+'.pssm')
            for line in fp:
                if line[:10]=='PSI Gapped':
                    flag = False
            fp.close()
            if flag:
                print('running '+self.filename_single+'.pssm')
                stdout, stderr = cline()
                print(stdout, stderr)
        return # runBLAST

    def construct_feature_seq(self):
        print('generate secondary structure information >>>>>>>>>>')
        FeatureSeq = []

        # Structure based DSSP
        #######################
        # PDB files for DSSP (version 4 or later) requires HEADER and CRYST1 line 
        # Otherwise DSSP will default to MMCIF 
        # DSSP still have problems with MMCIF ??? 
        parser = PDBParser()
        structure = parser.get_structure(self.PDBid, self.filename+'.pdb')
        model = structure[0]

        io = PDBIO() 
        io.set_structure(structure)
        io.save(self.filename+'.pdb1')

        file = open(self.filename+'.pdb1', 'r')
        contents = file.readlines()
        new_file = open(self.filename+'.pdb2', 'w+')
        new_file.write('HEADER\n')
        new_file.write('CRYST1\n')
        for i in range(len(contents)):
            #print(contents[i])
            new_file.write(contents[i])
            if "ENDMDL" in contents[i]:
                break
        new_file.close()

        parser = PDBParser()
        structure = parser.get_structure(self.PDBid, self.filename+'.pdb2')
        model = structure[0]

        # dssp = DSSP(model, self.filename+'.pdb2', dssp='mkdssp', file_type='PDB')
        # print(dssp[(self.Chain, (' ', self.ResID, ' '))])
        # ssindex = ss2index[dssp[(self.Chain, (' ', self.ResID, ' '))][2]]
        # FeatureSeq.append(ssindex)
        ####yiming
        pdb_file = self.filename + ".pdb2"
        dssp_file = self.filename + ".dssp"

        print(f"[Info] Running mkdssp on {pdb_file}")
        subprocess.run(["mkdssp", pdb_file, dssp_file], check=True)

        # --- parse DSSP output ---
        dssp, dssp_keys = make_dssp_dict(dssp_file)


        # --- your original logic ---
        key = (self.Chain, (' ', self.ResID, ' '))
        print("DSSP entry for key", key, ":", dssp[key])
        if key in dssp:
           ss_code = dssp[key][1]
           ssindex = ss2index.get(ss_code, 0)
        else:
            print(f"[Warning] residue {self.ResID} chain {self.Chain} missing in DSSP output")
            ssindex = 0
        FeatureSeq.append(ssindex)
        print('dssp finished')
       
        #os.system(f'rm {self.filename}.pdb1 {self.filename}.pdb2')
        # Sequence based Spider3
        # if not os.path.exists('../../../bin/SPIDER2_local/misc/pred_pssm.py'):
        #     sys.exit('Please make sure the SPIDER2_local exists')
        # os.system('../../../bin/SPIDER2_local/misc/pred_pssm.py '+self.filename_single+'.pssm -f')
        if not os.path.exists('/mnt/home/renyimi2/CANet/bin/SPIDER2_local/misc/pred_pssm.py'):
            sys.exit('Please make sure the SPIDER2_local exists')
        os.system('/mnt/home/renyimi2/CANet/bin/SPIDER2_local/misc/pred_pssm.py '+self.filename_single+'.pssm -f')
        spdfile = open(self.filename_single+'.spd3')
        #spdfile.readline()
        #for line in spdfile:
        #    if int(line.split()[0]) == self.ResIDSeq:
        #        break
        lines = spdfile.read().splitlines()
        line = lines[self.ResIDSeq+1] # Here, num+1 is because of the first line of spd3 is header
        if line.split()[1] != self.ResName:
            print(self.ResIDSeq, line.split()[1], self.ResName)
            print(self.filename_single+'.fasta is removed')
            print(self.filename_single+'.pssm is removed')
            #os.system('rm '+self.filename_single+'.fasta')
            #os.system('rm '+self.filename_single+'.pssm')
            sys.exit('Wrong residue when calling pssm for '+self.typeFlag)
        psi=0.; phi=0.; pc=0.; pe=0.; ph=0.
        d0, d1, d2, d3, phi, psi, d4, d5, pc, pe, ph = line.split()
        spdfile.close()
        FeatureSeq.extend([float(phi), float(psi), float(pc), float(pe), float(ph)])
        self.FeatureSeq = FeatureSeq

        #np.save(f'{self.filename}_dssp.npy', FeatureSeq)
        return self.FeatureSeq # secondary_structure
        #return 

def construct_features_PH0(b_dth_WT, b_bar_WT, b_dth_MT, b_bar_MT):
    Feature_ph0 = np.concatenate((b_dth_MT.flatten(), b_dth_WT.flatten()))
    Feature_ph0 = np.concatenate((Feature_ph0.flatten(), b_dth_MT.flatten()-b_dth_WT.flatten()))

    Feature_ph0 = np.concatenate((Feature_ph0.flatten(), b_bar_MT.flatten()))
    Feature_ph0 = np.concatenate((Feature_ph0.flatten(), b_bar_WT.flatten()))
    Feature_ph0 = np.concatenate((Feature_ph0.flatten(), b_bar_MT.flatten()-b_bar_WT.flatten()))

    return Feature_ph0


def construct_features_PH0_cnn(b_dth_WT, b_bar_WT, b_dth_MT, b_bar_MT):
    Feature_ph0 = np.concatenate((b_dth_MT, b_dth_WT), axis=1)
    Feature_ph0 = np.concatenate((Feature_ph0, b_dth_MT-b_dth_WT), axis=1)

    Feature_ph0 = np.concatenate((Feature_ph0, b_bar_MT), axis=1)
    Feature_ph0 = np.concatenate((Feature_ph0, b_bar_WT), axis=1)
    Feature_ph0 = np.concatenate((Feature_ph0, b_bar_MT-b_bar_WT), axis=1)

    return Feature_ph0

def construct_features_PH12(b_alpha_WT, b_alpha_all_WT, b_alpha_MT, b_alpha_all_MT):
    Feature_ph12 = np.concatenate((b_alpha_MT.flatten(), b_alpha_WT.flatten()))
    Feature_ph12 = np.concatenate((Feature_ph12, b_alpha_MT.flatten()-b_alpha_WT.flatten()))

    Feature_ph12 = np.concatenate((Feature_ph12, b_alpha_all_MT.flatten()))
    Feature_ph12 = np.concatenate((Feature_ph12, b_alpha_all_WT.flatten()))
    Feature_ph12 = np.concatenate((Feature_ph12, b_alpha_all_MT.flatten()-b_alpha_all_WT.flatten()))

    return Feature_ph12

def construct_features_FRI(c_WT,          c_MT,
                           c_WT_dist_b,   c_MT_dist_b):
    def feature_concatenate(c_WT_b, c_MT_b):
        feature = np.concatenate((c_MT_b, c_WT_b))
        feature = np.concatenate((feature, c_MT_b-c_WT_b))
        return feature

    kappa = 2; ElementTau = [5, 2*1.01, 2]
    c_WT_FRI_Exp_b   = c_WT.FRI_Exp_(c_WT_dist_b,   kappa, ElementTau)
    c_MT_FRI_Exp_b   = c_MT.FRI_Exp_(c_MT_dist_b,   kappa, ElementTau)
    feature_Exp = feature_concatenate(c_WT_FRI_Exp_b,   c_MT_FRI_Exp_b)

    kappa = 4; ElementTau = [1, 6*1.01, 6]
    c_WT_FRI_Lorentz_b   = c_WT.FRI_Lorentz_(c_WT_dist_b,   kappa, ElementTau)
    c_MT_FRI_Lorentz_b   = c_MT.FRI_Lorentz_(c_MT_dist_b,   kappa, ElementTau)
    feature_Lorentz = feature_concatenate(c_WT_FRI_Lorentz_b,   c_MT_FRI_Lorentz_b)

    Feature_FRI = np.concatenate((feature_Exp, feature_Lorentz))
    return Feature_FRI # construct_features_FRI_

def restriction_formula(simplex: list, coface: list, sst) -> float:
    """
    Computes the restriction map value between a simplex and its coface.
    Based on the snippet provided:
    - For Vertex -> Edge: related to Charge / Distance
    - For Edge -> Triangle: related to Charge
    """
    # Case 1: Restricting from Vertex (1 node) to Edge (2 nodes)
    if len(simplex) == 1:
        # Identify the "other" node in the edge (the sibling)
        if simplex == [coface[0]]:
            sibling = [coface[1]]
        else:
            sibling = [coface[0]]
        print('coface',coface)
        
        # Extract coordinates (indices 0-2) and data (index 3) from extra_data
        # Note: We assume extra_data stores [x, y, z, charge]
        coords_simplex = sst.extra_data[tuple(simplex)][0:3]
        coords_sibling = sst.extra_data[tuple(sibling)][0:3]
        
        # Euclidean distance
        distance = sqrt((coords_simplex[0] - coords_sibling[0])**2 \
                      + (coords_simplex[1] - coords_sibling[1])**2 \
                      + (coords_simplex[2] - coords_sibling[2])**2)
        
        # Avoid division by zero
        if distance < 1e-8: 
            return 0.0
            
        # Return Charge_of_Sibling / Distance
        return sst.extra_data[tuple(sibling)][3] / distance

    # Case 2: Restricting from Edge to Triangle
    elif len(simplex) == 2:
        coeff = 1.0
        # Iterate through boundaries of the triangle (coface) to find relationships
        for (sibling, _) in sst.st.get_boundaries(coface):
            if sibling == simplex:
                # Find the vertex opposite to this edge
                opposite_vertex_idx = sst.coface_index(simplex, coface)
                opposite_vertex = coface[opposite_vertex_idx]
                
                # Multiply by the charge of the opposite vertex
                coeff = coeff * sst.extra_data[tuple([opposite_vertex])][3] 
            else:
                # Normalize by filtration value of the other edges
                filt = sst.st.filtration(sibling)
                if filt > 1e-8:
                    coeff = coeff / filt
        return coeff



    return 1.0

def construct_feature_aux(p_WT, p_MT, flag_MIBPB=False, flag_BLAST=False):
    if flag_MIBPB:
        p_MTFeature = p_MT.FeatureMIBPB+p_MT.FeatureGLB+p_MT.FeatureMIBPBglb+p_MT.FeatureGLBother
        p_WTFeature = p_WT.FeatureMIBPB+p_WT.FeatureGLB+p_WT.FeatureMIBPBglb+p_WT.FeatureGLBother
    else:
        p_MTFeature = p_MT.FeatureGLB+p_MT.FeatureGLBother
        p_WTFeature = p_WT.FeatureGLB+p_WT.FeatureGLBother
    Feature = p_MTFeature+p_WTFeature
    Feature.extend(map(operator.sub, p_MTFeature, p_WTFeature))

    # pKa features
    pKaIndex = {'ASP':0, 'GLU':1, 'ARG':2, 'LYS':3, 'HIS':4, 'CYS':5, 'TYR':6}
    pKaGroup = ['ASP',   'GLU',   'ARG',   'LYS',   'HIS',   'CYS',   'TYR']
    mutpKa   = np.array(p_MT.pKa, float)
    wildpKa  = np.array(p_WT.pKa, float)
    wildpKaname = p_WT.pKaName
    defer = mutpKa-wildpKa
    absmax = np.max(np.abs(defer))
    abssum = np.sum(np.abs(defer))
    maxpos = np.max(defer)
    maxneg = np.min(defer)
    netchange = np.sum(defer)
    DetailShiftAbs = np.zeros([7], float)
    DetailShiftNet = np.zeros([7], float)
    for j in range(len(wildpKa)):
        if wildpKaname[j] in pKaGroup:
            DetailShiftAbs[pKaIndex[wildpKaname[j]]] += np.abs(mutpKa[j]-wildpKa[j])
            DetailShiftNet[pKaIndex[wildpKaname[j]]] += mutpKa[j]-wildpKa[j]
    mutC = p_MT.pKaCt; mutN = p_MT.pKaNt
    wildC = p_WT.pKaCt; wildN = p_WT.pKaNt
    mutsitepKa = p_MT.pKaSite; wildsitepKa = p_WT.pKaSite;
    Feature.extend([absmax, abssum, maxpos, maxneg, netchange, 
                    wildsitepKa, mutsitepKa, mutsitepKa-wildsitepKa, 
                    wildC, mutC, mutC-wildC, wildN, mutN, mutN-wildN])
    Feature.extend(DetailShiftNet.tolist())
    Feature.extend(DetailShiftAbs.tolist())

    # Environment features
    Feature.extend(p_WT.FeatureEnv)

    if flag_BLAST:
        # PSSM features
        AAind = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10, \
                 'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20}
        resWT = p_WT.ResName
        resMT = p_MT.ResName
        resNum = p_MT.ResIDSeq
        pssm_score1 = np.zeros([p_WT.SeqLength, 20])
        pssm_score2 = np.zeros([p_WT.SeqLength, 20])
        pssm_score3 = np.zeros([p_WT.SeqLength, 2])

        pssmfile = open(p_WT.filename_single+'.pssm')
        lines = pssmfile.read().splitlines()
        for idx, line in enumerate(lines[3:3+p_WT.SeqLength]):
            tmp_vec = line.split()
            pssm_score1[idx, :] = tmp_vec[2:22]
            pssm_score2[idx, :] = tmp_vec[22:42]
            pssm_score3[idx, :] = tmp_vec[42:]
        pssmfile.close()

        Feature.append(pssm_score1[p_MT.ResIDSeq, AAind[resMT]-1])
        Feature.append(pssm_score1[p_WT.ResIDSeq, AAind[resWT]-1])
        Feature.append(pssm_score1[p_MT.ResIDSeq, AAind[resMT]-1] \
                      -pssm_score1[p_WT.ResIDSeq, AAind[resWT]-1])
        Feature.append(np.sum(pssm_score1[p_WT.ResIDSeq, :]))

        Feature.append(pssm_score2[p_MT.ResIDSeq, AAind[resMT]-1])
        Feature.append(pssm_score2[p_WT.ResIDSeq, AAind[resWT]-1])
        Feature.append(pssm_score2[p_MT.ResIDSeq, AAind[resMT]-1] \
                      -pssm_score2[p_WT.ResIDSeq, AAind[resWT]-1])
        Feature.append(np.sum(pssm_score2[p_WT.ResIDSeq, :]))

        Feature.extend(pssm_score3[p_WT.ResIDSeq].tolist())

        #p_WT.FeatureSeq = np.load(f'{p_WT.filename}_dssp.npy', allow_pickle=True)
        #p_MT.FeatureSeq = np.load(f'{p_MT.filename}_dssp.npy', allow_pickle=True)

        # SS features
        Feature.extend(p_MT.FeatureSeq)
        Feature.extend(p_WT.FeatureSeq)
        Feature.extend(map(operator.sub, p_MT.FeatureSeq, p_WT.FeatureSeq))
    #print(len(Feature))

    return np.array(Feature) # construct_feature_aux

if __name__ == '__main__':
    from structure import get_structure
    s = get_structure(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    s.generateMutedPDBs()
    s.generateMutedPQRs()
    s.readFASTA()
    s.writeFASTA()

    # PDBid, Body, Chain, ResName, ResID, ResIDSeq, Sequence, typeFlag
    s_time = time.time()
    p = protein(s, 'WT')
    print(p.rips_complex_sheaf_spectra().max())
    print(p.alpha_complex_sheaf_spectra().max())
    # for idx, iAtom in enumerate(p.Atoms):
    #     print(iAtom.pos, iAtom.Charge)
    # p.construct_feature_global()
    # p.construct_feature_env()
    e_time = time.time()
    # print(e_time-s_time)

    #from src import PyProtein
    #PDBID = '1NCA_LH_WT'.encode('utf-8')
    #cp = PyProtein(PDBID)
    #if not cp.loadPQRFile(PDBID+'.pqr'.encode('utf-8')):
    #    sys.exit('reaqd PQR file faild')
    #cp.atomwise_interaction(10., 40.)
    #VDW2 = cp.feature_CLB()
    #for i in range(6539):
    #    if abs(np.sum(VDW1[i][:]-VDW2[i][:])) > 1:
    #        print(i, np.sum(VDW1[i][:]-VDW2[i][:]))
    #cp.Deallocate()
