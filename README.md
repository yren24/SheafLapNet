# SheafLapNet

Persistent Sheaf Laplacian analysis of Protein Stability and Solubility upon mutation

## Prerequisites

* numpy 1.23.5
* scipy 1.11.3
* scikit-learn 1.3.2
* python 3.10.12
* fair-esm 2.0.0
* torch 2.1.1
* pytorch-cuda 11.7
* softwares need to be installed (See readme at folder bin)

## Feature generation

### Persistent Sheaf Laplacian Embedding

```bash
#
python feature_Lap.py <PDB ID> <Protein Chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>

# examples
module purge
module load GCC/12.3.0 OpenMPI/4.1.5-GCC-12.3.0
python feature_SR.py 1AFO A A A 65 P 7
```

### Auxiliary Features

#### BLAST+ and generate PSSM

```bash
# Generate PSSM scoring matrix (Requires BLAST+ 2.14.1)
python prepare.py <PDB ID> <Protein chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>

# Add module load BLAST+/2.14.1-gompi-2023a into job script
# Run BLAST+ PSSM calculations
python prepare.py 1A4Y A A D 435 A 7.0 
```

```bash
conda install sbl::dssp (DSSP v4.2.2.1)
```

#### MIBPB calculation

```bash
# Requires pqr file
mibpb5 <PQR filename> h=0.7
```

### ESM-2 Transformer Features

```bash
# Generate transformer features
python feature_seq.py <PDB ID> <Protein chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>

# examples
python feature_seq.py 1AFO A A A 65 P 7.0
```

The `blast_jobs`, `feat_jobs`, `seq_jobs` and `Lap_jobs` folder contains scripts used to run feature generation process in a step-by-step procedure with the help of a high performance computing resource.

## Reproduce our results

1. Run `build_2648.py` `Fit_S2648.py` to generate the features for the dataset
2. Run `SheafLapNet.py` to perform the machine learning training and prediction
