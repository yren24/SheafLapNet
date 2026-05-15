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

## Feature Generation

This section outlines the procedures for extracting the multi-modal features required by the SheafLapNet framework. 

### 1. Persistent Sheaf Laplacian (PSL) Features

Extract structural and topological invariants using the PSL framework.

```bash
# Generic usage
python feature_Lap.py <PDB ID> <Protein Chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>

# Example (Ensure necessary modules are loaded if using an HPC environment)
module purge
module load GCC/12.3.0 OpenMPI/4.1.5-GCC-12.3.0

python feature_Lap.py 1AFO A A A 65 P 7
```

### 2. Auxiliary Physicochemical Features

#### 2.1 Evolutionary Conservation (PSSM via BLAST+)

Generate Position-Specific Scoring Matrices (PSSM) to capture evolutionary constraints.

```bash
# Generic usage (Requires BLAST+ v2.14.1)
python prepare.py <PDB ID> <Protein chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>

# Example (Ensure BLAST+ is loaded in your job script: module load BLAST+/2.14.1-gompi-2023a)
python prepare.py 1A4Y A A D 435 A 7.0 
```

> **Dependency Note:** Secondary structure feature extraction requires DSSP (v4.2.2.1). If not already configured, install it via: `conda install sbl::dssp`

#### 2.2 Electrostatic Solvation Energy (MIBPB)

Calculate electrostatic potentials using the MIBPB solver.

```bash
# Generic usage (Requires an input PQR file)
mibpb5 <PQR filename> h=0.7
```

### 3. ESM-2 Sequence Embeddings

Extract high-dimensional, pre-trained language model representations for the protein sequences.

```bash
# Generic usage
python feature_seq.py <PDB ID> <Protein chains> <Mutation chain> <Wild Residue> <Residue ID> <Mutant Residue> <pH>

# Example
python feature_seq.py 1AFO A A A 65 P 7.0
```
## Reproduce our results

1. Run `build_2648.py` `Fit_S2648.py` to generate the features for the dataset
2. Run `SheafLapNet.py` to perform the machine learning training and prediction
