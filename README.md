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
## Reproduce Our Results

Follow these steps to construct the final dataset and train the deep learning model.

### 1. Dataset Construction

Aggregate the extracted features into a unified dataset matrix:

```bash
python build_2648.py
python Fit_S2648.py
```

### 2. Model Training and Prediction

Run the main neural network script to perform model training and evaluation.

```bash
# Default execution
python SheafLapNet.py

# Example with custom hyperparameters
python SheafLapNet.py --dataset S2648 --epochs 100 --lr 0.001 --batch_size 50
```

#### Available Command-Line Arguments

You can fully customize the model architecture and training process using the following arguments:

| Option | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Target dataset for training/testing | `S2648` |
| `--datatype` | Specific feature combinations to load | `all` |
| `--batch_size` | Input batch size for training | `50` |
| `--epochs` | Number of training epochs | `100` |
| `--lr` | Initial learning rate | `0.001` |
| `--momentum` | SGD momentum factor | `0.9` |
| `--weight_decay`| Weight decay (L2 penalty) for optimizer | `0.05` |
| `--layers` | Architecture dimensions (e.g., hidden layer sizes) | `2048,1024,1024,512,512,64` |
| `--nlayer` | Total number of neural network layers | `6` |
| `--seed` | Random seed for reproducibility | `42` |
| `--log_interval`| Batches to wait before logging training status | `1` |
| `--no_cuda` | Flag to disable CUDA and force CPU training | `False` |
