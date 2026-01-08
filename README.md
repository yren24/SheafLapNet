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
* softwares need to be installed (See readme at folder software)

## Reproduce our results

1. Run `build_2648.py` `Fit_S2648.py` to generate the features for the dataset
2. Run `SheafLapNet.py` to perform the machine learning training and prediction
