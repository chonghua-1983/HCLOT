========================================= <br>
HCLOT
This is an implement of HCLOT algorithm on microbe-drug data. This file includes codes and data that are used to find the latent interaction relationships between drugs and microbes, and cross-community high-order interactions. Hypergraph contrastive learning and optimal transport plan are adopted to enhance the predictive ability of models. 

Datasets
This folder includes MDAD and aBiofilm datasets to train HCLOT. These two datasets are publicly available from the directory "datasets/MDAD" and "datasets/aBiofilm".
 

Runing environment and Requirements:
Running environment：python 3.11 or later.
Packages: numpy: 1.26.2, pandas: 2.0.0, torch: 2.1.1, tqdm: 4.66.1, scikit-learn: 1.5.1.

Usage Instructions:
HCLOT includes the main functions below:
args.py:
This is the first step, which sets model parameters and some hyperparameters including GPU, the dimensions of hidden layers, dropout rate, learning rate, epochs and so on.
test.py
This is the second step, which is a step-by-step tutorial for implementing HCLOT. This file contains data loading, preprocessing, normalization, hypergraph construction, HCOLT training and prediction.

model.py:
This is the main code, where the deep learning model are designed, built and trained, and finally the results are returned.

utils:
This folder includes some common functions and tools related to HCLOT, such as dataset.py used to load data, distance.py used to compute distance between samples, metrics.py and evaluate.py that are used to assess the performance of algorithms.

Please cite to our paper if you use any information from this repository

