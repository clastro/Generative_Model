## Variational Autoencoder (VAE) Overview

### 1. Introduction

A Variational Autoencoder (VAE) is a type of generative model introduced by Kingma and Welling in 2013. VAEs learn a latent space representation of high-dimensional data, which allows for the generation of new data samples and the analysis of the underlying features of the data.

### 2. Structure of VAE

VAEs have a structure similar to traditional Autoencoders but with key differences:

- Encoder: Maps input data to a probability distribution in the latent space. The encoder outputs the mean (
𝜇
μ) and standard deviation (
𝜎
σ) of the latent variables 
𝑧
z.
- Latent Space: Latent variables 
𝑧
z are sampled from a normal distribution.
- Decoder: Reconstructs the original input data from the latent variables 
𝑧
z. The decoder takes 
𝑧
z as input and outputs the distribution of the original data 
𝑥
x.

### 3. Loss Function of VAE

The loss function of a VAE consists of two parts:

Reconstruction Loss: Measures how well the input data is reconstructed. This is typically computed using Mean Squared Error (MSE) or Binary Cross-Entropy.
- KL Divergence Loss: Measures how closely the distribution of the latent variables 
𝑧
z matches a normal distribution.
The total loss function is expressed as:
𝐿
=
- Reconstruction Loss
+
KL Divergence Loss
L=Reconstruction Loss+KL Divergence Loss
