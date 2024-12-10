# Project1
This contains the implementation of various experiments related to training a Deep Convolutional GAN (DC GAN) and evaluating it on different tasks including image generation, latent space traversal, conditional generation, Wasserstein GAN (WGAN), and classifier training.

## Overview



1. **Training a DC GAN** with a chosen architecture using the standard GAN loss, along with plotting the loss curves for both the Generator and Discriminator.
2. **Generating images**: Plot a 10x10 grid of generated images after training the GAN.
3. **Varying training iterations**: Vary the number of times the Generator and Discriminator are trained and observe the effects on image quality and loss convergence.
4. **FID computation**: Compute the Fréchet Inception Distance (FID) by sampling 1000 data points from both the true and generated data distributions.
5. **Latent space traversal**: Implement latent space traversal using linear and non-linear interpolation methods and visualize the results.
6. **Conditional generation**: Implement a conditional GAN (c-GAN) to generate images by conditioning on class labels.
7. **Wasserstein GAN (WGAN)**: Modify the loss and Critic network to optimize the Wasserstein metric, and evaluate the model by plotting generated images and recomputing FID.
8. **Decoder network**: Implement a decoder network to reconstruct the latent variables used by the GAN generator and compute a reconstruction loss. Train this decoder along with the regular GAN losses.
9. **MLP classifier**: Use the decoder output as input to train a multi-layer perceptron (MLP) classifier and compute classification metrics (accuracy and F1 score).
10. **ResNet-based classifier**: Fine-tune a pre-trained ResNet model (32 or 50) and compare its classification results to the MLP classifier trained on decoded latents.
11. **ResNet on 20-class subset**: Train a ResNet-based classifier on a 20-class subset of the data and report classification metrics.
12. **Data augmentation with c-GAN**: Use the conditional GAN to generate 100 additional images per class and retrain the classifier with the augmented data. Compare results with the previous classifier.

## Example Generated images


<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://github.com/user-attachments/assets/466d508f-8629-4cc6-9087-ab019e30a891" alt="output_dc_gan_butterflies" width="400" style="margin-right: 10px;"/>
  <img src="https://github.com/user-attachments/assets/e1aa0e69-55ca-4886-980b-7624bd079299" alt="output_dc_gan_animals" width="400"/>
</div>

# Project2: Variational Autoencoder (VAE) and Related Experiments

This project implements various experiments related to training and evaluating a Variational Autoencoder (VAE) and its variants. The experiments focus on training a vanilla VAE, implementing conditional likelihood, evaluating performance on image generation and reconstruction, exploring latent space, and applying classifiers on latent vectors.

## Table of Contents

1. [Training a Vanilla VAE](#training-a-vanilla-vae)
2. [CNN-based Classifier](#cnn-based-classifier)
3. [Posterior Inference and MLP Classifier](#posterior-inference-and-mlp-classifier)
4. [Beta-VAE](#beta-vae)
5. [Latent Space Interpolation](#latent-space-interpolation)
6. [Adversarial Autoencoder](#adversarial-autoencoder)
7. [VQ-VAE with Discrete Latent Space](#vq-vae-with-discrete-latent-space)
8. [Gaussian Mixture Model (GMM) on Latent Space](#gmm-on-latent-space)

## 1. Training a Vanilla VAE

Implement a vanilla VAE with MSE loss for conditional likelihood. The model is trained, and results are evaluated with varying numbers of samples of `z` during training for the input to the decoder. The following outputs are reported:
- 10x10 grids of reconstructions and generations
- Loss curves for likelihood, KL divergence, and combined terms

### Results:
- **Reconstructed and Generated Images:**

## 2. CNN-based Classifier

A CNN-based classifier is built using the training images from the dataset. The classifier is evaluated on the test set, and classification accuracy is reported.


## 3. Posterior Inference and MLP Classifier

Using the VAE trained in the first step, posterior inference is performed on all images. The latent vectors obtained from this inference are then used to train a multi-layer perceptron (MLP) classifier. Classification performance is compared between the original CNN classifier and the MLP classifier based on latent vectors.


## 4. Beta-VAE

Implement a beta-VAE with four different values of the hyperparameter `beta`. The following results are observed and documented:
- 10x10 grids of generated and reconstructed images for each `beta` value
- Observations on how varying `beta` affects the results

### Results:
- Generated and Reconstructed Images:


## 5. Latent Space Interpolation

For the VAE trained with the optimal `beta`, posterior inference is performed on a pair of images. Linear interpolation is done along the latent space between corresponding latent vectors, and 10 interpolated points are visualized for 10 different image pairs.

### Results:


## 6. Adversarial Autoencoder

An adversarial autoencoder is implemented with MSE loss. The following outputs are evaluated:
- Generated images

### Results:
- **Generated Images:** Visual representation of images generated by the adversarial autoencoder.

## 7. VQ-VAE with Discrete Latent Space

A Vector Quantized VAE (VQ-VAE) is implemented with a discrete latent space. After training, posterior inference is performed on all images, and the latent vectors are used to build a classifier. The classifier's performance (accuracy) is computed.


## 8. Gaussian Mixture Model (GMM) on Latent Space

A Gaussian Mixture Model (GMM) is fit on the latent vectors obtained via posterior inference from the VAE. New latent vectors are sampled from the GMM, passed through the decoder, and a 10x10 grid of generated images is plotted.

# Project3: Denoising Diffusion Probabilistic Models (DDPM) and Related Experiments

This project implements various experiments related to training and evaluating Denoising Diffusion Probabilistic Models (DDPM) and other generative models such as DDIM. The project involves training DDPMs on both the butterfly dataset and the latent space of a VQ-VAE model, comparing different sampling procedures, and implementing advanced techniques like classifier-guided diffusion and score-based models.

## Table of Contents

1. [Training DDPM on the Butterfly Dataset](#training-ddpm-on-the-butterfly-dataset)
2. [Generated Images Visualization](#generated-images-visualization)
3. [FID Computation](#fid-computation)
4. [Training DDPM on VQ-VAE Latent Space](#training-ddpm-on-vq-vae-latent-space)
5. [Conditional Generation using Classifier-Guided Diffusion](#conditional-generation-using-classifier-guided-diffusion)
6. [Noise Conditional Score Network](#noise-conditional-score-network)
7. [DDIM Sampler Comparison](#ddim-sampler-comparison)
8. [DDIM Inversion Method](#ddim-inversion-method)
9. [ResNet-50 on Animal Dataset](#resnet-50-on-animal-dataset)
10. [Distillation to MLP](#distillation-to-mlp)
11. [i-JPEG on Animal Dataset](#i-jpeg-on-animal-dataset)

## 1. Training DDPM on the Butterfly Dataset

Train a Denoising Diffusion Probabilistic Model (DDPM) on the butterfly dataset. The following outputs are reported:
- U-Net training loss curves during DDPM training.

### Results:
- **U-Net Training Loss Curves:** Plot of loss curves for the U-Net model during training.
<img width="752" alt="image" src="https://github.com/user-attachments/assets/065a3693-1c59-4e1e-8bdc-3b4fbe665a13">


## 2. Generated Images Visualization

After training the DDPM on the butterfly dataset, generate images using the model and visualize the results by plotting a 10x10 grid of the generated images.

### Results:
- **Generated Images:** 10x10 grid of images generated by the trained DDPM.

<img width="647" alt="image" src="https://github.com/user-attachments/assets/b74db8d9-3863-4d78-8de1-980add03e366" width="400"/>


## 3. FID Computation

Compute the Fréchet Inception Distance (FID) by sampling 1000 data points from both the true and generated data distributions. This provides a measure of the quality of generated images.



## 4. Training DDPM on VQ-VAE Latent Space

Repeat the previous experiments by training the DDPM on the latent space of the VQ-VAE trained in the previous assignment. Compute the FID score and visualize the generated images.

### Results:
- **FID Score:** FID score computed for generated images from the DDPM trained on VQ-VAE latents.
- **Generated Images:** 10x10 grid of images generated by the DDPM trained on latent vectors.

## 5. Conditional Generation using Classifier-Guided Diffusion

Implement conditional generation using classifier-guided diffusion. This method guides the diffusion process based on class labels and is used for conditional image generation.

### Results:
- **Conditional Generation:** Generated images conditioned on class labels.
<img src="https://github.com/user-attachments/assets/4c3b56b6-54ad-4272-861f-ce1dab085274" alt="image" width="400"/>


## 6. Noise Conditional Score Network

Implement a Noise Conditional Score Network (NCSN) and repeat the above experiments. Compare the sampling procedures (speed and image quality) between the NCSN and DDPM.

### Results:
- **Sampling Comparison:** Comparison of sampling speed and generation quality (FID) between DDPM and NCSN.

## 7. DDIM Sampler Comparison

Using the same network trained for DDPM, implement a DDIM sampler and compare the generation quality (via FID) with the DDPM. No additional training is required for this step.

### Results:
- **DDIM vs DDPM Comparison:** FID scores comparing the quality of generated images between DDIM and DDPM.

## 8. DDIM Inversion Method

Implement a DDIM inversion method to get the latent vectors for a pair of real images. Perform linear interpolation in the latent space between these vectors and generate images corresponding to the interpolated latents.

### Results:
- **Latent Interpolation:** Generated images for interpolated latents between pairs of real images.

## 9. ResNet-50 on Animal Dataset

Train a ResNet-50 model on the Animal dataset and measure the accuracy on the test dataset.


## 10. Distillation to MLP

Distill the ResNet-50 model into a smaller-sized MLP using KL divergence loss across logits. Measure the test accuracy of the distilled MLP.
