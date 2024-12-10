This repository contains the implementation of various experiments related to training a Deep Convolutional GAN (DC GAN) and evaluating it on different tasks including image generation, latent space traversal, conditional generation, Wasserstein GAN (WGAN), and classifier training.

## Overview

The repository covers the following tasks:

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
