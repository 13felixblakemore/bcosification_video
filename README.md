# B-Cosification for Video Action Recognition

Extending **B-Cosification to 3D convolutional neural networks**, enabling inherently interpretable deep learning for video action recognition.

## Overview

Most explainable AI methods generate explanations after a neural network has made a prediction. B-Cosification instead modifies neural network architectures to make their predictions inherently interpretable.

This project extends B-Cosification from image classification to **spatiotemporal video classification**, allowing models to highlight which regions and frames contribute to their predictions.

The approach is implemented using an I3D-style architecture and evaluated on the **UCF101 action recognition dataset**.

## Implementation

- **3D B-Cosification:** Extended B-Cosification to 3D convolutional architectures for video processing.
- **Video classification:** Trained and evaluated B-cosified video models on UCF101.
- **Spatiotemporal explanations:** Generated visualisations identifying influential regions across video frames.
- **Interpretability evaluation:** Implemented experiments measuring explanation localisation, faithfulness and logit reconstruction.
- **Benchmarking:** Compared B-Cos explanations against Grad-CAM++ and investigated computational overhead.

## Technologies

**Python · PyTorch · Computer Vision · Explainable AI · 3D CNNs**

## Acknowledgements

This project builds upon [B-Cosification: Transforming Deep Neural Networks to be Inherently Interpretable](https://github.com/shrebox/B-cosification) (Arya et al., NeurIPS 2024). The original framework is adapted and extended for video classification and spatiotemporal interpretability.

## Author

[Felix Blakemore](https://github.com/13felixblakemore)
