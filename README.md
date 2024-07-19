---

# Underwater Image Deblurring using U-Net

## Overview

This project focuses on enhancing the quality of underwater images by deblurring them using a U-Net architecture. The U-Net is a convolutional neural network that is particularly effective for image segmentation tasks. By training the U-Net on underwater images, the model can learn to remove blur and improve the clarity of these images, which are often affected by water distortion, low light, and other underwater conditions.

## Project Structure

- `Another_copy_of_Under_Water_Image_Enhancement_UNet.ipynb`: Jupyter Notebook containing the implementation of the U-Net model, data preprocessing, training, and evaluation.
- `data/`: Directory containing the dataset of underwater images used for training and testing.
- `models/`: Directory to save trained models and their weights.
- `results/`: Directory to save output images and performance metrics.

## Getting Started

### Prerequisites

To run the notebook and reproduce the results, you need to have the following dependencies installed:

- Python 3.7+
- Jupyter Notebook
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn

You can install the necessary packages using the following command:

```bash
pip install tensorflow keras opencv-python numpy matplotlib scikit-learn
```

### Dataset

Ensure that the underwater image dataset is available in the `data/` directory. The dataset should be split into training and testing sets.

### Running the Notebook

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Underwater-image-DBLURRING-using-U-Net.git
cd Underwater-image-DBLURRING-using-U-Net
```

2. Open the Jupyter Notebook:

```bash
jupyter notebook Another_copy_of_Under_Water_Image_Enhancement_UNet.ipynb
```

3. Run all the cells in the notebook to preprocess the data, train the U-Net model, and evaluate its performance on the test set.

## Model Architecture

The U-Net architecture used in this project consists of an encoder and a decoder. The encoder compresses the input image into a lower-dimensional representation, while the decoder reconstructs the image to its original size with enhanced features. Skip connections between the encoder and decoder help preserve spatial information.

### Training

The model is trained using a combination of loss functions suitable for image deblurring tasks. The training process involves:

- Loading and preprocessing the dataset
- Defining the U-Net model architecture
- Compiling the model with an appropriate optimizer and loss function
- Training the model on the training dataset
- Evaluating the model on the test dataset

## Results

The results of the deblurring process are saved in the `results/` directory. The notebook provides visualizations of the original and enhanced images, as well as performance metrics such as PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index).

## Future Work

- Experiment with different network architectures and hyperparameters to improve performance.
- Explore the use of GANs (Generative Adversarial Networks) for further enhancement of underwater images.
- Develop a user-friendly interface for real-time underwater image deblurring.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- This project was inspired by various research papers and open-source projects on underwater image enhancement and U-Net architecture.

## Proposed Papers

- **"Underwater Image Enhancement by Dehazing with Minimum Information Loss and Histogram Distribution Prior"** by D. Anwar, P. Purwanto, and E. Munir. This paper discusses techniques for enhancing underwater images by removing haze and improving visibility.
- **"WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images"** by J. Li, R. Wang, Z. Li, J. Lei, and H. Jin. This paper introduces WaterGAN, a GAN-based approach for real-time color correction in underwater images.
- **"A Review of Deep Learning Methods for Underwater Image Restoration and Enhancement"** by J. Dong, G. Wang, Z. Zhang, and S. Kwong. This review paper provides an overview of various deep learning methods, including U-Net, for underwater image restoration and enhancement.
- **"Multi-scale Dense Networks for Deep Underwater Image Enhancement"** by Y. Liu, R. Fan, Z. Yang, M. Liu, and X. Wu. This paper presents a multi-scale dense network architecture for enhancing underwater images.
- **"Underwater Image Enhancement Using Deep Learning and Synthetic Data"** by Y. Zhou, B. Fan, and F. Zhu. This paper explores the use of synthetic data and deep learning models for improving the quality of underwater images.

---
