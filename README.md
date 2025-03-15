# RF Fingerprinting using CNNs

## Project Overview
This repository contains code for **CS6482 Assignment-1: RF Fingerprinting using CNNs**, developed as part of the **M.Sc. in Artificial Intelligence and Machine Learning** program. The project leverages deep learning techniques to analyze and classify RF signals, specifically exploiting **IQ imbalance due to hardware imperfection**.

## Dataset
The project works with RF signal datasets, where signals are transformed into spectrograms using **Short-Time Fourier Transform (STFT)**. These spectrograms are then used as input for a **ResNet-50-based Convolutional Neural Network (CNN)** for classification.

## Technologies Used
- **Python 3.x**
- **PyTorch** (for deep learning model implementation)
- **Torchvision** (for ResNet-50 model)
- **SciPy** (for signal processing and STFT generation)
- **Matplotlib** (for visualization)
- **NumPy** (for numerical operations)

## Installation
To set up the environment, install the required dependencies:

```bash
pip install numpy matplotlib torch torchvision scipy pillow
```

## Code Structure
- **Data Preprocessing:** Converts raw IQ signals into spectrogram images.
- **Model Definition:** Utilizes a pre-trained ResNet-50 model for feature extraction and classification.
- **Training Pipeline:** Includes data loading, model training, and evaluation.
- **Visualization:** Generates plots to analyze the training process and model performance.

## Running the Notebook
To run the Jupyter Notebook, execute the following commands:

```bash
jupyter notebook CS6482_Assign1_24141623.ipynb
```

Ensure that your dataset is placed in the appropriate directory, and modify the paths in the notebook accordingly.

## Results and Observations
The model's performance is evaluated using accuracy, loss curves, and confusion matrices. Further improvements may include:
- Data augmentation techniques for robustness.
- Experimenting with different CNN architectures.
- Fine-tuning hyperparameters for optimal performance.

## Author
**Ahmad Salim**
M.Sc. in Artificial Intelligence and Machine Learning

## License
This project is for academic purposes and should not be used for commercial applications without permission.

