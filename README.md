# Conditional Variational Autoencoder (CVAE) for Motor Imagery EEG Signal Generation

## Project Overview

This project implements a Conditional Variational Autoencoder (CVAE) designed to generate synthetic EEG signals corresponding to four distinct motor imagery classes: left hand, right hand, feet, and tongue. The goal is to create a deep learning model that can generate realistic brain signal patterns associated with specific imagined movements.

### Key Features
- Preprocessing of EEG data from the BCI Competition IV 2a dataset
- Conditional generative model using Variational Autoencoder
- Synthetic EEG signal generation for four motor imagery classes
- Detailed data exploration and visualization

## Project Structure

```
project_root/
│
├── bci_iv_2a_data/           # Directory for raw BCI dataset
│   ├── A01T.mat               # Training data for subject 1
│   ├── A01E.mat               # Evaluation data for subject 1
│   └── ...
│
├── explore_mat.py            # Script to explore .mat file structure
├── preprocess_bci_data.py    # EEG data preprocessing script
├── cvae_eeg_model.py         # CVAE model implementation and training
│
├── eeg_data_preprocessed.npz # Preprocessed data (generated during run)
├── cvae_loss_curve.png       # Training loss visualization
├── cvae_generated_samples.png# Generated EEG signal samples
└── README.md                 # Project documentation
```

## Prerequisites

### Hardware Requirements
- a gpu (preferrably a good one)
- 16 gb 

### Software Dependencies
- Python 3.8+
- libraries:
  - numpy
  - scipy
  - matplotlib
  - scikit-learn
  - torch
  - MNE-Python
  - Braindecode

### how to install

1. Clone the repository:
```bash
git clone "insert project linK"
cd eeg-cvae-project
```

2. Install dependencies:
```bash
pip install numpy scipy matplotlib scikit-learn torch mne
```

## Dataset Source and Preparation

### Data Origin
The dataset used in this project is from the BCI Competition IV 2a, which can be found at:
- **Source**: [BNCI Horizon 2020 Database](https://bnci-horizon-2020.eu/database/data-sets)
- this is our `bci_iv_2a_data` dir

### Dataset Details
- **Type**: Motor Imagery EEG Dataset
- **Number of Subjects**: 9
- **Motor Imagery Classes**: 
  1. Left Hand
  2. Right Hand
  3. Feet
  4. Tongue
- **experimental setup**: 
  - High-resolution EEG recordings
  - 22 EEG channels
  - Sampling rate: 250 Hz
  - Recorded using the Graz BCI setup

### Data Preparation Steps
1. Download the BCI Competition IV 2a dataset from the BNCI website
2. Place .mat files in the `bci_iv_2a_data/` dir
3. Files should be named in the format: 
   - Training files: A01T.mat, A02T.mat, ..., A09T.mat
   - Evaluation files: A01E.mat, A02E.mat, ..., A09E.mat

## Project Workflow

### 1. Data Exploration
```bash
python explore_mat.py
```
- We used this to investigate/understand the structure of .mat files

### 2. Data Preprocessing
```bash
python preprocess_bci_data.py
```
- Loads and preprocesses EEG data
- Performs:
  - Trial extraction
  - Normalization
  - Train-test splitting
- Generates `eeg_data_preprocessed.npz`

### 3. CVAE Model Training
```bash
python cvae_eeg_model.py
```
- Trains the Conditional Variational Autoencoder
- Generates:
  - Model checkpoints
  - Loss curve visualization
  - Synthetic EEG signal samples

## Model Architecture

### Encoder
- Input: Flattened EEG signals + one-hot encoded class
- Layers: 
  - Fully connected layers
  - ReLU activation
- Outputs: 
  - Latent space mean
  - Latent space log variance

### Decoder
- Input: Latent vector + one-hot encoded class
- Layers:
  - Fully connected layers
  - ReLU activation
- Output: Reconstructed EEG signal

## Key Hyperparameters
- Latent Dimension: 64
- Epochs: 100
- Batch Size: 64
- Learning Rate: 1e-3

## Visualizations

![image](https://github.com/user-attachments/assets/38c85f59-79f8-4f7c-b2f5-9762462ccee2)

Here we have the visualization of the left hand across 6 channels.


### Loss Curve
`cvae_loss_curve.png` shows training and testing loss over epochs.

![cvae_loss_curve](https://github.com/user-attachments/assets/0b4843f9-4a08-4d4a-9b2f-2f6e7c5cf739)

- The model learns quickly in the first 20-30 epochs
- The training loss is consistently lower than the test loss, which could indicate:
  - Some overfitting
  - The model is learning the training data more closely than generalizing
- The gradual increase in test loss suggests the model might be starting to overfit

### Generated Samples
`cvae_generated_samples.png` displays synthetic EEG signals for each motor imagery class.

![cvae_generated_samples](https://github.com/user-attachments/assets/bfaef8c3-8a81-4c1d-9991-a64585c85378)

- The CVAE is able to generate synthetic signals that maintain key characteristics of EEG data
- The signals appear to have:
  - Realistic amplitude ranges
  - Complex, non-uniform fluctuations
  - Consistent statistical properties within each class

Subtle differences between classes suggest the model has learned to capture class-specific signal characteristics, however, we beed to work on improving our cvae.

## Project Implementation Status and Roadmap

### Current Progress

#### 1. Data Preprocessing and Feature Extraction
- Successfully implemented comprehensive data loading and preprocessing pipeline
- Key achievements:
  - Developed robust method to extract trials from continuous EEG data
  - Implemented z-score normalization across trials
  - Created flexible data loading for multiple subjects
  - Added detailed error handling and debugging mechanisms
- Challenges overcome:
  - Handling complex .mat file structures
  - Managing inconsistent data formats across different subjects

#### 2. CVAE Model Implementation and Initial Training
- Developed a Conditional Variational Autoencoder (CVAE) architecture
- Model features:
  - Fully connected encoder and decoder networks
  - Reparameterization trick for latent space sampling
  - Conditional generation based on motor imagery class
- Training progress:
  - Implemented custom loss function combining reconstruction and KL divergence
  - Created training and testing loops with epoch-wise checkpointing
  - Generated initial visualizations of loss curves and synthetic signals

### Next Steps and Future Work

#### 3. Refinement and Hyperparameter Tuning
Planned improvements:
- Implement systematic hyperparameter search
  - Explore different latent space dimensions
  - Experiment with network architectures
  - Optimize learning rates and batch sizes
- Add more sophisticated regularization techniques
- Investigate alternative network architectures
  - exploration of convolutional layers (maybe)
  - attention mechanisms for better signal representation

#### 4. Evaluation and Validation
Proposed validation approaches:
- Quantitative Evaluation:
  - Implement signal quality metrics
  - Compare generated signals with original data
  - Statistical tests for signal similarity
- Classification Performance:
  - Train classifiers on original and synthetic data
  - Compare classification accuracy
  - Assess generalization capabilities
- Potential Validation Methods:
  - Inception score
  - Fréchet Inception Distance (adapted for EEG)
  - Cross-domain classification tests

## Considerations
- Current implementation uses a subset of available subjects
- Model performance can be improved with:
  - More advanced preprocessing
  - Comprehensive hyperparameter tuning
  - Larger and more diverse dataset

## Authors
- Zain Khalid
- Nate Joseph
```
