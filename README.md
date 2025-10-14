# MLA_SpyNet
MLA-SpyNet: Multi-Layer Attention SpyNet for Video Anomaly Detection using optical flow and CBAM attention modules.

# Overview
MLA-SpyNet is a deep learning framework designed for video anomaly detection using optical flow predictions. It leverages multi-layer attention mechanisms (CBAM) applied at each convolutional layer to improve feature extraction and focus on salient motion patterns in video sequences. The framework consists of three separate components:

1. Training Phase: Trains the MLA-SpyNet model on consecutive frames from video sequences. Computes predicted optical flow and End-Point Error (EPE) with respect to actual consecutive frames. Applies Bayesian thresholding on EPE to identify anomalous motion patterns without requiring manual threshold tuning. Saves the trained model and computed Bayesian threshold for later testing.

2. Testing Phase: Uses the trained MLA-SpyNet to process unseen video sequences. Computes predicted optical flow between consecutive frames. Compares predictions with actual consecutive frames using EPE. Flags frames as anomalous if the EPE exceeds the Bayesian threshold computed during training.

3. Evaluation Phase: Computes ROC curves and AUC values to assess anomaly detection performance quantitatively. Generates plots for visualization of the trade-off between true positive and false positive rates.

# Requirements
Python 3.8+
TensorFlow 2.x
OpenCV (cv2)
NumPy
scikit-learn
SciPy
Matplotlib

# Install dependencies using:
pip install tensorflow opencv-python numpy scikit-learn scipy matplotlib

# Link for datasets: 
UCSD Ped1 and Ped2: http://svcl.ucsd.edu/projects/anomaly/dataset.htm 
Avenue: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html 
ShanghaiTech: https://svip-lab.github.io/dataset/campus_dataset.html

# DSAPM Model
The training script uses DSAPM as a pre-trained placeholder model for initial optical flow prediction. The DSAPM model can be downloaded from [https://github.com/DevashreePatrikar/DSAPM_RBFN_DAS.git] Place the downloaded model in the models/ folder as dsapm.h5.

# Usage
1. Training
python train_mla_spynet.py

*Inputs: Training video sequences from data/Train_Data/
*Outputs: models/mla_spynet_trained.h5 (trained model)
          models/bayesian_threshold.npy (Bayesian EPE threshold)

2. Testing
python test_mla_spynet.py

*Inputs: Test video sequences from data/Test_Data/
Uses the trained MLA-SpyNet and Bayesian threshold.
*Outputs: Prints frames detected as anomalous.

3. Evaluation
python evaluate_mla_spynet.py

Computes ROC curve and AUC using flow scores and ground-truth labels.

*Outputs: Printed AUC value. ROC plot showing detection performance.

# Notes
Attention: CBAM (Channel + Spatial Attention) is applied after each convolutional layer in SpyNet for enhanced feature focus. Bayesian Thresholding: Automatically computes anomaly threshold using the distribution of EPE values from training sequences. Ground Truth: Consecutive frames are used as a reference to compute optical flow for EPE calculation. Ensure bayesian_threshold.npy is saved after training for use during testing.

# References
1. CBAM: Convolutional Block Attention Module – Woo et al., 2018
2. SpyNet: Optical Flow Estimation – Ranjan & Black, 2017
3. DSAPM pre-trained model [https://github.com/DevashreePatrikar/DSAPM_RBFN_DAS.git]

This code is directly associated with the manuscript ‘[Multi-Level Attention Enhanced Spatial Pyramid Network for Precise Video Anomaly Localization]’ submitted to The Visual Computer. Please cite this manuscript if you use this code
