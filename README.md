# Multi-Database-DCE-MRI-Breast-Tumor-Seg

## Overview
This repository provides the method described in the paper:
```
Longxi Zhou, et al. "Seeing the Unseen: Discovering Interpretable Sub-Visual Abnormalities in CT Scans of COVID-19 Patients and Survivors by Deep Learning"
```

## Description
Deep-LungParenchyma-Enhancing (DLPE) is a computer-aided detection (CADe) method for detecting and quantifying pulmonary parenchyma lesions on chest computerized tomography (CT). Using deep-learning, DLPE removes irrelevant tissues other than pulmonary parenchyma, and calculates the scan-level optimal window which enhances parenchyma lesions for dozens of times compared to the lung window. Aided by DLPE, radiologists discovered novel and interpretable lesions from COVID-19 inpatients and survivors, which are previously invisible under the original lung window and have strong predictive power for key COVID-19 clinical metrics and sequelae.

### Workflow for DLPE Method
<div align="center">
  <img src="./resources/FIg_one.png" width="1500" height="400">
</div>

## Run DLPE Method
- Step 1): Download the file: "trained_models/" and "example_data/" from [Google Drive](https://drive.google.com/drive/folders/16ZvZfhqMmuF7wqNPKUOntw2P-Mfx5C4l?usp=sharing).
- Step 2): Dowload the source codes from github (note in github, "trained_models/" and "example_data/" are empty files).
- Step 3): Replace the "trained_models/" and "example_data/" with Google Drive downloaded.
- Step 4): Establish the python environment by 'resources/req.txt'.
- Step 5): Open 'interface/dcm_to_enhanced.py', follow the instructions to change global parameters "trained_model_top_dict", "dcm_directory" and "enhance_array_output_directory".
- Step 6): Run 'interface/dcm_to_enhanced.py'.

## Time and Memory Complexity
- DLPE method requires GPU ram >= 6 GB and CPU ram >= 24 GB.
- Enhancing one chest CT scan needs about two minutes on one V100 GPU. 

## Contact
If you request our training code for DLPE method, please contact Prof. Xin Gao at xin.gao@kaust.edu.sa.

