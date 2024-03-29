# Multi-Database-DCE-MRI-Breast-Tumor-Segmentation

## Overview
This repository provides the method described in the paper:
```
Longxi Zhou, et al. "Heart Region Anchoring: a Model-Agnostic Workflow to Improve Segmentation Robustness for Breast Cancer in DCE-MRI"
```

## Description
The repository gives a novel deep-learning workflow specifically designed for harmonizing intra-dataset variations and inter-datasets biases. The workflow is trained and evaluated on multi-centric datasets. Our workflow includes four components: 1) the machine-agnostic standard embedding space for breast DCE-MRI; 2) the stage-one model that outputs a high recall mask for tumors (average recall >0.99); 3) the stage-two model that refines the high recall mask and outputs the 3D tumor probability map; and 4) the model that calculates adaptive thresholds to get the final segmentation.

### Workflow
<div align="center">
  <img src="./resources/Fig_one.png" width="1500" height="350">
</div>

## Run The Trained 2.5D Models
2.5D model performed the best in the breast tumor segmentation.
- Step 1): Download the file: "trained_models/" and "DCE-MRI_data/" from [Google Drive](https://drive.google.com/drive/folders/1qOUi9gdrsFk3TCtTbcmP4BYBJJAOkcyz?usp=sharing). 
- Step 2): Dowload the source codes from github (note in github, "trained_models/" and "DCE-MRI_data/" are empty files).
- Step 3): Replace the "trained_models/" and "DCE-MRI_data/" with Google Drive downloaded.
- Step 4): Establish the python environment by 'resources/req.txt'.
- Step 5): Open './breast tumor seg.py', change the directory in line 5 and line 8.
- Step 6): Run './breast tumor seg.py'.

## Time and Memory Complexity
- The workflow requires GPU ram >= 6 GB and CPU ram >= 24 GB.
- Segment one scan needs about 15 seconds on one V100 GPU. 

## Demos for Other Models
You may use other models mentioned in our paper. But they will require more CPU and GPU ram.
- 2DUnet: https://github.com/lzx325/COVID-19-repo/tree/master/03.baselines.demo/2D%20U-net
- 3DUnet: https://github.com/lzx325/COVID-19-repo/tree/master/03.baselines.demo/3D%20U-net
- 3DVnet: https://github.com/lzx325/COVID-19-repo/tree/master/03.baselines.demo/3D%20V-net
- MPUnet: https://github.com/lzx325/COVID-19-repo/tree/master/03.baselines.demo/MPUnet

## Contact
If you request our training code for DLPE method, please contact Prof. Xin Gao at xin.gao@kaust.edu.sa.

