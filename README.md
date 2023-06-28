# Deep learning-based age estimation from clinical Computed Tomography image data of the thorax and abdomen in the adult population

Aging is an important risk factor for disease, leading to morphological change that can be assessed on Computed Tomography (CT) scans. We propose a deep learning model for automated age estimation based on CT-scans of the thorax and abdomen of 1653 subjects, that were generated in a clinical routine setting.

The model and its weights are available in this project as well as the used Score-CAM-implementation and preprocessing suite.

![model](https://github.com/BjarneKerber/age_estimation/blob/main/images/model.jpg "Visualization of our proposed model.")

## Overview on Methods
A pre-trained ResNet-18 model was modified to predict chronological age as well as to quantify its aleatoric uncertainty. The model was trained using 1653 non-pathological CT-scans of the thorax and abdomen of subjects aged between 20 and 85 years in a 5-fold cross-validation scheme. Generalization performance, reliability and robustness were assessed on a publicly available test dataset [TCIA dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287) consisting of thorax-abdomen CT-scans of 421 subjects. Score-CAM saliency maps were generated for interpretation of model outputs and to identify hallmarks of aging.

During preprocessing, the original DICOM-data was converted to Nifti-format. To remove artifacts like stretchers, the patient's body was segmentend from the image using sobel edge detection and watershed segmentation. Maximum intesitiy projections were computed. Data augmentation was performed by cropping random patches and resizing them. 

The model was implemented using the Pytorch Framework. Models were trained and evaluated using 5-fold patient-leave-out cross-validation. Testing was performed on a publicly available dataset 

Heteroscedastic aleatoric uncertainty was modeled using the Gaussian negative log likelihood loss as empirical risk minimization objective.

To identify subregions, that were most influential for the prediction of the network, Score-CAM saliency maps were generated. 

Technical validation was performed to ensure robustness, reproducibility and generalizability of our proposed model.

## Exemplary model predictions and saliency maps generated using Score-CAM
<p float="left">
  <img src="https://github.com/BjarneKerber/age_estimation/blob/main/images/scc3.png" width="300" />
  <img src="https://github.com/BjarneKerber/age_estimation/blob/main/images/scc2.png" width="300" /> 
  <img src="https://github.com/BjarneKerber/age_estimation/blob/main/images/scc1.png" width="300" />
</p>
