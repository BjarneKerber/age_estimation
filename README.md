# Deep learning-based age estimation from clinical Computed Tomography image data of the thorax and abdomen in the adult population

Aging is an important risk factor for disease, leading to morphological change that can be assessed on Computed Tomography (CT) scans. We propose a deep learning model for automated age estimation based on CT-scans of the thorax and abdomen of 1653 subjects, that were generated in a clinical routine setting.

![model](https://github.com/BjarneKerber/age_estimation/blob/main/images/model.jpg "Visualization of our proposed model.")

A pre-trained ResNet-18 model was modified to predict chronological age as well as to quantify its aleatoric uncertainty. Generalization performance was assessed on a publicly available test dataset consisting of thorax-abdomen CT-scans of 421 subjects. Score-CAM saliency maps were generated for interpretation of model outputs and to identify hallmarks of aging.

The model and its weights are available in this project as well as the used Score-CAM-implementation and preprocessing suite.

![saliency](https://github.com/BjarneKerber/age_estimation/blob/main/images/scorecam.png "Score-CAM saliency maps")
