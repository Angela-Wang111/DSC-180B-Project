# DSC-180B-Project
GROUP NAME: AC/DS (Angela + Cecilia -> AC, Data Science -> DS, AC/DC -> AC/DS)
- segmentation + classification of penumothorax dataset

## Goal
Compare the classification performance of 
- classfication model e.g. resnet 34
- segmentation model e.g. resnet 34 + unet, after setting hard cutoff for binarization threshold & minimum activation size
- ensemble model with segmentation model + classification model

### Current ensemble approach
1. Binarized mask
- binarized mask as input for classification model
- extended gray-scale mask (max-value convolutional kernel to extend the mask area to include the "sharp edge")
2. Probability mask
- prabability mask as input for classfication model
- 2-channel mask with original xray value + probability mask
