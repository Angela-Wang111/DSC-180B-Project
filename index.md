---
layout: default
title: "Pneumothorax Classification"
categories: main
---


# Introduction
What is pneumothorax? It is also called collapsed lung. It happens when a patient has a puncture/punctures on their lung, causing air to leak out into the space between lung and the chest wall. In this way, patients breathe in less air for each breath. Some of the most common symptoms include sudden chest pain and shortness of breath. Chest X-rays are the most common method for patients to get diagnosis for pneumothorax. However, only very few experts in radiology are able to give the diagnosis, creating very long wait times for patients to get proper treatments. 

<center>
<figure>
<img src="website_fig/pneumo.png" alt="Trulli" style="width:70%">
<figcaption align = "center"><b>Fig 1: Pneumothorax Example. Credit: https://en.wikipedia.org/wiki/Pneumothorax </b></figcaption>
</figure>
</center>

To solve this pressing problem, machine learning models like convolutional neural networks (CNNs) have been widely used to help give patients diagnoses based on their chest X-rays. There are many structures of CNNs, including classification, localization, object detection, and instance segmentation models. The differences between these models are illustrated below:

<center>
<figure>
<img src="website_fig/classification_segmentation.png" alt="Trulli" style="width:70%">
<figcaption align = "center"><b>Fig 2: Different CNN models. Credit: https://www.v7labs.com/blog/image-segmentation-guide </b></figcaption>
</figure>
</center>

There has been a lot of previous research testing on the performance of classification and segmentation models on pneumothorax detection. However, very few previous publications look at cascade models, where a classification model is piped after a segmentation model. We are interested in asking: **Do cascade models outperform both classification and segmentation models, given that cascade models should theoretically inherit the benefits of both models? **


# Method
We have three main types of CNNs to compare:
 - Classification model
 - Segmentation model
 - Cascade model (a segmentation model followed by a classification model)

## Model Selection 
For the classification model, we tried ResNet 34 and EfficientNet-B3. 

## Model Pipeline

# Result

# Conclusion
