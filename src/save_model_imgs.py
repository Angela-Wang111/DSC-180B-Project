import pandas as pd
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import cv2
from tqdm import tqdm

import torch 
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.optim import Adam

import segmentation_models_pytorch as smp

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score



def save_model(cur_model, model_name):
    """
    model_name: usually in the form of 'UNet_ResNet34_ep20_bs4_lr-4' / 'RN34_UN_ep20_bs4_lr-4'
    """
    # Save segmentation model!!!!
    torch.save(cur_model.state_dict(), 'test/saved_model/{}.pth'.format(model_name))
    
    
    
    

def load_model(model_name):
    """
    model_name: usually in the form of 'UNet_ResNet34_ep20_bs4_lr-4' / 'cla_RN34_UN_ep20_bs4_lr-4'
    """
    # Load saved segmentation model
    path = 'test/saved_model/{}.pth'.format(model_name)
    
    # Decide the type of the model
    model_params = model_name.split('_')
    model_type = model_params[0]
    
    if model_type == "seg":
        encoder = model_params[1]
        if encoder = "RN34":
            model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels = 3, classes=1, activation=None)
        else:
            model = smp.Unet("efficientnet-b3", encoder_weights="imagenet", in_channels = 3, classes=1, activation=None)
    else:
        if model_type == "cla":
            encoder = model_params[1]
        else:
            encoder = model_params[3]
                 
        if encoder = "RN34":
            model = resnet34()
        else:
            model = eNet_b3()

    model.load_state_dict(torch.load(path))
    model.eval()
    
    return model


def save_images_predicted_by_static_model(model, data_loader, BATCH_SIZE, model_name):
    """
    Helper function to help the training process less repetitive. Take in trained, static model (assumed to be moved to GPU already),
    and save the predicted images from each separate data_loader.
    
    model_name: EB3_UN / RN34_UN
    """
    for i, (imgs, masks, sops) in tqdm(enumerate(data_loader)):
        # Get predicted masks from current model
        imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(DEVICE, dtype=torch.float)
        preds = model(imgs)

        # Save the first channel as the original image (Format: [4 x 3 x 256 x 256])
        # The 3 channels are exactly the same due to gray scale
        new_img = imgs.detach().cpu().numpy()
        # Change the last channel to predicted mask
        new_img[:, 2] = preds.detach().cpu().squeeze()

        for k in range(BATCH_SIZE):
        # Deal with the dimensionality issue and the normalization for the predicted mask channel, then save
            ## deal with the last batch which has less than BATCH_SIZE number of samples
            if k >= len(new_img):
                break
                
            cur_img = new_img[k]
            cur_img = np.swapaxes(cur_img.transpose(), 0, 1)
            out_min = np.min(cur_img[:, :, 2])
            out_max = np.max(cur_img[:, :, 2])
            out_norm = (cur_img[:, :, 2] - out_min) / (out_max - out_min)
            cur_img[:, :, 2] = out_norm
            # Save
            plt.imsave('test/testdata/intermediate_data/{}/{}_predicted.png'.format(model_name, sops[k]), cur_img)
    
    return


def save_imgs_based_on_model(model, loaders_seg, all_neg_loader, loader_type, model_name):
    """
    Main function to actually save all the predicted images
    """
    model.eval()
    model.to(DEVICE)
    if loader_type == "train":
        save_images_predicted_by_static_model(model, all_neg_loader, BATCH_SIZE, model_name)
        print("Saved all the predicted masks for training!")
    elif loader_type == 'validation':
        save_images_predicted_by_static_model(model, loaders_seg[1], BATCH_SIZE, model_name)
        print("Saved all the predicted masks for validation!")
    else:
        save_images_predicted_by_static_model(model, loaders_seg[2], BATCH_SIZE, model_name)
        print("Saved all the predicted masks for test!")
    
    return
    
    
    
    
