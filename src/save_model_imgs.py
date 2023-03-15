"""
save_model_imgs.py contains functions to save predicted masks from pre-trained segmentation models. Mainly used for preparing images to be input to the classification models during the cascade model training.
"""
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

from create_dataloader import create_train_loaders



def save_model(cur_model, model_name):
    """
    Helper function to save the weights of any models to a designated folder.
    Input model_name: usually in the form of 'RN34_UN_ep20_bs4_lr0.0001'
    """
    torch.save(cur_model.state_dict(), 'test/saved_model/{}.pth'.format(model_name))
    
    return


def save_images_predicted_by_static_model(model, data_loader, batch_size, model_name):
    """
    Helper function to help the training process less repetitive. Called by "save_imgs_based_on_model"
    Input: model_name: EB3_UN / RN34_UN
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (imgs, masks, sops) in tqdm(enumerate(data_loader)):
        # Get predicted masks from current model
        imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(DEVICE, dtype=torch.float)
        preds = model(imgs)

        # Save the first channel as the original image (Format: [4 x 3 x 256 x 256])
        # The 3 channels are exactly the same due to gray scale
        new_img = imgs.detach().cpu().numpy()
        # Change the last channel to predicted mask
        new_img[:, 2] = preds.detach().cpu().squeeze()

        for k in range(batch_size):
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


def save_imgs_based_on_model(model, val_loader, test_loader, loader_type, model_name, resolution, batch_size, num_workers, pin_memory, drop_last):
    """
    Main function to actually save all the predicted images based on trained models. Take in trained, static model (assumed to be moved to GPU already), and save the predicted images from each separate data_loader.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(DEVICE)
    if loader_type == "train":
        train_loader, cur_num = create_train_loaders(resolution, batch_size, num_workers, pin_memory, DROP_LAST=False, schedule_type=3, num_neg=0, model_type="seg", model_prev=model_name)
        print("current training loader number of samples is: {}".format(cur_num))
        save_images_predicted_by_static_model(model, train_loader, batch_size, model_name)
        print("Saved all the predicted masks for training!")
    elif loader_type == 'validation':
        save_images_predicted_by_static_model(model, val_loader, batch_size, model_name)
        print("Saved all the predicted masks for validation!")
    else:
        save_images_predicted_by_static_model(model, test_loader, batch_size, model_name)
        print("Saved all the predicted masks for test!")
    
    return
    
    
    
    
