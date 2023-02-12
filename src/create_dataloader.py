## Helper class for DataLoader
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

import pandas as pd
import cv2

import torch 
from  torchvision import transforms, models
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# import warnings
# warnings.filterwarnings('ignore')

def read_df(df_type):
    """
    Helper function to read each csv file
    """
    df_path = 'test/testdata/{}.csv'.format(df_type)
    df = pd.read_csv(df_path)[['Mask_Path', 'XRay_Path']]
    df['No_Pneumothorax'] = df['Mask_Path'].str.contains('negative_mask').astype(int)
    df['Yes_Pneumothorax'] = 1 - df['No_Pneumothorax']
    
    
    return df


class CANDID_PTX(Dataset):
    """
    Main class for the data loader, 'C' for Classification model, 'S' for Segmentation model
    """
    def __init__(self, df, resolution, model_type):
        self.img_paths = df['XRay_Path'].values
        self.mask_paths = df['Mask_Path'].values
        self.labels = torch.tensor(df[['Yes_Pneumothorax', 'No_Pneumothorax']].values, dtype=torch.float32)
    
        self.resolution = resolution
        
        # model_type: 'C' for Classification, 'S' for Segmentation
        self.model_type = model_type
            
    
    def __len__(self):
        
        return self.img_paths.shape[0]
    
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        ### Read in png files instead of dicom since uploaded radiograph images are in png format ### 
        img = plt.imread(img_path)[:, :, 0]
        #############################################################################################
        img_min = np.min(img)
        img_max = np.max(img)
        img_norm = (img - img_min) / (img_max - img_min)
        img_norm = cv2.resize(img_norm, (self.resolution, self.resolution))
        img_norm = torch.tensor(img_norm).expand(3, self.resolution, self.resolution)
        
        if self.model_type == 'C':
            label = self.labels[idx]
            
            return img_norm, label
        
        elif self.model_type == 'S':
            mask_path = self.mask_paths[idx]
            mask = plt.imread(mask_path)[:, :, 0]
            mask = cv2.resize(mask, (self.resolution, self.resolution))
            mask = torch.tensor(mask).expand(1, self.resolution, self.resolution) 

            return img_norm, mask
        

def create_loader(RESOLUTION, model_type, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DROP_LAST):
    """
    Main function to call in run.py file to generate three data loaders
    """
    train_df = read_df('train')
    val_df = read_df('validation')
    test_df = read_df('test')

    train_ds = CANDID_PTX(train_df, RESOLUTION, model_type)
    val_ds = CANDID_PTX(val_df, RESOLUTION, model_type)
    test_ds = CANDID_PTX(test_df, RESOLUTION, model_type)

    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, 
                      pin_memory = PIN_MEMORY, drop_last = DROP_LAST, shuffle = True)

    val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, 
                              pin_memory = PIN_MEMORY, drop_last = DROP_LAST, shuffle = False)

    test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, 
                              pin_memory = PIN_MEMORY, drop_last = DROP_LAST, shuffle = False)

    return train_loader, val_loader, test_loader
        
