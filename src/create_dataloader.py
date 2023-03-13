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


def read_df(df_type, model_name):
    """
    Helper function to read each csv file
    
    df_type: 'train_pos'/'train_neg'/'validation'/'test' (type of the dataframe)
    model_name: 'RN34_UN'/'EB3_UN' (name of the segmentation model)
    """
    df_path = 'test/testdata/{}.csv'.format(df_type)
    df = pd.read_csv(df_path)[['Mask_Path', 'XRay_Path']]
    
    
    df['SOP'] = df['XRay_Path'].apply(lambda x: x.split('/')[-1])
    directory_path = 'test/testdata/intermediate_data/{}/'.format(model_name)
    predicted_suffix = '_predicted.png'
    df['Intermediate_Predicted_Path'] = df['SOP'].apply(lambda x: directory_path + x + predicted_suffix)


    df['No_Pneumothorax'] = df['Mask_Path'].str.contains('negative_mask').astype(int)
    df['Yes_Pneumothorax'] = 1 - df['No_Pneumothorax']
    
    
    return df




class CANDID_PTX(Dataset):
    def __init__(self, df, resolution, model_type):
        self.img_paths = df['XRay_Path'].values
        self.intermediate_paths = df['Intermediate_Predicted_Path'].values
        self.mask_paths = df['Mask_Path'].values
        self.labels = torch.tensor(df[['No_Pneumothorax', 'Yes_Pneumothorax']].values, dtype=torch.float32)
        # Just changed by Angela
        self.sop = df['SOP'].values
        self.resolution = resolution
        
        # model_type: 'Class' for Classification, 'Seg' for Segmentation, 'Cas' for Cascade
        self.model_type = model_type
              
        return
            
    
    def __len__(self):
        
        return self.img_paths.shape[0]
    
    
    def __getitem__(self, idx):
        if self.model_type == 'Cas':
            # Designed for ensemble model's classification part
            label = self.labels[idx]
            
            new_img_path = self.intermediate_paths[idx]
            new_img = plt.imread(new_img_path)[:, :, :3]
            to_tensor = transforms.ToTensor()
            new_img = to_tensor(new_img)
            
            return new_img, label
        
        else:
            img_path = self.img_paths[idx]
            img = dicom.dcmread(img_path).pixel_array
            img_min = np.min(img)
            img_max = np.max(img)
            img_norm = (img - img_min) / (img_max - img_min)
            img_norm = cv2.resize(img_norm, (self.resolution, self.resolution))
            img_norm = torch.tensor(img_norm).expand(3, self.resolution, self.resolution)

            if self.model_type == 'Cla':
                # Designed for classification model
                label = self.labels[idx]

                return img_norm, label

            elif self.model_type == 'Seg':
                # Designed for segmentaion models 
                mask_path = self.mask_paths[idx]
                mask = plt.imread(mask_path)[:, :, 0]
                mask = cv2.resize(mask, (self.resolution, self.resolution))
                mask = torch.tensor(mask).expand(1, self.resolution, self.resolution) 
        
                sop = self.sop[idx]

                return img_norm, mask, sop
        

def create_loader(RESOLUTION, model_type, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DROP_LAST):
    """
    Main function to call in run.py file to generate three data loaders
    """
    val_df = read_df('validation')
    test_df = read_df('test')

    val_ds = CANDID_PTX(val_df, RESOLUTION, model_type)
    test_ds = CANDID_PTX(test_df, RESOLUTION, model_type)


    val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, 
                              pin_memory = PIN_MEMORY, drop_last = DROP_LAST, shuffle = False)

    test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, 
                              pin_memory = PIN_MEMORY, drop_last = DROP_LAST, shuffle = False)

    return val_loader, test_loader


def create_train_loaders(schedule_type, cur_df=None, num_neg=0, model_type=None):
    if schedule_type == 3:
        cur_df = pd.concat([pos_df, neg_df.sample(n=num_neg, replace=False)]).sample(frac=1, ignore_index=True)
        
    train_ds = CANDID_PTX(cur_df, RESOLUTION, model_type)

    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, 
                          pin_memory = PIN_MEMORY, drop_last = DROP_LAST, shuffle = True)

    return train_loader, cur_df.shape[0]
