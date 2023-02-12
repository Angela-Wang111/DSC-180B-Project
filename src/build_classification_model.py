import pandas as pd
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os

import torch 
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.optim import Adam


import cv2
from tqdm.notebook import tqdm


import warnings
warnings.filterwarnings('ignore')

import segmentation_models_pytorch as smp


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

## Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
print(torch.cuda.device_count())


class resnet34(nn.Module):

    """

    ResNet34 model, pretrained with ImageNet weights, for pure classification model. 

    """

    def __init__(self):

        super().__init__()

        self.model = models.resnet34(pretrained=True)
        
        layers = np.array([layer for layer in self.model.children()])
        
        for layer in layers[:-2]:
        # Only unfroze the last two layers for ResNet 34 model
            for param in layer.parameters():
                param.requires_grad = False
                
        self.model.fc = nn.Linear(512, 2)
        

    def forward(self, x):

        x = self.model(x)

        return x
    
def plot_save_both_loss(all_train_loss, all_val_loss, model_name, resolution):
    """
    Helper function to plot out the train and validation loss throughout the training process 
    """
    plt.figure(figsize=(20, 10))
    sns.set(style="whitegrid")
    epoch_num = len(all_train_loss)
    df = pd.DataFrame({'x':range(epoch_num),
                    'train_loss':all_train_loss,
                      'val_loss':all_val_loss})
    df = df.set_index('x')
    
    train_val_loss = sns.lineplot(data=df, linewidth=2.5)

    ## now label the y- and x-axes.
    plt.ylabel('BCE Loss')
    plt.xlabel('Epoch Number')
    plt.title('BCE Loss of {} with resolution {}'.format(model_name, resolution))
    plt.savefig('output/train_val_loss_{}.png'. format(model_name))
    plt.show()
    
    return
        
        
        
def training_classifier(model, num_epochs, batch_size, learning_rate, 
                    train_loader, val_loader, resolution):
    """
    Main function to call in run.py to return trained models and lists of training and validation loss.
    Save the plotted loss functions into the "output/" folder
    """
    
    model.to(DEVICE)
    
    all_train_loss = []
    all_val_loss = []
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in tqdm(range(num_epochs)):
        total_train_loss = 0
        batch_num = 0
        model.train()
        
        for i, (imgs, labels) in enumerate(train_loader):
            batch_num += 1
            
            imgs, labels = imgs.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.float)
            
            optimizer.zero_grad()
            preds = model(imgs)

            
            loss = loss_fn(preds, labels)
    
            loss.backward()
            optimizer.step()
            
            total_train_loss += float(loss)
            
            
        if epoch == 0:
            print("Total # of training batch: ", i + 1)

        all_train_loss.append(total_train_loss / batch_num)
            
            
    ## validation set
        batch_num = 0
        total_val_loss = 0
        model.eval()
        
        for i, (imgs, labels) in enumerate(val_loader):
            batch_num += 1
            
            imgs, labels = imgs.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.float)
            
            preds = model(imgs)
            
            loss = loss_fn(preds, labels) # is this mean or sum?

            total_val_loss += float(loss) # accumulate the total loss for this epoch.
            
            
        if epoch == 0:
            print("Total # of validation batch: ", i + 1)

        all_val_loss.append(total_val_loss / batch_num)
        
    print("Training losses: ", all_train_loss)
    print("Validation losses: ", all_val_loss)
    plot_save_both_loss(all_train_loss, all_val_loss, "ResNet34", str(resolution))
    print("Successfully saved the training and validation loss graph!")
        
    return model, all_train_loss, all_val_loss
        