import pandas as pd
import numpy as np
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
from tqdm import tqdm


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
    
class eNet_b3(nn.Module):

    """

    EfficientNet-B3 model, pretrained with ImageNet weights, for pure classification model. 

    """

    def __init__(self):

        super().__init__()

        self.model = models.efficientnet_b3(pretrained = True)

        layers = np.array([layer for layer in self.model.children()])
        
        for layer in layers[-3][:-3]:
            for param in layer.parameters():
                param.requires_grad = False
        
                
        self.model.classifier[1] = nn.Linear(in_features=1536, out_features=2)

    def forward(self, x):

        x = self.model(x)

        return x
    

def plot_save_both_loss(all_train_loss, all_val_loss, model_type, model_name, model_schedule='2'):
    """
    Helper function to plot out the train and validation loss throughout the training process 
    """
    plt.figure(figsize=(20, 10))
    sns.set(style="whitegrid")
    epoch_num = len(all_train_loss)
    df = pd.DataFrame({'x':np.arange(1, epoch_num+1),
                    'train_loss':all_train_loss,
                      'val_loss':all_val_loss})
    df = df.set_index('x')
    train_val_loss = sns.lineplot(data=df, linewidth=2.5)
    # set the ticks first
    train_val_loss.set_xticks(np.arange(1, epoch_num+1, 1))

    # set the labels
    train_val_loss.set_xticklabels(np.arange(1, epoch_num+1, 1))


    ## now label the y- and x-axes.
    plt.ylabel('BCE Loss')
    plt.xlabel('Epoch Number')
    title = 'BCE Loss of {} model, {}'.format(model_type, model_name)
    plt.title(title)
    plt.show()
    
    fig = train_val_loss.get_figure()
    fig.savefig('output/both_loss/{}_type{}.png'.format(title, model_schedule), dpi=400)


        


def training_class(model, num_epochs, batch_size, learning_rate, 
                    val_loader, model_name, model_type):
    
    model.to(DEVICE)
    
    all_train_loss = []
    all_val_loss = []

    optimizer = Adam(model.parameters(), lr=learning_rate)
    pos_total = pos_df.shape[0]
    neg_total = neg_df.shape[0]
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    for epoch in tqdm(range(num_epochs)):
        total_train_loss = 0
        batch_num = 0
        model.train()
        
        if epoch % 4 == 0:
            cur_group = epoch // 4
            cur_df = pd.concat([pos_df, 
                        neg_df.iloc[int(cur_group * pos_total) : 
                                    int(np.minimum((cur_group + 1) * pos_total, neg_total))]]).sample(frac=1, 
                                                                                              ignore_index=True)
            train_loader, cur_num = create_train_loaders(2, cur_df, num_neg=0, model_type=model_type[:3])

        
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
            
            loss = loss_fn(preds, labels) 

            total_val_loss += float(loss) # accumulate the total loss for this epoch.

        if epoch == 0:
            print("Total # of validation batch: ", i + 1)

        all_val_loss.append(total_val_loss / batch_num)
        
    
    print("Training losses: ", all_train_loss)
    print("Validation losses: ", all_val_loss)
    plot_save_both_loss(all_train_loss, all_val_loss, model_type=modele_type, model_name=model_name, model_schedule='2')
    print("Successfully saved the training and validation loss graph!") 
    
    return model, all_train_loss, all_val_loss



def training_seg(model, num_epochs, batch_size, learning_rate, 
                    val_loader, test_loader, val_threshold, model_name, model_type):
    """
    Main training function to train the first part of the ensemble model, which is the segmentation model.
    """
    
    model.to(DEVICE)
    
    all_train_loss = []
    all_val_loss = []
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    pos_total = pos_df.shape[0]
    neg_total = neg_df.shape[0]
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    for epoch in tqdm(range(num_epochs)):
        total_train_loss = 0
        batch_num = 0
                
        # If indicated number of epochs are not met, then keep optimizing the segmentation order.
        model.train()
        
        # Imbalanced dataset solution: (Strategy 2, resampling negative cases)
        if epoch % 4 == 0:
            cur_group = epoch // 4
            cur_df = pd.concat([pos_df, 
                        neg_df.iloc[int(cur_group * pos_total) : 
                                    int(np.minimum((cur_group + 1) * pos_total, neg_total))]]).sample(frac=1, 
                                                                                              ignore_index=True)
            train_loader, cur_num = create_train_loaders(2, cur_df, num_neg=0, model_type=model_type[0])

        for i, (imgs, masks, sops) in enumerate(train_loader):
            batch_num += 1
            imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(DEVICE, dtype=torch.float)
            optimizer.zero_grad()
            preds = model(imgs)

            # Calculate loss and do back-propagation, then calculate total loss for this epoch
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.detach().cpu())

        all_train_loss.append(total_train_loss / batch_num)
        # Print the number of training batch
        if epoch == 0:
            print("Total # of training batch: ", i + 1)

    ## validation set
        batch_num = 0
        total_val_loss = 0
        model.eval()

        for i, (imgs, masks, sops) in enumerate(val_loader):
            batch_num += 1
            # Send imgs and masks to GPU so that they can be input to the model
            imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(DEVICE, dtype=torch.float)

            preds = model(imgs)
            loss = loss_fn(preds, masks) # is this mean or sum?
            total_val_loss += float(loss.detach().cpu()) # accumulate the total loss for this epoch.

        # Calculate the overall validation loss and dice-coefficient
        all_val_loss.append(total_val_loss / batch_num)

        if epoch == 0:
            print("Total # of validation batch: ", i + 1)
        
    
    print("Training losses: ", all_train_loss)
    print("Validation losses: ", all_val_loss)
    plot_save_both_loss(all_train_loss, all_val_loss, model_type=modele_type, model_name=model_name, model_schedule='2')
    print("Successfully saved the training and validation loss graph!") 
        
    return model, all_train_loss, all_val_loss
        


