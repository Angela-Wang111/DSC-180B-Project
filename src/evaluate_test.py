"""
evaluate_test.py contains functions to plot, print, and save metrics based on the test set to evaluate all models.
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

def bi_mask(logit_mask, threshold):
    """
    Helper function to binarize masks based on given threshold for segmentation models.
    """
    mask = np.where(logit_mask <= threshold, 0, 1)
    
    return mask


def calculate_dc(preds, true_mask):
    """
    Helper function to calculate dice coefficient from predicted and true masks, both in binary.
    """
    if np.sum(true_mask) == 0:
        # Return 0 if prediction is positive but true mask is negative
        if np.sum(preds) != 0:
            return 0
        else:
            # Return 1 if prediction and true mask are both negative
            return 1
    else:
        # Return 0 if prediction is negative but true mask is positive
        if np.sum(preds) == 0:
            return 0
        # Actually calculate dc if both prediction and mask are both positive
        dc = np.sum(preds[true_mask==1])*2.0 / (np.sum(preds) + np.sum(true_mask))
        return dc
    

def plot_confusion_matrix(y_test, y_true, model_type, model_name, model_schedule='2'):
    """
    Helper function to plot and save confusion matrices for given model based on test set results.
    """
    cm = confusion_matrix(y_true, y_test)
    cm = sns.heatmap(cm, annot=True, cmap = 'Blues', fmt="d")
    title = 'Confusion matrix of {} model, {}'.format(model_type, model_name)
    plt.title(title)
    plt.xlabel('Prediction')
    plt.ylabel('True')
    
    fig = cm.get_figure()
    fig.savefig('output/confusion_matrix/{}_type{}.png'.format(title, model_schedule), dpi=400)
    plt.show()
    plt.close(fig)
    
    return


    
def plot_roc_curve(y_test, y_true, model_type, model_name, model_schedule='2'):
    """
    Helper function to plot and save ROC curves for given model based on test set results.
    """
    fpr, tpr, threshold = roc_curve(y_true, y_test, drop_intermediate = False)
    roc_auc = roc_auc_score(y_true, y_test)

    roc_plt = plt.figure(1)
    plt.plot([0, 1], [0, 1])
    plt.plot(fpr, tpr, label='{}(area = {:.3f})'.format(model_name, roc_auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    title = 'ROC curve of {} model, {}'.format(model_type, model_name)
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig('output/auc_roc/{}_type{}.png'.format(title, model_schedule), dpi=400)
    
    plt.show()
    plt.close(roc_plt)
    
    return

    
def test_metrics_class(test_loader, model, model_type, model_name, model_schedule='2'):
    """
    Main function for plotting and printing metrics for classification models. 
    Call helper functions to plot and save confusion matrix and ROC curves. Print out F1-Score and Recall.
    Return true test labels and predicted test labels.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_test = np.array([])
    y_true = np.array([])
    y_test_prob = np.array([])
    
    total_num_batch = 0
    

    for i, (imgs, labels) in enumerate(test_loader):
        total_num_batch += 1
        imgs, labels = imgs.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.float)
        preds = model(imgs)
        
        soft_max = nn.Softmax(dim=1)
        pred_prob = soft_max(preds).detach().cpu().numpy()
        
        pred_label = np.argmax(pred_prob, axis=1)
        true_label = labels.detach().cpu().numpy().astype(int)[:, 1]
        
        y_test = np.concatenate((y_test, pred_label))
        y_true = np.concatenate((y_true, true_label))

        y_test_prob = np.concatenate((y_test_prob, pred_prob[:, 1]))
    
    
    plot_confusion_matrix(y_test, y_true, model_type, model_name, model_schedule)
    print("The F1-Score is: {}".format(f1_score(y_true, y_test)))
    print("The Recall (Sensitivity) is: {}".format(recall_score(y_true, y_test)))
    plot_roc_curve(y_test_prob, y_true, model_type, model_name, model_schedule)
    
    print('Total Number of Batch Size: ', total_num_batch)
    return y_test, y_true


def test_metrics_seg(test_loader, model, model_type, model_name, threshold, min_activation, batch_size, model_schedule='2'):
    """
    
    Main function for plotting and printing metrics for segmentation models. 
    Call helper functions to plot and save confusion matrix. No ROC curve for segmentation models. Print out model details and metrics.
    Return true test labels and predicted test labels.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    model.eval()

    y_test = np.array([])
    y_true = np.array([])
    test_dice = []

    total_num_batch = 0
    for i, (imgs, labels, sops) in tqdm(enumerate(test_loader)):
        total_num_batch += 1
        imgs, labels = imgs.to(DEVICE, dtype=torch.float), labels.to(DEVICE, dtype=torch.float)
        preds = model(imgs)
        preds_sigmoid = torch.sigmoid(preds[:, 0].squeeze())
        binarized = bi_mask(preds_sigmoid.detach().cpu().squeeze(), threshold)
        pred_label = np.array([np.sum(binarized_ind) > min_activation for binarized_ind in binarized]).astype(int)
        true_label = labels[:, 0].squeeze().detach().cpu().numpy()
        # True labels are 1 if sum of all pixels > 0, otherwith true label = 0
        true_label = np.array([np.sum(true_label_ind) > 0 for true_label_ind in true_label])
        y_test = np.concatenate((y_test, pred_label))
        y_true = np.concatenate((y_true, true_label))
        # Calculate Dice Coefficient (Disclaimer: NOT PERFECT, since the true masks are not binary)             
        batch_dice = []
        for batch_i in range(batch_size):
            cur_dc = calculate_dc(binarized[batch_i], labels[batch_i].detach().cpu().squeeze().numpy())
            batch_dice.append(cur_dc)
        test_dice.append(np.mean(batch_dice))
    # Plot confusion matrix for the segmentation model
    plot_confusion_matrix(y_test, y_true, model_type, model_name, model_schedule)
    print("Threshold for this segmentation model: ", threshold)
    print("Minimum Activation Size: ", min_activation)
    
    print("The F1-Score is: {}".format(f1_score(y_true, y_test)))
    print("The Recall (Sensitivity) is: {}".format(recall_score(y_true, y_test)))
                     
    print("The Mean Test Dice Coefficient is {}".format(np.mean(test_dice)))
    
    print('Total Number of Batch Size: ', total_num_batch)
    return y_test, y_true
    
