import pandas as pd
import numpy as np
import pydicom as dicom
import matplotlib.pylab as plt
import matplotlib.cm as cm
import os


def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    """
    Helper function to decode run length masks
    """
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    
    return component


def save_positive_masks(output_path, df):
    """
    Helper function to save processed masks as png
    """
    for i in range(df.shape[0]):
        row = df.iloc[i]
        sop = row.name
        plt.imsave(output_path + '{}.png'.format(sop), row.DecodedPixels, cmap = cm.gray)
    
    return

def save_negative_masks(output_path):
    """
    Helper funtion to save negative, black mask
    """
    negative_mask = np.zeros((1024, 1024))
    plt.imsave(output_path + 'negative_mask.png', negative_mask, cmap = cm.gray)
    
    return
    


def decode_mask(test_csv_path, output_path):
    """
    Main function to decode mask. Assume input df with 100 rows
    ONLY CALL THIS ONE!
    """
    df_pneumo = pd.read_csv(test_csv_path)
    df_pos = df_pneumo[df_pneumo['EncodedPixels'] != "-1"][['SOPInstanceUID', 'EncodedPixels']]
    df_pos['DecodedPixels'] = df_pos['EncodedPixels'].apply(run_length_decode)
    df_pos_combine = df_pos.groupby('SOPInstanceUID').sum()
    save_positive_masks(output_path, df_pos_combine)
    save_negative_masks(output_path)
    
    
    return
    
    