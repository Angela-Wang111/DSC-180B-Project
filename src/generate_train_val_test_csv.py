import pandas as pd
import matplotlib.pyplot as plt
# import pydicom as dicom

def SOPInstanceUID_to_Mask_Path(SOPInstanceUID):
    """
    Helper function to turn SOPInstanceUID into paths with mask png files
    """
    return 'test/testdata/masks/' + SOPInstanceUID + '.png'

def SOPInstanceUID_to_Path(SOPInstanceUID):
    """
    Helper function to turn SOPInstanceUID into paths with radiograph png files
    """
    return 'test/testdata/images/' + SOPInstanceUID + '.png'



def generate_three_csv(input_csv_path):
    """
    Main function to take the sample original csv, split into 8, 1, 1, and then save three desired CSV files. 
    """
    input_csv = pd.read_csv(input_csv_path)
    csv_small = input_csv[['SOPInstanceUID', 'EncodedPixels']]
    # Chest X-Ray path is the same for positive and negative cases
    csv_small['XRay_Path'] = csv_small['SOPInstanceUID'].apply(SOPInstanceUID_to_Path)
    positive_cases = csv_small[csv_small['EncodedPixels'] != '-1']
    positive_cases['Mask_Path'] = positive_cases['SOPInstanceUID'].apply(SOPInstanceUID_to_Mask_Path)
    positive_cases = positive_cases[['Mask_Path', 'XRay_Path']]
    
    negative_cases = csv_small[csv_small['EncodedPixels'] == '-1']
    negative_cases['Mask_Path'] = 'test/testdata/masks/negative_mask.png'
    negative_cases = negative_cases[['Mask_Path', 'XRay_Path']]
    
    csv_small = pd.concat([positive_cases, negative_cases]).reset_index(drop = True)
    csv_small = csv_small.sample(frac = 1, random_state = 42)
    
    # Take the first 80 rows as control, then 10 for validation, 10 for test
    train_set = csv_small.iloc[:80]
    print("Number of positive cases in training set: ", 
          80-(train_set['Mask_Path'].str.contains("negative").sum()))
    train_set.to_csv("test/testdata/train.csv")
    
    val_set = csv_small.iloc[80:90]
    print("Number of positive cases in validation set: ", 
          10-(val_set['Mask_Path'].str.contains("negative").sum()))
    val_set.to_csv("test/testdata/validation.csv")
    
    test_set = csv_small.iloc[90:]
    print("Number of positive cases in test set: ", 
          10-(test_set['Mask_Path'].str.contains("negative").sum()))
    test_set.to_csv("test/testdata/test.csv")
    
    # For the train dataframe, we split them into positive part and negative part to address the imbalance during training process
    train_pos = train_set[~train_set['Mask_Path'].str.contains("negative")]
    train_pos.to_csv("test/testdata/train_pos.csv")
    
    train_neg = train_set[train_set['Mask_Path'].str.contains("negative")]
    train_neg.to_csv("test/testdata/train_neg.csv")
         
    
    return
