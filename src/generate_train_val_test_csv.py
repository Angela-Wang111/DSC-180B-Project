import pandas as pd
import matplotlib.pyplot as plt
import pydicom as dicom

def SOPInstanceUID_to_Mask_Path(SOPInstanceUID):
    """
    Helper function to turn SOPInstanceUID into paths with mask png files
    CHANGE USER NAME!!!
    """
    return '/home/anw008/teams/dsc-180a---a14-[88137]/CANDID_PTX_MASKS/' + SOPInstanceUID + '.png'



def generate_three_csv(input_csv_path):
    """
    Main function to take the sample original csv, split into 8, 1, 1, and then save three desired CSV files. 
    """
    input_csv = pd.read_csv(input_csv_path)
    csv_small = input_csv[['SOPInstanceUID', 'EncodedPixels']]
    # Take the first 80 rows as control, then 10 for validation, 10 for test
    train_set = csv_small.iloc[:80]
    val_set = csv_small.iloc[80:90]
    test_set = csv_small.iloc[90:]
    
    
    return