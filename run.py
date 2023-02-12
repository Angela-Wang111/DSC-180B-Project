import sys
import os
import json

import pandas as pd
import numpy as np
import pydicom as dicom
import matplotlib.pylab as plt
import matplotlib.cm as cm
import os

sys.path.insert(0, 'src')

# Import functions from other py files
from data_preprocessing import save_positive_masks
from data_preprocessing import save_negative_masks
from data_preprocessing import decode_mask

from generate_train_val_test_csv import generate_three_csv

def main():
    """
    Run the main project pipeline logic
    """
    decode_mask("test/testdata/Pneumothorax_reports_small.csv", "test/testdata/")
    generate_three_csv("test/testdata/Pneumothorax_reports_small.csv")
    
    
if __name__ == '__main__':
#     targets = sys.argv[1:]
    main()