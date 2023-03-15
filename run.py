import sys
import os
import json

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import os

sys.path.insert(0, 'src')

# Import functions from other py files
from run_model import run_class
from run_model import run_seg
from run_model import run_cas


def main(targets):
    """
    Run the main project pipeline logic
    """

    # Detect model type, and run all models for this type
    model_type = targets[0]
    
    with open('config.json', 'r') as fh:
        params = json.load(fh)
    
    if model_type == "classification":
        run_class(model_type, "", **params)
    elif model_type == "segmentation":
        run_seg(model_type, **params)
    else:
        run_cas(model_type, **params)
    
    
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)