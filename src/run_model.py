"""
run_model.py contains the entire pipeline to run classification, segmentation, and cascade models. Automatically run all models within the same structure.
"""
import sys
import numpy as np
import segmentation_models_pytorch as smp

from data_preprocessing import decode_mask
from generate_train_val_test_csv import generate_four_csv
from create_dataloader import create_loader
from build_model import resnet34
from build_model import eNet_b3
from build_model import training_class
from build_model import training_seg
from evaluate_test import test_metrics_class
from evaluate_test import test_metrics_seg
from save_model_imgs import save_model
from save_model_imgs import save_images_predicted_by_static_model
from save_model_imgs import save_imgs_based_on_model

def run_class(model_type, prev_model, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, THRESHOLD, MIN_ACTIVATION, RESOLUTION, NUM_WORKERS, PIN_MEMORY, DROP_LAST):
    """
    Main function to train classification models, usable for both pure classification and cascade. Runs both ResNet 34 and EfficientNet-B3.
    Input prev_model: "" for classification / "RN34_UN_" OR "EB3_UN_" for cascade
    """
    # Data Preprocessing
    decode_mask("test/testdata/Pneumothorax_reports_small.csv", "test/testdata/masks/")
    generate_four_csv("test/testdata/Pneumothorax_reports_small.csv")
    
    # generate val & test loaders
    val_loader, test_loader = create_loader(RESOLUTION, model_type, 
                                            BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DROP_LAST, prev_model[:-1])
    
    # define rn34 & efficientnet-b3
    model_rn34 = resnet34()
    model_eb3 = eNet_b3()

    model_set = np.array([model_rn34, model_eb3])
    model_name_set = np.array(['{}RN34'.format(prev_model), '{}EB3'.format(prev_model)])


    for model_idx in np.arange(model_set.shape[0]):
        cur_model = model_set[model_idx]
        cur_name = model_name_set[model_idx]
        # train the model
        cla_model, train_loss, val_loss = training_class(model=cur_model, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate = LEARNING_RATE, val_loader=val_loader, model_name=cur_name, model_type=model_type, resolution=RESOLUTION, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST, model_prev=prev_model[:-1])

        # test metric
        y_test, y_true = test_metrics_class(test_loader=test_loader, model=cur_model, model_type=model_type, model_name=cur_name, model_schedule='2')

        # save_models -> e.g.'RN34_UN_ep20_bs4_lr-4'
        file_name = '{}_ep{}_bs{}_lr{}'.format(cur_name, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
#         save_model(cla_model, file_name)
#         print("model saved")
        print("Finished classification testing!")

            
def run_seg(model_type, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, THRESHOLD, MIN_ACTIVATION, RESOLUTION, NUM_WORKERS, PIN_MEMORY, DROP_LAST):
    """
    Main function to train segmentation models, usable for both pure segmentation and cascade. Runs both ResNet 34 and EfficientNet-B3 as encoders.
    """
    # Data Preprocessing
    decode_mask("test/testdata/Pneumothorax_reports_small.csv", "test/testdata/masks/")
    generate_four_csv("test/testdata/Pneumothorax_reports_small.csv")
    
    # generate val & test loaders
    val_loader, test_loader = create_loader(RESOLUTION, model_type, 
                                            BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DROP_LAST)
    
    # define rn34_un & efficientnet-b3_un
    model_rn34 = smp.Unet("resnet34", encoder_weights="imagenet", in_channels = 3, classes=1, activation=None)
    model_eb3 = smp.Unet("efficientnet-b3", encoder_weights="imagenet", in_channels = 3, classes=1, activation=None)

    model_set = np.array([model_rn34, model_eb3])
    model_name_set = np.array(['RN34_UN', 'EB3_UN'])
    
    trained_model = []
   
    for model_idx in np.arange(model_set.shape[0]):
        cur_model = model_set[model_idx]
        cur_name = model_name_set[model_idx]
        # train the model
        seg_model, train_loss, val_loss = training_seg(model=cur_model, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate = LEARNING_RATE, val_loader=val_loader, model_name=cur_name, model_type=model_type, resolution=RESOLUTION, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=DROP_LAST)

        # test metric
        y_test, y_true = test_metrics_seg(test_loader=test_loader, model=cur_model, model_type=model_type, model_name=cur_name, threshold=THRESHOLD, min_activation=MIN_ACTIVATION, batch_size=BATCH_SIZE, model_schedule='2')

        # save_models -> e.g.'RN34_UN_ep20_bs4_lr-4'
        file_name = '{}_ep{}_bs{}_lr{}'.format(cur_name, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE)
#         save_model(seg_model, file_name)
#         print("model saved")
        
        trained_model.append(seg_model)
        
        print("finish segmentation testing")

    # re-create val & test loaders for cascade model with DROP_LAST = False
    val_loader, test_loader = create_loader(RESOLUTION, model_type, 
                                            BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DROP_LAST=False)
    
    return trained_model, val_loader, test_loader


def run_cas(model_type, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, THRESHOLD, MIN_ACTIVATION, RESOLUTION, NUM_WORKERS, PIN_MEMORY, DROP_LAST):
    """
    Main function to train cascade models. Run all four combinations of encoder/classification models at once.
    """
    # Data Preprocessing
    decode_mask("test/testdata/Pneumothorax_reports_small.csv", "test/testdata/masks/")
    generate_four_csv("test/testdata/Pneumothorax_reports_small.csv")
    
    # run both seg models
    seg_models, val_loader, test_loader = run_seg("segmentation", NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, THRESHOLD, MIN_ACTIVATION, RESOLUTION, NUM_WORKERS, PIN_MEMORY, DROP_LAST)
    seg_names = np.array(['RN34_UN', 'EB3_UN'])
    loader_types = np.array(['train', 'validation', 'test'])
    
    for seg_idx in np.arange(len(seg_models)): 
        cur_model = seg_models[seg_idx]
        cur_name = seg_names[seg_idx]
        cur_prev = cur_name + "_"
        
        # save intermediate imgs with both models
        for loader_type in loader_types:      
            save_imgs_based_on_model(cur_model, val_loader, test_loader, loader_type=loader_type, model_name=cur_name, resolution=RESOLUTION, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
        
        # run both classification models on both seg intermediate imgs
        run_class(model_type, cur_prev, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, THRESHOLD, MIN_ACTIVATION, RESOLUTION, NUM_WORKERS, PIN_MEMORY, DROP_LAST)
        
        return