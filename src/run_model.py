"""
import decode_mask
import generate_four_csv
import create_loader
import resnet34()
import eNet_b3()
import 
"""


def run_model(model_type):
    # Data Preprocessing
    decode_mask("test/testdata/Pneumothorax_reports_small.csv", "test/testdata/masks/")
    generate_four_csv("test/testdata/Pneumothorax_reports_small.csv")
    
    # generate val & test loaders
    val_loader, test_loader = create_loader(RESOLUTION, model_type=model_type, 
                                            BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, DROP_LAST)
    
    if model_type == "cla":
        # define rn34 & efficientnet-b3
        model_rn34 = resnet34()
        model_eb3 = eNet_b3()
        
        model_set = np.array([model_rn34, model_eb3])
        model_name_set = np.array(['RN34', 'EB3'])
        
        for model_idx in model_set.shape[0]:
            cla_model, train_loss, val_loss = training_class(model=model_set[model_idx], num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate = LEARNING_RATE, val_loader=val_loader, model_name=model_name_set[model_idx], model_type=model_type)

            
        # test metric
        # save_models
        
        
        
        
        
    elif model_type == "seg":
        
        
        
    else:
        
     