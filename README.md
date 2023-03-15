# DSC-180B-Project
**Project Topic:** classification of penumothorax dataset [CANDID-PTX](https://pubs.rsna.org/doi/10.1148/ryai.2021210136) using classification models, segmentation models, and cascade models.

GROUP NAME: AC/DS :metal: (Angela + Cecilia -> AC, Data Science -> DS, AC/DC -> AC/DS) :fist_right::fist_left:

**Brave Angela Not Afraid of Error :partying_face:**

**Be Calm and Write Code Cecilia :innocent:**

## Goal :pray:
- [x] Birthday (Angela) :birthday:
- [x] Classification Final Results
- [x] Segmentation Final Results
- [x] Cascade Final Results
- [x] Poster [03/09 DDL]
- [x] Website/Report/Code [03/14 DDL]
- [ ] Presentation/Birthday (Cecilia) :birthday: [03/15 DDL]

## Website
If you just want to have an idea of what this project is about without seeing all these codes (which I understand :stuck_out_tongue_winking_eye:), click here :point_right: https://angela-wang111.github.io/Pneumothorax_classification/

## Documentation
The useful files for checkpoint testing phase are: run.py, submission.json, config.json, src(folder), test(folder), and outpout(folder). 

:heavy_exclamation_mark:Execution instruction (in terminal):
1. `ssh <username>@dsmlp-login.ucsd.edu`
2. `launch.sh -i angela010101/pneumothorax:latest -c 8 -m 64 -g 1` at least 1 GPU and 64 GB memory is needed
3. `git clone https://github.com/Angela-Wang111/Pneumothorax_classification` (first time execution only)
4. `cd Pneumothorax_classification`
5. `python run.py <model type>` model type has to be one of "classification", "segmentation", or "cascade"
6. :crossed_fingers:
### submission.json
Contains the URLs for this Github Repository and the DockerHub Repository used for building a docker image for this pipeline.
### config.json
Contains the hyperparameters used for model training.
### run.py
This is the main .py file for executing the whole pipeline from data preprocessing to training and test classification models, segmentation models, cascade models. To run it, just run `python run.py <model type>` in the terminal (and hope everything goes fine :crossed_fingers: ). Model type has to be in the following three formats: "classification", "segmentation", "cascade" (all in lowercase). Example of the full terminal command: `python run.py classification`.
### src
#### data_preprocessing.py
This file contains functions to decode the RLE encoded pixels from the source "test/testdata/Pneumothorax_reports_small.csv" file and to save both positive and negative masks into test/testdata/masks, so they could be used for the segmentation model training/testing.
#### generate_train_val_test_csv.py
This file contains functions to generate "test/testdata/train.csv", "test/testdata/train_pos.csv", "test/testdata/train_neg.csv", "test/testdata/validation.csv", and "test/testdata/test.csv" for model training/validation/test.
#### create_dataloader.py
This file contains functions to create dataframe from .csv file like "test/testdata/validation.csv", create custermized Dataset, and create DataLoader. The function to create customized Dataset is modified to read .png formatted original images. The full version code is written to read DICOM format images stored in the team group folder.
#### build_model.py
This file contains functions to build customized pytorch pretrained ResNet 34 model and pretrained EfficientNet-B3 model, and train/validate classification models & segmentation models & cascade models.
#### evaluate_test.py
This file contains fucntions to plot, print, and save metrics based on the test set to evaluate all models.
#### save_model_imgs.py
This file contains functions to save predicted masks from pre-trained segmentation models. Mainly used for preparing images to be input to the classification models during the cascade model training.
#### run_model.py
This file contains functions to execute the entire pipeline to run classification, segmentation, and cascade models. It automatically run all models within the same structure.
- We currently disabled save_model() function to avoid saving large files on github. To enable it, please uncomment the lines `save_model(cla_model, file_name)` in the run_class() function and `save_model(seg_model, file_name)` in the run_seg() function.

### test
All the data here are just a small portion of CANDID-PTX (100/19237) for pipeline testing purpose only since the original data size is ~30GB.
*All empty folders currently store the outputs after test trials.*
#### testdata
- *Pneumothorax_reports_small.csv*: source test .csv file. Includes 100 penumothorax samples (15 positive, 85 negative) with **SOPInstanceUID** to identify each sample, and **EncodedPixels** to specify the penumothorax region (in RLE encoded format if positive, -1 if negative).
- *images*: folder contains the original X-Ray images (1024x1024). Named in the format "\<SOPInstanceUID>.png". The original images are in DICOM format, but are changed to .png format for the pipeline testing purpose. 
- *masks*: empty folder to store the decoded binary masks.
- *intermediate_data*: empty folder to store the intermediate images generated by the segmentation models. These images will be used as input for classification models in the cascade structure.

After runing the pipeline, the following files would be created :point_down:
- *masks*: folder contains the decoded binary masks (1024x1024). Named in the format "\<SOPInstanceUID>.png" if positive, "negative_mask.png" if negative.
- *train.csv*: training set with 80 samples (12 postive, 68 negative).
- *train_pos.csv*: all positive samples in the training set
- *train_neg.csv*: all negative samples in the training set
- *validation.csv*: validation set with 10 samples (2 positive, 8 negative).
- *test.csv*: test set with 10 samples (1 positive, 9 negative).

#### saved_model
This is the folder where saved models will be located if the function is enabled.

### output
This should be an empty folder before executing the pipeline. After executing the pipeline, the following metrics plots will be created :point_down:
- auc-roc plot inside *auc-roc* folder.
- train/val losses inside *both_loss* folder.
- confusion matrix inside *confusion_matrix* folder.
