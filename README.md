# Kaggle RSNA 2019 - 8th place solution

Solution overview: <kaggle-discussion-link>

Reproduction instructions:
* [Data conversion](#data-conversion)
* [Dataset files](#dataset-files)
* [Training models 1/2 (Dmytro's models)](#train-dmytro)
* [Training models 2/2 (BrainScan models)](#train-brainscan)
* [Generating predictions for challenge data 1/2 (Dmytro's models)](#predict-dmytro)
* [Generating predictions for challenge data 2/2 (BrainScan models)](#predict-brainscan)
* [Second level model and generating final predictions](#second-level-model-and-generating-final-predictions)
* [Testing on new data](#testing-on-new-data)


## Data conversion

Project source is expected to operate on a converted form of challenge data, involving e.g. changing directory structure and image format. The new directory structure generated below will however create symlinks to the original data, so the original data should be kept in original location after completing data conversion process.

First you need to download the Kaggle data, including: stage_1_train_images, stage_1_test_images, stage_2_test_images, stage_1_train.csv and stage_2_train.csv. After downloading the data, open project config file located at rsna19/config.py:


```python
class config:
    train_dir = '/kolos/storage/ct/data/rsna/stage_1_train_images'
    test_dir = '/kolos/storage/ct/data/rsna/stage_1_test_images'
    labels_path = "/kolos/storage/ct/data/rsna/stage_1_train.csv"

    data_root = "/kolos/m2/ct/data/rsna/"
    
    # Used for Dmytro's models
    checkpoints_dir = "../output/checkpoints"
    tensorboard_dir = "../output/tensorboard"
    oof_dir = "../output/oof"

    # Used for Brainscan models
    model_outdir = '/kolos/m2/ct/models/classification/rsna/'
```

Modify train_dir, test_dir and labels_path variables to make them point to appropriate data paths on your hard drive. Also, modify the data_root variable to indicate output directory, where the converted data should be saved. Finally, set models output paths to desired locations, which will be used later during trainings.

Next, three scripts should be executed to perform the full process of conversion. The scripts can be run right away if you open the project in PyCharm. Otherwise you may need add project package to PYTHONPATH:

```export PYTHONPATH="$PYTHONPATH:/{path}/{to}/kaggle-rsna-2019/rsna19/"```

Finally you can run the three scripts (please keep the order):

```
# Generate data index and save it as .pkl
$ python rsna19/data/scripts/create_dataframe.py   

# Create new directory structure and symlinks to original dicoms
$ python rsna19/data/scripts/create_symlinks.py

# Convert dicom images (slices) to npy arrays and pngs images
$ python rsna19/data/scripts/convert_dataset.py
$ python rsna19/data/scripts/prepare_3d_data.py
```

As a result of the conversion, for each examination a set of subdirs will be created:

* /dicom - original dicom files
* /png - png images with drawn labels for easier viewing and browsing
* /npy - slices saved as numpy arrays for faster loading during training (>3x faster)
* /3d - transformed slices that are used for actual trainings, transforms include fixing scan gantry tilt and 400x400 crop in x and y dimensions around volume center of mass


## Dataset files

In rsna19/data/csv directory you can find a set .csv dataset files defining train/val/test splits, cross-validation splits and samples labels. The dataset files were generated using rsna19/data/notebooks/generate_folds.ipynb notebook. The most significant dataset files are:

* 5fold.csv - stage 1 training data sample split into 5 folds
* 5folds-rev3.csv - same as above, but labels for 196 are modified basing on our manual annotation of scans
* 5fold-test.csv - stage 2 training data (stage 1 extended with stage 1 test data), split into 5 fold - unfortunately, as it turned out later, patient leaks between folds occured in this split
* 5folds-test-rev3.csv - same as above, but labels for 196 are modified basing on our manual annotation of scans


<a name="train-dmytro"></a>
## Training models 1/2 (Dmytro's models)

For training stage 1 models run the following commands for folds 0-4:

```
$ python models/clf2D/train.py train --model resnet18_400 --fold 0
$ python models/clf2D/train.py train --model resnet34_400_5_planes_combine_last_var_dr0 --fold 0
$ python models/clf2D/train_segmentation.py train --model resnet18_384_5_planes_bn_f8 --fold 0
$ python models/clf3D/train_3d.py train --model dpn68_384_5_planes_combine_last --fold 0
$ python models/clf2D/train.py train --model airnet50_384 --fold 0 --apex
```

For stage 2 models run the following for folds 0-4:

```
$ python models/clf2D/train.py train --model se_preresnext26b_400 --fold 0 --apex
$ python models/clf2D/train.py train --model resnext50_400 --fold 0 --apex
```


<a name="train-brainscan"></a>
## Training models 2/2 (BrainScan models)

First you need to train baseline models that are used for initiating weights in final trainings. Trainings are conducted by running rsna19/models/clf2Dc/train.py script with appropriate config imported at the top of the file instead of the default place of ‘clf2Dc’. For example, to train train 'clf2Dc_resnet34_3c' config, change:

```python
from rsna19.configs.clf2Dc import Config
```

to:

```python
from rsna19.configs.clf2Dc_resnet34_3c import Config
```
 
Also each training must be repeated 5 times with different ‘val_folds’ attributes (from [0] to [4]), modified in appropriate config files. You can also change gpu used for training using ‘gpu’ attribute.

Configs to train for baseline models:
* clf2Dc_resnet34_3c.py
* clf2Dc_resnet50_3c_384.py

Then final models that we trained for stage 1 can be trained using configs:
* clf2Dc_resnet34_3x3.py
* clf2Dc_resnet50_7c_400.py

In stage 2 we trained to additional models (make sure to set 5fold-test.csv for both 'train_dataset_file' and 'val_dataset_file' in config files):
* clf2Dc_resnet34_3x3_5_slices.py
* clf2Dc_resnet34_3x3.py


<a name="predict-dmytro"></a>
## Generating predictions for challenge data 1/2 (Dmytro's models)

Run the following set of commands for folds 0-4:

```
# Stage 1 models out-of-fold predictions
$ python models/clf2D/predict.py predict_oof --model resnet18_400 --epoch 6 --fold 0 --mode all
$ python models/clf2D/predict.py predict_oof --model resnet34_400_5_planes_combine_last_var_dr0 --epoch 7 --fold 0 --mode all
$ python models/clf3D/predict.py predict_oof --model dpn68_384_5_planes_combine_last --epoch 72 --fold 0 --mode all
$ python models/clf2D/predict.py predict_oof --model resnet18_384_5_planes_bn_f8 --epoch 6 --fold 0 --mode all
$ python models/clf2D/predict.py predict_oof --model airnet50_384 --epoch 6 --fold 0 --mode all

# Stage 1 models test predictions
$ python models/clf2D/predict.py predict_test --model resnet18_400 --epoch 6 --fold 0 --mode all
$ python models/clf2D/predict.py predict_test --model resnet34_400_5_planes_combine_last_var_dr0 --epoch 7 --fold 0 --mode all
$ python models/clf3D/predict.py predict_test --model dpn68_384_5_planes_combine_last --epoch 72 --fold 0 --mode all
$ python models/clf2D/predict.py predict_test --model resnet18_384_5_planes_bn_f8 --epoch 6 --fold 0 --mode all
$ python models/clf2D/predict.py predict_test --model airnet50_384 --epoch 6 --fold 0 --mode all

# Stage 2 models out-of-fold predictions
$ python models/clf2D/predict.py predict_oof --model se_preresnext26b_400 --epoch 6 --fold 0 --mode all
$ python models/clf2D/predict.py predict_oof --model resnext50_400 --epoch 6 --fold 0 --mode all

# Stage 2 models test predictions
$ python models/clf2D/predict.py predict_test --model se_preresnext26b_400 --epoch 6 --fold 0 --mode all
$ python models/clf2D/predict.py predict_test --model resnext50_400 --epoch 6 --fold 0 --mode all

```

<a name="predict-brainscan"></a>
## Generating predictions for challenge data 2/2 (BrainScan models)

Calculate model predictions including TTAs by running: 

```
$ python rsna19/models/clf2Dc/predict.py
```

## Second level model and generating final predictions

Make sure that all models are copied to common directory, so that the directory structure matches the following form: 

```
{model_outdir}/{model_name}/{fold}/predictions/{predictions.csv}
```
Set path to this directory in rsna19/configs/base_config.Config.model_outdir. Next, generate the dataset for the second level model by running the following script for each of 5 folds (fold name can be specified in kaggle-rsna-2019/rsna19/configs/second_level.py):

```
$ python rsna19/models/second_level/dataset2.py
```


## Testing on new data

Unfortunately, as of now we don't provide a script to generate predictions on new data directly. However, if you can save the new data in the same format as challenge data, you can use the instructions above to preprocess the data and run inference.

Specifically, you need take the following steps:
* set the path to the new test data directory in 'test_dir' in 'rsna19/config.py'
* continue with data conversion steps
* generate new test .csv file using rsna19/data/notebooks/generate_folds.ipynb notebook
* update test .csv path in appropriate prediction scripts first
* run the predictions steps using the new test .csv file 
