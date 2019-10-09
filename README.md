# Kaggle RSNA 2019

## Before using the code

Project source is expected to operate on a converted form of challenge data, involving e.g. changing directory structure and image format. The new directory structure generated below will however create symlinks to the original data, so the original data should be kept in original location after completing data conversion process.

First you need to download the Kaggle data, including: stage_1_train_images, stage_1_test_images, and stage_1_train.csv. After downloading the data, open project config file located at:


```
rsna19/config.py
----------------

class config:
    train_dir = '/kolos/storage/ct/data/rsna/stage_1_train_images'
    test_dir = '/kolos/storage/ct/data/rsna/stage_1_test_images'
    labels_path = "/kolos/storage/ct/data/rsna/stage_1_train.csv"

    data_root = "/kolos/m2/ct/data/rsna/"
```

Modify train_dir, test_dir and labels_path variables to make them point to appropriate data paths on your hard drive. Also, modify the data_root variable to indicate output directory, where the converted data should be saved.

Next, three scripts should be executed to perform the full process of conversion. The scripts can be run right away if you open the project in PyCharm. Otherwise you may need add project package to PYTHONPATH:

```export PYTHONPATH="$PYTHONPATH:/{path}/{to}/kaggle-rsna-2019/rsna19/"```

Finally you can run the three scripts (please keep the order):

```
# Generate data index and save it as .pkl
$ python rsna19/data/scripts/create_dataframe.py   

# Create new directory structure and symlinks to original dicoms
$ python rsna19/data/scripts/create_symlinks.py

# Export dicom images (slices) to:
#   - npy arrays - for faster loading during training (>3x faster)
#   - png images - for easier viewing and browsing the images
$ python rsna19/data/scripts/convert_dataset.py
```

Now you can safely use project's pytorch data loader class for your training. For example, if you want to read the data for 2D classification, using four of the first folds for training and the last one for validation, you can do:

```
train_data = IntracranialDataset('5fold.csv', folds=[0, 1, 2, 3], return_labels=True)
val_data = IntracranialDataset('5fold.csv', folds=[4], return_labels=True)
```

5fold.csv is a dataset file including all the training data, split into 5 folds. The file is located in rsna19/data/csv.


## Notes on diagnostic windows

* it seems to be a common belief (used also in the ResNeXt 32x8d kernel) that dicom images should be 'windowed' after loading using windowing metadata of the particular dicom image, as here:

```
img_min = window_center - window_width//2
img_max = window_center + window_width//2
img[img<img_min] = img_min
img[img>img_max] = img_max
```
* however window parameters from dicom metadata are just suggested window settings for given image based on CT reconstruction parameters, which are not optimized in any way for pathologies visible in the scan
* the window parameters are automatically ignored by radiologists when reading scans using professional dicom viewers
* moreover, window parameters in dicoms vary greatly throughout the dataset
* applying different windows to scans results in discarding normalized and scaled HU intensity values, which are meaningul for the diagnosis - e.g. water has 0 HU value and should have same intensity on all scans, while in this approach usually have different intensity values
* hence we should apply one fixed windowing method for all scans (optimized for enhancing hemorrhages) or not apply windowing at all
* the only reason why windowing is used in clinical practise are the limitations of human vision in distinguishing grayscale tones
* this limitation does not affect ConvNets, which should easily be able to find the optimal range of intensity values that are meaningful for detecting hemorrhages
* hence, for now we do not apply windowing at all, we just pass HU values in their full resolution (using 16-bit representations)
* we will run some experiments to see whether a unified, hemorrhage-oriented (and maybe non-linear) window might improve the model accuracy
