# Kaggle RSNA 2019

## Before using the code

Project source is expected to operate on a converted form of challenge data, involving e.g. changing directory structure and image format. The new directory structure generated below will however create symlinks to the original data, so the original data should be kept in original location after completing data conversion process.

First you need to download the Kaggle data, including: stage_1_train_images, stage_1_test_images, and stage_1_train.csv. After downloading the data, open project config file located at:

```rsna19/config.py```

```
class config:
    train_dir = '/kolos/storage/ct/data/rsna/stage_1_train_images'
    test_dir = '/kolos/storage/ct/data/rsna/stage_1_test_images'
    labels_path = "/kolos/storage/ct/data/rsna/stage_1_train.csv"

    data_root = "/kolos/m2/ct/data/rsna/"
```

Modify train_dir, test_dir and labels_path variables to make them point to appropriate data paths on your hard drive. Also, modify the data_root variable to indicate output directory, where the converted data should be saved.

Next, three scripts be should executed to perform the full process of conversion. The scripts can be run right away if you open the project in PyCharm. Otherwise you may need add project package to PYTHONPATH:

```export PYTHONPATH="$PYTHONPATH:/{path}/{to}/kaggle-rsna-2019/rsna19/"```

Finally you can run the three scripts (please keep the order):

```
# Generate data index and save it as .pkl
$ python rsna19/data/scripts/create_dataframe.py   


# Create new directory structure and symlinks to original dicoms
$ python rsna19/data/scripts/create_symlinks.py


# Export dicom images (slices) to:
#   - npy arrays - for faster loading durng training (>3x faster)
#   - png images - for easier viewing and browsing the images
$ python rsna19/data/scripts/convert_dataset.py
```

Now you can safely use project's pytorch data loader class for your training (rsna19.data.dataset.IntracranialDataset). 
