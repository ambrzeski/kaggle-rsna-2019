import json


def get_train_folds(val_folds):
    return list({0, 1, 2, 3, 4} - set(val_folds))


def get_val_folds_str(val_folds):
    return "".join([str(i) for i in val_folds])


def load(config_path):
    dct = json.load(open(config_path))
    if 'dataset_file' in dct:
        dct['train_dataset_file'] = dct['dataset_file']
        dct['val_dataset_file'] = dct['dataset_file']
        dct['test_dataset_file'] = 'test.csv'
    config = type('', (), dct)()
    return config
