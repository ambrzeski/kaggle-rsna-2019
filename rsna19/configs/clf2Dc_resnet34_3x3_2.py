import glob

from rsna19.configs import get_train_folds, get_val_folds_str
from rsna19.configs.clf2Dc_resnet34_3x3 import Config as ParentConfig


class Config(ParentConfig):
    train_out_dir = ParentConfig.model_outdir + '/0036_3x3_pretrained_stage2'

    train_dataset_file = '5fold-test.csv'
    val_dataset_file = '5fold-test.csv'
    test_dataset_file = 'test2.csv'
    data_version = '3d'  # '3d', 'npy', 'npy256' etc.
    use_cq500 = False
    gpus = [0]
    val_folds = [0]
    train_folds = get_train_folds(val_folds)
    folds_str = '/fold' + get_val_folds_str(val_folds)
    train_out_dir += folds_str

    backbone = 'resnet34'

    # 'imagenet', None or path to weights
    # pretrained = 'imagenet'
    pretrained = ParentConfig.model_outdir + f'/0034_resnet34_3c/{folds_str}/models/*'
    pretrained = sorted(glob.glob(pretrained))[-1]
