from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    train_out_dir = '/kolos/m2/ct/models/classification/rsna/'

    dataset_file = '5fold.csv'
    train_folds = [0, 1, 2, 3]
    val_folds = [4]

    backbone = 'se_resnext50'
    pretrained = True

    lr = 0.0001
    batch_size = 64   # 16 (3, 512, 512) images fits on TITAN XP

    gpus = [0, 1, 2, 3]
    num_workers = 3 * len(gpus)

    max_epoch = 20

    min_hu_value = -1000
    max_hu_value = 1000

