import glob

from rsna19.configs import get_train_folds, get_val_folds_str
from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    train_out_dir = BaseConfig.model_outdir + '/seg0003_ours_any_iou'

    train_dataset_file = '5fold.csv'
    val_dataset_file = '5fold.csv'
    test_dataset_file = 'test.csv'
    data_version = '3d'  # '3d', 'npy', 'npy256' etc.
    val_folds = [0]
    train_folds = get_train_folds(val_folds)
    folds_str = '/fold' + get_val_folds_str(val_folds)
    train_out_dir += folds_str

    backbone = 'se_resnext50_32x4d'

    # 'imagenet', None or path to weights
    # pretrained = 'imagenet'
    pretrained = BaseConfig.model_outdir + f'/0014_384/{folds_str}/models/*'
    pretrained = sorted(glob.glob(pretrained))[-1]

    lr = 2e-4
    decoder_lr = 2e-4
    encoder_lr = 8e-6

    batch_size = 14

    weight_decay = 0.001
    optimizer = 'radam'

    scheduler = {
        'name': 'flat_anneal',
        'flat_iterations': 3000,
        'anneal_iterations': 7000,
        'min_lr': 1e-6
    }

    gpus = [0]
    num_workers = 3 * len(gpus)

    max_epoch = 60

    negative_data_steps = [2000, 4500, 7000]
    # negative_data_steps = None

    num_slices = 3  # must be odd
    pre_crop_size = 400
    train_image_size = 448

    crop_size = 384
    random_crop = False
    center_crop = False

    shift_value = 0.04
    vertical_flip = False
    pixel_augment = False
    elastic_transform = False
    use_cdf = True
    augment = True

    # used only if use_cdf is False
    min_hu_value = 20
    max_hu_value = 100

