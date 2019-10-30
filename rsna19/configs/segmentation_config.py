import glob
from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):

    dataset_file = '5fold.csv'
    data_version = '3d'  # '3d', 'npy', 'npy256' etc.
    val_folds = [2]
    train_folds = list({0, 1, 2, 3, 4} - set(val_folds))
    folds_str = "".join([str(i) for i in train_folds])

    train_out_dir = f'/kolos/m2/ct/models/classification/rsna/seg0001_ours/{folds_str}'

    backbone = 'se_resnext50_32x4d'

    # 'imagenet', None or path to weights
    # pretrained = 'imagenet'
    pretrained = f'/kolos/m2/ct/models/classification/rsna/0014_384/{folds_str}/models/*'
    pretrained = sorted(glob.glob(pretrained))[-1]

    lr = 1e-4
    decoder_lr = 1e-4
    encoder_lr = 5e-6

    batch_size = 16

    weight_decay = 0.001
    optimizer = 'radam'

    scheduler = {
        'name': 'flat_anneal',
        'flat_iterations': 1500,
        'anneal_iterations': 6000,
        'min_lr': 1e-6
    }

    gpus = [0]
    num_workers = 3 * len(gpus)

    max_epoch = 60

    num_slices = 3  # must be odd
    pre_crop_size = 400
    train_image_size = None

    crop_size = 384
    random_crop = False
    center_crop = False

    shift_pixel_value = 15
    vertical_flip = False
    pixel_augment = False
    elastic_transform = False
    use_cdf = True
    augment = True

    # used only if use_cdf is False
    min_hu_value = 20
    max_hu_value = 100

