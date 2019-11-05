import glob

from rsna19.configs import get_train_folds, get_val_folds_str
from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    train_out_dir = BaseConfig.model_outdir + '/0033_2Dc_5slices'

    train_dataset_file = '5fold-rev3.csv'
    val_dataset_file = '5fold.csv'
    test_dataset_file = 'test.csv'
    data_version = '3d'  # '3d', 'npy', 'npy256' etc.
    use_cq500 = False
    val_folds = [0]
    train_folds = get_train_folds(val_folds)
    folds_str = '/fold' + get_val_folds_str(val_folds)
    train_out_dir += folds_str

    backbone = 'se_resnext50'

    # 'imagenet', None or path to weights
    # pretrained = 'imagenet'
    pretrained = BaseConfig.model_outdir + f'/0014_384/{folds_str}/models/*'
    pretrained = sorted(glob.glob(pretrained))[-1]

    lr = 1e-4
    batch_size = 24  # 16 (3, 512, 512) images fits on TITAN XP
    accumulate_grad_batches = 1
    dropout = 0.5
    weight_decay = 0.001
    optimizer = 'radam'

    scheduler = {
        'name': 'flat_anneal',
        'flat_iterations': 16000,
        'anneal_iterations': 30000,
        'min_lr': 1e-7
    }

    freeze_backbone_iterations = 0

    gpus = [1]
    num_workers = 3 * len(gpus)

    max_epoch = 20

    append_masks = False
    num_slices = 5  # must be odd
    pre_crop_size = 400
    padded_size = None
    crop_size = 384
    shift_limit = 0.1
    random_crop = False
    vertical_flip = False
    pixel_augment = False
    elastic_transform = False
    use_cdf = True
    augment = True

    # used only if use_cdf is False
    min_hu_value = 20
    max_hu_value = 100

    balancing = False
    # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', no_bleeding
    probas = [0.1, 0.14, 0.14, 0.14, 0.14, 0.34]

    multibranch = False

    # use 3d conv to merge features from different branches
    multibranch3d = False
    multibranch_embedding = 256

    # number of input channels to each branch
    multibranch_input_channels = 3
    num_branches = 3

    # None, or list of slices' indices ordering them before splitting between branches,
    # length must be equal to num_branches * multibranch_input_channels
    multibranch_channel_indices = None
    # multibranch_channel_indices = [0, 1, 2, 1, 2, 3, 2, 3, 4]

    contextual_attention = False
    spatial_attention = False