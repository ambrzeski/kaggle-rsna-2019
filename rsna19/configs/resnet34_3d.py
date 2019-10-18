from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    train_out_dir = '/kolos/m2/ct/models/classification/rsna/0013_multiple_input_channels_5_v2/0_1_2_3'

    dataset_file = '5fold.csv'
    train_folds = [0, 1, 2, 3]
    val_folds = [4]

    backbone = 'resnet34'
    pretrained = True

    lr = 0.0001
    batch_size = 64  # 16 (3, 512, 512) images fits on TITAN XP
    dropout = 0.5
    weight_decay = 0.01
    optimizer = 'radam'

    gpus = [3]
    num_workers = 3 * len(gpus)

    max_epoch = 20

    num_slices = 5  # must be odd
    slice_size = 256
    use_cdf = True
    augment = True

    # used only if use_cdf is False
    min_hu_value = 20
    max_hu_value = 100

    balancing = False
    # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', no_bleeding
    probas = [0.1, 0.14, 0.14, 0.14, 0.14, 0.34]

