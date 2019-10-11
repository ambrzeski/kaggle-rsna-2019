from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    train_out_dir = '/kolos/m2/ct/models/classification/rsna/0002_balancing_sampler/1_2_3_4'

    dataset_file = '5fold.csv'
    train_folds = [1, 2, 3, 4]
    val_folds = [0]

    backbone = 'se_resnext50'
    pretrained = True

    lr = 0.0001
    batch_size = 64  # 16 (3, 512, 512) images fits on TITAN XP

    gpus = [0]
    num_workers = 3 * len(gpus)

    max_epoch = 20

    num_slices = 3  # must be odd
    slice_size = 256
    use_cdf = False

    # used only if use_cdf is False
    min_hu_value = -1000
    max_hu_value = 1000

    # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', no_bleeding
    probas = [0.1, 0.14, 0.14, 0.14, 0.14, 0.34]

