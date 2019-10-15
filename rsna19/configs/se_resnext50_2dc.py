from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    train_out_dir = '/kolos/m2/ct/models/classification/rsna/0012_flat_anneal_4/0_1_2_3'

    dataset_file = '5fold.csv'
    train_folds = [0, 1, 2, 3]
    val_folds = [4]

    backbone = 'se_resnext50'
    pretrained = True

    lr = 1e-4
    batch_size = 64  # 16 (3, 512, 512) images fits on TITAN XP
    dropout = 0.5
    weight_decay = 0.001
    optimizer = 'radam'

    scheduler = {
        'name': 'flat_anneal',
        'flat_iterations': 8000,
        'anneal_iterations': 17000,
        'min_lr': 1e-7
    }

    gpus = [1]
    num_workers = 3 * len(gpus)

    max_epoch = 20

    num_slices = 3  # must be odd
    slice_size = 256
    use_cdf = True
    augment = True

    # used only if use_cdf is False
    min_hu_value = 20
    max_hu_value = 100

    balancing = False
    # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', no_bleeding
    probas = [0.1, 0.14, 0.14, 0.14, 0.14, 0.34]
