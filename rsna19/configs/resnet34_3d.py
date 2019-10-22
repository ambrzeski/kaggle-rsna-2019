from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    train_out_dir = '/kolos/m2/ct/models/classification/rsna_3d/0001/0_1_2_3'

    dataset_file = '5fold.csv'
    data_version = '3d'  # '3d', 'npy', 'npy256' etc.
    train_folds = [0, 1, 2, 3]
    val_folds = [4]

    backbone = 'resnet34'
    pretrained = '/kolos/m2/ct/models/classification/rsna/pretrain/resnet_34.pth'
    resnet_shortcut = 'A'   # 'A' or 'B'
    new_layer_names = ['fc']

    lr = 1e-4
    new_params_lr_boost = 1e2
    batch_size = 16  # 16 (3, 512, 512) images fits on TITAN XP
    dropout = 0.5
    weight_decay = 0.001
    optimizer = 'radam'

    gpus = [2]
    num_workers = 3 * len(gpus)

    max_epoch = 20

    num_slices = 15  # must be odd
    pre_crop_size = 400
    crop_size = 384
    random_crop = True
    use_cdf = True
    augment = True

    # used only if use_cdf is False
    min_hu_value = 20
    max_hu_value = 100

    balancing = False
    # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', no_bleeding
    probas = [0.1, 0.14, 0.14, 0.14, 0.14, 0.34]
