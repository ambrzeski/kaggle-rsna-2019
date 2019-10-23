from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    train_out_dir = '/kolos/m2/ct/models/classification/rsna/test4'

    dataset_file = '5fold.csv'
    data_version = '3d'  # '3d', 'npy', 'npy256' etc.
    train_folds = [0, 1, 2, 3]
    val_folds = [4]

    backbone = 'se_resnext50'

    # 'imagenet', None or path to weights
    # pretrained = 'imagenet'
    pretrained = '/kolos/m2/ct/models/classification/rsna/0014_384/0123/models/_ckpt_epoch_2.ckpt'

    lr = 1e-4
    batch_size = 26  # 16 (3, 512, 512) images fits on TITAN XP
    dropout = 0.5
    weight_decay = 0.001
    optimizer = 'radam'

    scheduler = {
        'name': 'flat_anneal',
        'flat_iterations': 8000,
        'anneal_iterations': 17000,
        'min_lr': 1e-7
    }

    gpus = [0]
    num_workers = 3 * len(gpus)

    max_epoch = 20

    num_slices = 3  # must be odd
    pre_crop_size = 400
    crop_size = 384
    random_crop = True
    vertical_flip = True
    pixel_augment = False
    elastic_transform = True
    use_cdf = True
    augment = True

    # used only if use_cdf is False
    min_hu_value = 20
    max_hu_value = 100

    balancing = False
    # 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', no_bleeding
    probas = [0.1, 0.14, 0.14, 0.14, 0.14, 0.34]

    multibranch = True
    multibranch_embedding = 256
