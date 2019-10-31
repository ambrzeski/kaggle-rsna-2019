from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    train_out_dir = '/kolos/m2/ct/models/classification/rsna/0032_5_branch_new2/0123'

    dataset_file = '5fold.csv'
    data_version = '3d'  # '3d', 'npy', 'npy256' etc.
    train_folds = [0, 1, 2, 3]
    val_folds = [4]

    # backbone = 'se_resnext50'
    backbone = 'resnet34'

    # 'imagenet', None or path to weights
    pretrained = 'imagenet'
    # pretrained = '/kolos/m2/ct/models/classification/rsna/0014_384/0123/models/_ckpt_epoch_2.ckpt'

    lr = 1e-4
    batch_size = 24  # 16 (3, 512, 512) images fits on TITAN XP
    accumulate_grad_batches = 2
    dropout = 0.5
    weight_decay = 0.001
    optimizer = 'radam'

    # scheduler = {
    #     'name': 'flat_anneal',
    #     'flat_iterations': 12000,
    #     'anneal_iterations': 24000,
    #     'min_lr': 1e-7
    # }

    scheduler = {
        'name': 'LambdaLR',
        'iter_to_lr': {
            0: 1e-3,
            4000: 1e-4,
            14000: 2e-5,
            24000: 4e-6
        },
    }

    freeze_backbone_iterations = 4000
    freeze_first_layer = False

    gpus = [3]
    num_workers = 3 * len(gpus)

    max_epoch = 20

    num_slices = 5  # must be odd
    append_masks = False
    pre_crop_size = 400
    crop_size = 384
    random_crop = True
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

    multibranch = True

    # use 3d conv to merge features from different branches
    multibranch3d = False
    multibranch_embedding = 256

    # number of input channels to each branch
    multibranch_input_channels = 1
    num_branches = 5

    # None, or list of slices' indices ordering them before splitting between branches,
    # length must be equal to num_branches * multibranch_input_channels
    multibranch_channel_indices = None
    # multibranch_channel_indices = [0, 1, 2, 1, 2, 3, 2, 3, 4]

    contextual_attention = False
    spatial_attention = False

    use_cq500 = True
