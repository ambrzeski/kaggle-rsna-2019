from pathlib import Path
from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    class_weights = [1, 1, 1, 1, 1, 2]
    models_root = Path("/kolos/m2/ct/models/classification/rsna-ready")
    seg_areas_path = models_root / 'seg_areas.csv'

    gt_columns = ['gt_epidural', 'gt_intraparenchymal', 'gt_intraventricular',
                  'gt_subarachnoid', 'gt_subdural', 'gt_any']
    pred_columns = ['pred_epidural', 'pred_intraparenchymal', 'pred_intraventricular',
                    'pred_subarachnoid', 'pred_subdural', 'pred_any']

    num_slices = 5
    predictions_in = num_slices * len(pred_columns)
    append_area_feature = False

    features_out = 6
    hidden = 128
    n_epochs = 2000
    lr = 0.008

    sklearn_loss = False

    fold = "fold0"
    cache_dir = Path('/kolos/m2/ct/models/classification/rsna-cache2/v01/') / fold
    cache_dir = cache_dir if not append_area_feature else Path(str(cache_dir) + "_area")

    train_x = cache_dir / 'train_x.npy'
    train_y = cache_dir / 'train_y.npy'
    val_x = cache_dir / 'val_x.npy'
    val_y = cache_dir / 'val_y.npy'

    # TODO for stage 2: enable/list all models trained as listed in readme
    models = [
        "0036_3x3_pretrained",
        "0038_7s_res50_400",
        # "dpn68_384_5_planes_combine_last",
        # "resnet18_400",
        # "resnet18_384_5_planes_bn_f8",
        # "resnet34_400_5_planes_combine_last_var_dr0",
        "stage2_0036_3x3_5_slices_pretrained",
        "stage2_0036_3x3_pretrained",
        "stage2_airnet50_384",
        "stage2_dpn68_384_5_planes_combine_last",
        "stage2_resnet18_384_5_planes_bn_f8",
        "stage2_resnet18_400",
        "stage2_resnet34_400_5_planes_combine_last_var_dr0",
        "stage2_resnext50_400",
        "stage2_se_preresnext26b_400"
    ]
