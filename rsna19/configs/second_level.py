from pathlib import Path
from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    class_weights = [1, 1, 1, 1, 1, 2]
    models_root = Path("/kolos/m2/ct/models/classification/rsna-ready2")
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
    cache_dir = Path('/kolos/m2/ct/models/classification/rsna-cache2/v01b/') / fold
    cache_dir = cache_dir if not append_area_feature else Path(str(cache_dir) + "_area")

    train_x = cache_dir / 'train_x.npy'
    train_y = cache_dir / 'train_y.npy'
    val_x = cache_dir / 'val_x.npy'
    val_y = cache_dir / 'val_y.npy'

    models = [
        "0036_3x3_pretrained",
        "0038_7s_res50_400",
        "dpn68_384_5_planes_combine_last",
        "resnet18_400",
        "resnet34_400_5_planes_combine_last_var_dr0",
        "resnet18_384_5_planes_bn_f8",
        "airnet50_384",
        "0036_3x3_5_slices_pretrained",
        "0036_3x3_pretrained_stage2",
        "resnext50_400",
        "se_preresnext26b_400"
    ]
