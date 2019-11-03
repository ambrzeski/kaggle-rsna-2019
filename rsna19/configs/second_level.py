from pathlib import Path
from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    class_weights = [1, 1, 1, 1, 1, 2]
    models_root = Path("/kolos/m2/ct/models/classification/rsna-ready")
    seg_areas_path = models_root / 'seg_areas.csv'
    prediction_paths = {
        4: models_root / '0036_3x3_pretrained/fold4/predictions/val_0.csv',
        3: models_root / '0036_3x3_pretrained/fold3/predictions/val_0.csv',
        2: models_root / '0036_3x3_pretrained/fold2/predictions/val_0.csv',
        1: models_root / '0036_3x3_pretrained/fold1/predictions/val_0.csv',
        0: models_root / '0036_3x3_pretrained/fold0/predictions/val_0.csv'
    }

    gt_columns = ['gt_epidural', 'gt_intraparenchymal', 'gt_intraventricular',
                  'gt_subarachnoid', 'gt_subdural', 'gt_any']
    pred_columns = ['pred_epidural', 'pred_intraparenchymal', 'pred_intraventricular',
                    'pred_subarachnoid', 'pred_subdural', 'pred_any']

    num_slices = 5
    predictions_in = num_slices * len(pred_columns)
    append_area_feature = True

    features_out = 6
    hidden = 128
    n_epochs = 2000
    lr = 0.008

    sklearn_loss = False

    cache_dir = Path('cache')
    cache_dir = cache_dir if not append_area_feature else Path(str(cache_dir) + "_area")

    train_x = cache_dir / 'train_x.npy'
    train_y = cache_dir / 'train_y.npy'
    val_x = cache_dir / 'val_x.npy'
    val_y = cache_dir / 'val_y.npy'
