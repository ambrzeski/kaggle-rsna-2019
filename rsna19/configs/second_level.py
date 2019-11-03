from rsna19.configs.base_config import BaseConfig


class Config(BaseConfig):
    class_weights = [1, 1, 1, 1, 1, 2]
    seg_areas_path = '/andrew/ssd/rsna/second_level/seg_areas.csv'
    prediction_paths = {
        4: '/andrew/ssd/rsna/second_level/0014_384/0123/results/_ckpt_epoch_2_val.csv',
        3: '/andrew/ssd/rsna/second_level/0014_384/0124/results/_ckpt_epoch_2_val.csv',
        2: '/andrew/ssd/rsna/second_level/0014_384/0134/results/_ckpt_epoch_4_val.csv',
        1: '/andrew/ssd/rsna/second_level/0014_384/0234/results/_ckpt_epoch_4_val.csv',
        # 0:'/andrew/ssd/rsna/second_level/0014_384/1234/results/_ckpt_epoch_4_val.csv'
    }

    gt_columns = ['gt_epidural', 'gt_intraparenchymal', 'gt_intraventricular',
                  'gt_subarachnoid', 'gt_subdural', 'gt_any']
    pred_columns = ['pred_epidural', 'pred_intraparenchymal', 'pred_intraventricular',
                    'pred_subarachnoid', 'pred_subdural', 'pred_any']

    num_slices = 5
    predictions_in = num_slices * len(pred_columns)
    append_area_feature = False

    features_out = 6
    hidden = 128
    n_epochs = 10000
    lr = 0.008

    sklearn_loss = False

