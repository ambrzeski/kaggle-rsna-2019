from models.clf3D import model_3d
from rsna19.models.clf2D.experiments import ModelInfo

MODELS = {
    'resnet34_400_5_planes_combine_last': ModelInfo(
        factory=model_3d.classification_model_resnet34_combine_last,
        use_3d=True,
        args=dict(combine_slices=5),
        dataset_args=dict(
            img_size=400,
            # center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=8,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=4
    ),
}
