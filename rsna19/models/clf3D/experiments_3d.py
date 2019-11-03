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

    'resnet34_400_5_planes_combine_last_acc16': ModelInfo(
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
        accumulation_steps=16,
        clip_grad=4.0
    ),

    'resnet34_400_5_planes_combine_last_no_acc': ModelInfo(
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
        accumulation_steps=1,
        clip_grad=1.0
    ),

    'resnet34_400_5_planes_combine_l3': ModelInfo(
        factory=model_3d.classification_model_resnet34_combine_l3,
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

    'resnet34_384_5_planes_combine_l3_acc1': ModelInfo(
        factory=model_3d.classification_model_resnet34_combine_l3,
        use_3d=True,
        args=dict(combine_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=8,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=1
    ),

    'resnet34_384_5_planes_combine_l3_acc1_wso': ModelInfo(
        factory=model_3d.classification_model_resnet34_combine_l3,
        use_3d=True,
        args=dict(combine_slices=5, use_wso=True),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=False),
        batch_size=8,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=1
    ),

    'resnet34_384_5_planes_combine_l3_sgd_acc1': ModelInfo(
        factory=model_3d.classification_model_resnet34_combine_l3,
        use_3d=True,
        args=dict(combine_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=8,
        optimiser='sgd',
        initial_lr=1e-2,
        accumulation_steps=1
    ),

    'resnet34_384_5_planes_combine_l3_adabound_acc1': ModelInfo(
        factory=model_3d.classification_model_resnet34_combine_l3,
        use_3d=True,
        args=dict(combine_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=8,
        optimiser='adabound',
        initial_lr=1e-4,
        accumulation_steps=1
    ),

    'dpn68_384_5_planes_combine_last': ModelInfo(
        factory=model_3d.classification_model_dpn68_combine_last,
        use_3d=True,
        args=dict(combine_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=4,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=8
    ),
    'resnext50_384_5_planes_combine_last': ModelInfo(
        factory=model_3d.classification_model_resnext50_combine_last,
        use_3d=True,
        args=dict(combine_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=2,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=32
    ),

    'enet_b0_384_5_planes_combine_last': ModelInfo(
        factory=model_3d.classification_model_enet_b0_combine_last,
        use_3d=True,
        args=dict(combine_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=4,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=32
    ),

    'enet_b2_384_5_planes_combine_last': ModelInfo(
        factory=model_3d.classification_model_enet_b2_combine_last,
        use_3d=True,
        args=dict(combine_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=2,
        optimiser='adabound',
        initial_lr=2e-5,
        accumulation_steps=1
    ),

    'dpn68_384_5_planes_3d_dr0': ModelInfo(
        factory=model_3d.classification_model_dpn68_combine_last,
        use_3d=True,
        args=dict(combine_slices=5, dropout=0, combine_conv_features=128),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=4,
        optimiser='radam',
        initial_lr=2e-5,
        accumulation_steps=1
    ),

    'dpn68_384_5_planes_3d_dr0_cos': ModelInfo(
        factory=model_3d.classification_model_dpn68_combine_last,
        use_3d=True,
        args=dict(combine_slices=5, dropout=0, combine_conv_features=128),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=4,
        optimiser='radam',
        initial_lr=2e-5,
        scheduler='cos_restarts',
        accumulation_steps=1
    ),

    'airnet50_384_5_planes_3d_dr0': ModelInfo(
        factory=model_3d.classification_model_airnet50,
        use_3d=True,
        args=dict(combine_slices=5, dropout=0, combine_conv_features=128),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=8,
            combine_slices_padding=2,
            convert_cdf=True),
        batch_size=4,
        optimiser='radam',
        initial_lr=2e-5,
        accumulation_steps=1
    ),
}
