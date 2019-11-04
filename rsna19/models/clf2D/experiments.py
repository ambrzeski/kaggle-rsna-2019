from rsna19.models.clf2D import model_2d, model_2dc, model_2dc_segmentation
from models.clf3D import model_3d


class ModelInfo:
    def __init__(self,
                 factory,
                 args,
                 dataset_args,
                 batch_size,
                 nb_slices=1,
                 optimiser='sgd',
                 scheduler='steps',
                 initial_lr=1e-3,
                 optimiser_milestones=None,
                 accumulation_steps=1,
                 weight_decay=0,
                 is_pretrained=True,
                 use_3d=False,
                 clip_grad=1.0,
                 single_slice_steps=0,
                 freeze_bn_step=-1,
                 use_vflip=True
                 ):
        self.use_vflip = use_vflip
        self.freeze_bn_step = freeze_bn_step
        self.single_slice_steps = single_slice_steps
        self.clip_grad = clip_grad
        self.use_3d = use_3d
        self.nb_slices = nb_slices
        self.is_pretrained = is_pretrained
        self.weight_decay = weight_decay
        self.accumulation_steps = accumulation_steps
        self.optimiser_milestones = optimiser_milestones
        self.initial_lr = initial_lr
        self.scheduler = scheduler
        self.optimiser = optimiser
        self.factory = factory
        self.args = args
        self.dataset_args = dataset_args
        self.batch_size = batch_size


def _w(w, l):
    return l-w/2, l+w/2


MODELS = {
    'se_resnext50_gwap': ModelInfo(
        factory=model_2d.classification_model_se_resnext50_gwap,
        args=dict(),
        dataset_args=dict(),
        batch_size=12,
        optimiser='adam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'se_resnext50_gwap_256': ModelInfo(
        factory=model_2d.classification_model_se_resnext50_gwap,
        args=dict(),
        dataset_args=dict(img_size=256, scale_values=1e-3),
        batch_size=32,
        optimiser='adam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'se_resnext50_gwap_256_scaled': ModelInfo(
        factory=model_2d.classification_model_se_resnext50_gwap,
        args=dict(),
        dataset_args=dict(img_size=256, scale_values=1e-3),
        batch_size=32,
        optimiser='adam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'dpn92_gwap_256': ModelInfo(
        factory=model_2d.classification_model_dpn92,
        args=dict(use_gwap=True),
        dataset_args=dict(img_size=256, scale_values=1e-3),
        batch_size=32,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'dpn92_256': ModelInfo(
        factory=model_2d.classification_model_dpn92,
        args=dict(use_gwap=False),
        dataset_args=dict(img_size=256, scale_values=1e-3),
        batch_size=32,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'dpn68_gwap_256': ModelInfo(
        factory=model_2d.classification_model_dpn68b,
        args=dict(use_gwap=True),
        dataset_args=dict(img_size=256, scale_values=1e-3),
        batch_size=32,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'dpn68_256': ModelInfo(
        factory=model_2d.classification_model_dpn68b,
        args=dict(use_gwap=False),
        dataset_args=dict(img_size=256, scale_values=1e-3),
        batch_size=32,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'mobilenet_v2_gwap_256': ModelInfo(
        factory=model_2d.classification_model_mobilenet_v2,
        args=dict(use_gwap=True),
        dataset_args=dict(img_size=256, scale_values=1e-3),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1,
        optimiser_milestones=[8, 16, 24]
    ),
    'resnet34_256': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True),
        dataset_args=dict(img_size=256, scale_values=1e-3),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1,
        optimiser_milestones=[8, 16, 24]
    ),
    'resnet34_256_16_windows': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True, nb_windows_conv=16),
        dataset_args=dict(img_size=256, scale_values=1e-3),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1,
        optimiser_milestones=[8, 16, 24]
    ),
    'dpn68_256_64_windows': ModelInfo(
        factory=model_2d.classification_model_dpn68b,
        args=dict(use_gwap=True, nb_windows_conv=64),
        dataset_args=dict(img_size=256, scale_values=1e-3),
        batch_size=32,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'resnet34_256_16_window_0..100': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True, nb_windows_conv=32),
        dataset_args=dict(img_size=256, apply_windows=[(0, 100)]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'resnet34_256_16_window_-50..150': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True, nb_windows_conv=32),
        dataset_args=dict(img_size=256, apply_windows=[(-50, 150)]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'resnet34_256_16_window_-100..200': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True, nb_windows_conv=32),
        dataset_args=dict(img_size=256, apply_windows=[(-100, 200)]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'resnet34_256_16_window_0..50_20..100_-50..150': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True, nb_windows_conv=32, nb_input_planes=3),
        dataset_args=dict(img_size=256, apply_windows=[(0, 50), (20, 100), (-50, 150)]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'resnet34_256_32_window_set7': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True, nb_windows_conv=32, nb_input_planes=7),
        dataset_args=dict(img_size=256, apply_windows=[
            _w(w=80, l=40),
            _w(w=130, l=75),
            _w(w=300, l=75),
            _w(w=400, l=40),
            _w(w=2800, l=600),
            _w(w=8, l=32),
            _w(w=40, l=40)
        ]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'vgg16_256_window_0..50_20..100_-50..150': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True, nb_input_planes=3),
        dataset_args=dict(img_size=256, apply_windows=[(0, 50), (20, 100), (-50, 150)]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'vgg16_256_window_set7': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True, nb_input_planes=7),
        dataset_args=dict(img_size=256, apply_windows=[
            _w(w=80, l=40),
            _w(w=130, l=75),
            _w(w=300, l=75),
            _w(w=400, l=40),
            _w(w=2800, l=600),
            _w(w=8, l=32),
            _w(w=40, l=40)
        ]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'vgg16_128_window_set7': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True, nb_input_planes=7),
        dataset_args=dict(img_size=128, apply_windows=[
            _w(w=80, l=40),
            _w(w=130, l=75),
            _w(w=300, l=75),
            _w(w=400, l=40),
            _w(w=2800, l=600),
            _w(w=8, l=32),
            _w(w=40, l=40)
        ]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'vgg16_512_window_set7': ModelInfo(  # it's resnet34, nott vgg!
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=True, nb_input_planes=7),
        dataset_args=dict(img_size=512, apply_windows=[
            _w(w=80, l=40),
            _w(w=130, l=75),
            _w(w=300, l=75),
            _w(w=400, l=40),
            _w(w=2800, l=600),
            _w(w=8, l=32),
            _w(w=40, l=40)
        ]),
        batch_size=32,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=2
    ),
    'enet_b0_256_window_set7': ModelInfo(
        factory=model_2d.classification_model_efficient_net_b0,
        args=dict(use_gwap=True, nb_input_planes=7),
        dataset_args=dict(img_size=256, apply_windows=[
            _w(w=80, l=40),
            _w(w=130, l=75),
            _w(w=300, l=75),
            _w(w=400, l=40),
            _w(w=2800, l=600),
            _w(w=8, l=32),
            _w(w=40, l=40)
        ]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'now_vgg16_256_window_0..50_20..100_-50..150': ModelInfo(
        factory=model_2d.classification_model_vgg,
        args=dict(use_gwap=True, nb_input_planes=3),
        dataset_args=dict(img_size=256, apply_windows=[(0, 50), (20, 100), (-50, 150)]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'now_vgg16_256_window_set7': ModelInfo(
        factory=model_2d.classification_model_vgg,
        args=dict(use_gwap=True, nb_input_planes=7),
        dataset_args=dict(img_size=256, apply_windows=[
            _w(w=80, l=40),
            _w(w=130, l=75),
            _w(w=300, l=75),
            _w(w=400, l=40),
            _w(w=2800, l=600),
            _w(w=8, l=32),
            _w(w=40, l=40)
        ]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'now_vgg16_128_window_set7': ModelInfo(
        factory=model_2d.classification_model_vgg,
        args=dict(use_gwap=True, nb_input_planes=7),
        dataset_args=dict(img_size=128, apply_windows=[
            _w(w=80, l=40),
            _w(w=130, l=75),
            _w(w=300, l=75),
            _w(w=400, l=40),
            _w(w=2800, l=600),
            _w(w=8, l=32),
            _w(w=40, l=40)
        ]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'now_vgg16_512_window_set7': ModelInfo(
        factory=model_2d.classification_model_vgg,
        args=dict(use_gwap=True, nb_input_planes=7),
        dataset_args=dict(img_size=512, apply_windows=[
            _w(w=80, l=40),
            _w(w=130, l=75),
            _w(w=300, l=75),
            _w(w=400, l=40),
            _w(w=2800, l=600),
            _w(w=8, l=32),
            _w(w=40, l=40)
        ]),
        batch_size=32,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=2
    ),
    'dpn68_256_window_set7': ModelInfo(
        factory=model_2d.classification_model_dpn68b,
        args=dict(use_gwap=True, nb_input_planes=7, nb_windows_conv=16),
        dataset_args=dict(img_size=512, apply_windows=[
            _w(w=80, l=40),
            _w(w=130, l=75),
            _w(w=300, l=75),
            _w(w=400, l=40),
            _w(w=2800, l=600),
            _w(w=8, l=32),
            _w(w=40, l=40)
        ]),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=2
    ),
    'resnet34_512_crop_384_window_set7': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=False, nb_input_planes=7),
        dataset_args=dict(
            img_size=512,
            center_crop=384,
            apply_windows=[
                _w(w=80, l=40),
                _w(w=130, l=75),
                _w(w=300, l=75),
                _w(w=400, l=40),
                _w(w=2800, l=600),
                _w(w=8, l=32),
                _w(w=40, l=40)
            ]),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'resnet34_512_crop_384_cdf': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=False, nb_input_planes=1),
        dataset_args=dict(
            img_size=512,
            center_crop=384,
            convert_cdf=True),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'resnext50_512_crop_384_cdf_no_bn2': ModelInfo(
        factory=model_2d.classification_model_se_resnext50,
        args=dict(use_gwap=False, nb_input_planes=1, add_bn2=False),
        dataset_args=dict(
            img_size=512,
            center_crop=384,
            convert_cdf=True),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=4
    ),
    'resnext50_512_crop_384_cdf': ModelInfo(
        factory=model_2d.classification_model_se_resnext50,
        args=dict(use_gwap=False, nb_input_planes=1, add_bn2=True),
        dataset_args=dict(
            img_size=512,
            center_crop=384,
            convert_cdf=True),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=4
    ),
    'resnet34_256_cdf_5_planes_combine_last': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_last,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=256,
            num_slices=5,
            convert_cdf=True),
        batch_size=32,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=2
    ),
    'resnet34_256_cdf_5_planes_combine_first': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_first,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=256,
            num_slices=5,
            convert_cdf=True),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'resnet34_256_cdf_1_plane': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_first,
        args=dict(nb_input_slices=1),
        dataset_args=dict(
            img_size=256,
            num_slices=1,
            convert_cdf=True),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'resnet34_256_cdf_3_planes_combine_last': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_last,
        args=dict(nb_input_slices=3),
        dataset_args=dict(
            img_size=256,
            num_slices=3,
            convert_cdf=True),
        batch_size=64,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
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
    'resnet34_400_cdf': ModelInfo(
        factory=model_2d.classification_model_resnet34,
        args=dict(use_gwap=False, nb_input_planes=1),
        dataset_args=dict(
            img_size=400,
            num_slices=2,
            convert_cdf=True),
        batch_size=32,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1
    ),
    'resnet34_400_5_planes_combine_last_individual': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_last,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=4
    ),

    'resnet34_400_5_planes_combine_last_var': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_last_var,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'resnet34_400_5_planes_combine_last_var_no_acc': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_last_var,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True),
        batch_size=16,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=1,
        single_slice_steps=6
    ),

    'resnet34_400_5_planes_combine_last_var7': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_last_var,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=2,
        single_slice_steps=7
    ),

    'resnet34_400_5_planes_combine_last_var_freeze_bn5': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_last_var,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=2,
        single_slice_steps=6,
    ),

    'resnext50_384_5_planes_combine_last_var': ModelInfo(
        factory=model_2dc.classification_model_resnext50_combine_last_var,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=5,
            convert_cdf=True),
        batch_size=4,
        optimiser='radam',
        initial_lr=2e-5,
        accumulation_steps=1,
        single_slice_steps=6
    ),

    'resnet34_400_5_planes_combine_last_var_dr0.2': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_last_var,
        args=dict(nb_input_slices=5, dropout=0.2),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'resnet34_400_5_planes_combine_last_var_dr0': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_last_var,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'resnet34_384_5_planes_combine_last_var': ModelInfo(
        factory=model_2dc.classification_model_resnet34_combine_last_var,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=5,
            convert_cdf=True),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'dpn68_400_5_planes_combine_last_var': ModelInfo(
        factory=model_2dc.classification_model_dpn68_combine_last_var,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            # center_crop=384,
            num_slices=5,
            convert_cdf=True),
        batch_size=4,
        optimiser='radam',
        initial_lr=2e-5,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'resnet34_384_5_planes_segmentation': ModelInfo(
        factory=model_2dc_segmentation.segmentation_model_resnet34_combine_last_var,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=True,
            segmentation_oversample=25
        ),
        batch_size=8,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'resnet34_384_5_planes_segmentation2': ModelInfo(
        factory=model_2dc_segmentation.segmentation_model_resnet34_combine_last_var2,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=True,
            segmentation_oversample=16
        ),
        batch_size=8,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'resnet34_384_5_planes_seg2_dec2': ModelInfo(
        factory=model_2dc_segmentation.segmentation_model_resnet34_combine_last_var2_dec2,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=True,
            segmentation_oversample=16
        ),
        batch_size=8,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=2,
        single_slice_steps=6
    ),


    'xception_400': ModelInfo(
        factory=model_2dc.classification_model_xception,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=4,
        optimiser='radam',
        initial_lr=2e-5,
        accumulation_steps=4,
        single_slice_steps=6
    ),

    'nasnet_mobile_400': ModelInfo(
        factory=model_2dc.classification_model_nasnet_mobile,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=8,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'bninception_400': ModelInfo(
        factory=model_2dc.classification_model_bninception,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=8,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'resnet18_400': ModelInfo(
        factory=model_2dc.classification_model_resnet18_combine_last_var,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1,
        single_slice_steps=5
    ),

    'resnet18_400_no_vflip': ModelInfo(
        factory=model_2dc.classification_model_resnet18_combine_last_var,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=16,
        optimiser='radam',
        initial_lr=1e-4,
        accumulation_steps=1,
        single_slice_steps=6,
        use_vflip=False
    ),

    'resnet50_400': ModelInfo(
        factory=model_2dc.classification_model_resnet50_combine_last_var,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=8,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'inc_resnet_v2_384': ModelInfo(
        factory=model_2dc.classification_model_inception_resnet_v2,
        args=dict(nb_input_slices=3, dropout=0),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=3,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=4,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=2,
        single_slice_steps=6
    ),

    'airnet50_400_no_vflip': ModelInfo(
        factory=model_2dc.classification_model_airnet_50,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=8,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=2,
        single_slice_steps=6,
        use_vflip=False
    ),

    'airnet50_384': ModelInfo(
        factory=model_2dc.classification_model_airnet_50,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=8,
        optimiser='radam',
        initial_lr=2.5e-5,
        accumulation_steps=2,
        single_slice_steps=5,
        use_vflip=True
    ),

    'airnext50_384': ModelInfo(
        factory=model_2dc.classification_model_airnext_50,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=12,
        optimiser='radam',
        initial_lr=2.5e-5,
        accumulation_steps=1,
        single_slice_steps=5,
        use_vflip=True
    ),


    'se_preresnext26b_400': ModelInfo(
        factory=model_2dc.classification_model_se_preresnext26b,
        args=dict(nb_input_slices=5, dropout=0),
        dataset_args=dict(
            img_size=400,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False
        ),
        batch_size=12,
        optimiser='radam',
        initial_lr=2.5e-5,
        accumulation_steps=1,
        single_slice_steps=5,
        use_vflip=True
    ),

    'resnet18_384_5_planes_bn_f8': ModelInfo(
        factory=model_2dc_segmentation.segmentation_model_resnet18_bn_filters8,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=True,
            segmentation_oversample=16
        ),
        batch_size=16,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=1,
        single_slice_steps=5
    ),
    'resnet18_384_5_planes_bn_f8_masked': ModelInfo(
        factory=model_2dc_segmentation.segmentation_model_resnet18_bn_filters8_masked,
        args=dict(nb_input_slices=5),
        dataset_args=dict(
            img_size=400,
            center_crop=384,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=True,
            segmentation_oversample=16
        ),
        batch_size=16,
        optimiser='radam',
        initial_lr=5e-5,
        accumulation_steps=1,
        single_slice_steps=5
    ),

    'resnext50_400': ModelInfo(
        factory=model_2dc.classification_model_resnext50,
        args=dict(nb_input_slices=5, dropout=0.5),
        dataset_args=dict(
            img_size=400,
            # center_crop=384,
            num_slices=5,
            convert_cdf=True,
            add_segmentation_masks=False),
        batch_size=8,
        optimiser='radam',
        initial_lr=2.5e-5,
        accumulation_steps=2,
        single_slice_steps=5
    ),
}
