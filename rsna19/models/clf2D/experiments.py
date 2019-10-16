from rsna19.models.clf2D import model_2d, model_2dc


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
                 ):
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
}
