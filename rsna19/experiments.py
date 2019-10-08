from models import model_2d


class ModelInfo:
    def __init__(self,
                 factory,
                 args,
                 dataset_args,
                 batch_size,
                 optimiser='sgd',
                 scheduler='steps',
                 initial_lr=1e-3,
                 optimiser_milestones=None,
                 accumulation_steps=1,
                 weight_decay=0,
                 is_pretrained=True,
                 ):
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
}
