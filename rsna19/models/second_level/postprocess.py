from pathlib import Path
from rsna19.configs.second_level import Config
from rsna19.models.smoothing.utils import compute_log_loss, compute_log_loss_per_class
from glob import glob

config = Config


def main(config):
    for path in glob(str(config.models_root / "*/*/predictions/*")):
        if 'test' in path or 'smoothed' in path:
            continue

        path = Path(path)
        path_smoothed = path.with_name(path.stem + "_smoothed.csv")

        # TODO weighted sum of smoothed predictions
        # print(compute_log_loss_per_class(path), compute_log_loss(path), path)


if __name__ == "__main__":
    main(Config())
