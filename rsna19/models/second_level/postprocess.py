from pathlib import Path
from rsna19.configs.second_level import Config
from rsna19.models.smoothing.utils import compute_log_loss, compute_log_loss_per_class
from glob import glob

config = Config


def main(config):
    for path in glob(str(config.models_root / "*/*/predictions/*")):
        # TODO parallelize
        if 'test' in path or 'smoothed' in path:
            continue

        path = Path(path)
        path_smoothed = path.with_name(path.stem + "_smoothed.csv")

        # smooth_predictions(path, path_smoothed)

        # orig_loss = compute_log_loss(path)
        # smth_loss = compute_log_loss(path_smoothed)

        print(compute_log_loss_per_class(path), compute_log_loss(path), path)

        # print(f"diff:{(orig_loss - smth_loss):.04f}, orig: {orig_loss:.04f}, smth: {smth_loss:.04f}, {path}")


if __name__ == "__main__":
    main(Config())
