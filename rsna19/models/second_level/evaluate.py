from multiprocessing import Pool

from pathlib import Path
from rsna19.configs.second_level import Config
from rsna19.models.smoothing.utils import compute_log_loss, smooth_predictions
from glob import glob
from tqdm import tqdm


def path_generator(config):
    for path in glob(str(config.models_root / "*/*/predictions/*")):
        if 'test' in path or 'smoothed' in path:
            continue

        yield path


def worker(path, smooth_predictions_func=smooth_predictions):
    path = Path(path)
    path_smoothed = path.with_name(path.stem + "_smoothed.csv")

    smooth_predictions_func(path, path_smoothed)

    orig_loss = compute_log_loss(path)
    smth_loss = compute_log_loss(path_smoothed)
    diff = orig_loss - smth_loss

    print(f"diff:{diff:.05f}, orig: {orig_loss:.05f}, smth: {smth_loss:.05f}, {path}")
    return diff


def main(config):
    with Pool(8) as p:
        paths = list(path_generator(config))
        diffs = list(tqdm(p.imap(worker, paths), total=len(paths)))
        print(sum(diffs)/len(diffs))


if __name__ == "__main__":
    main(Config())
