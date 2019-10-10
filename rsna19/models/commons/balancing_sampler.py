from copy import deepcopy
import os

import pandas as pd
import numpy as np
from torch.utils.data.sampler import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self, config, folds):
        self.config = config

        if config.csv_root_dir is None:
            csv_root_dir = os.path.normpath(__file__ + '/../../../data/csv')
        else:
            csv_root_dir = config.csv_root_dir

        data = pd.read_csv(os.path.join(csv_root_dir, config.dataset_file))
        data = data[data.fold.isin(folds)]
        data = data.reset_index(drop=True)
        self.data = data

        class_examples_dfs = {}
        for class_name in ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']:
            class_examples_dfs[class_name] = data[data[class_name] == 1.0]

        class_examples_dfs['negative'] = data[data['any'] == 0.0]
        self.class_examples_dfs = class_examples_dfs
        self.class_examples_dfs_used = deepcopy(class_examples_dfs)

    def __iter__(self):
        for _ in range(len(self)):
            out = np.random.choice(self.config.n_classes, int(self.config.batch_size/len(self.config.gpus)), p=self.config.probas)
            num_examples = np.zeros(self.config.n_classes, dtype=np.int32)
            for i in out:
                num_examples[i] += 1

            batch = []
            for class_name, n in zip(self.class_examples_dfs_used.keys(), num_examples):
                indices = self._sample_n(class_name, n)
                batch.extend(indices)

            yield batch

    def _sample_n(self, class_name, n):
        if len(self.class_examples_dfs_used[class_name]) <= n:
            self.class_examples_dfs_used[class_name] = deepcopy(self.class_examples_dfs[class_name])
        sampled_df = self.class_examples_dfs_used[class_name].sample(n)
        self.class_examples_dfs_used[class_name] = self.class_examples_dfs_used[class_name].drop(sampled_df.index)
        return sampled_df.index

    def __len__(self):
        return int(len(self.data)/int(self.config.batch_size/len(self.config.gpus)))
