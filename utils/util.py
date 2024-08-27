import json
from pathlib import Path
from collections import OrderedDict
from itertools import repeat
import pandas as pd
import os
import numpy as np
from glob import glob
import math

def load_folds_data_shhs(np_data_path, n_folds):
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    r_p_path = r"utils/r_permute_shhs.npy"
    r_permute = np.load(r_p_path)
    npzfiles = np.asarray(files , dtype='<U200')[r_permute]
    train_files = np.array_split(npzfiles, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = train_files[fold_id]
        training_files = list(set(npzfiles) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data

def load_folds_data(np_data_path, n_folds):
    # Gather all .npz files in the specified directory
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    print(f"Total files found in {np_data_path}: {len(files)}")  # Debug statement

    # Select the appropriate permutation file based on the directory name
    if "78" in np_data_path:
        r_p_path = "utils/r_permute_78.npy"
    else:
        r_p_path = "utils/r_permute_20.npy"

    # Load or regenerate the permutation indices
    if os.path.exists(r_p_path):
        r_permute = np.load(r_p_path)
        print(f"Loaded permutation indices from {r_p_path}: {len(r_permute)}")
    else:
        print("Permutation file not found, regenerating...")
        r_permute = np.random.permutation(len(files))
        np.save(r_p_path, r_permute)

    # Check if the permutation array is larger than the number of files
    if len(r_permute) > len(files):
        print("Permutation index exceeds number of files, resizing...")
        r_permute = np.random.permutation(len(files))  # Regenerate permutation indices
        np.save(r_p_path, r_permute)  # Save the new permutation indices

    # Organize files into pairs according to permutation
    files_pairs = np.array(files)[r_permute]  # Apply permutation
    train_files = np.array_split(files_pairs, n_folds)  # Split files into folds

    # Prepare the fold data structure
    folds_data = {}
    for fold_id, fold_files in enumerate(train_files):
        subject_files = fold_files
        training_files = list(set(files) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
        print(f"Fold {fold_id}: {len(subject_files)} subjects, {len(training_files)} training files")

    return folds_data


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = dict()
    num_classes = len(labels_count)

    factor = 1 / num_classes
    mu = [factor * 1.5, factor * 2, factor * 1.5, factor, factor * 1.5] # THESE CONFIGS ARE FOR SLEEP-EDF-20 ONLY

    for key in range(num_classes):
        score = math.log(mu[key] * total / float(labels_count[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        class_weight[key] = round(class_weight[key] * mu[key], 2)

    class_weight = [class_weight[i] for i in range(num_classes)]

    return class_weight


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)