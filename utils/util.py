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
    files = sorted(glob(os.path.join(np_data_path, "*.npz")))
    print("Total files found:", len(files))
    if "78" in np_data_path:
        r_p_path = r"utils/r_permute_78.npy"
    else:
        r_p_path = r"utils/r_permute_20.npy"
    if not os.path.exists(r_p_path):
        raise FileNotFoundError(f"Permutation file not found: {r_p_path}")
    r_permute = np.load(r_p_path)
    print("Permutation indices:", r_permute)
    if len(r_permute) > len(files):
        raise ValueError("Permutation index exceeds the number of available files.")
    files_dict = {os.path.split(i)[-1][3:5]: files_dict.get(os.path.split(i)[-1][3:5], []) + [i] for i in files}
    files_pairs = np.array([files_dict[key] for key in sorted(files_dict)], dtype=object)
    if len(r_permute) > len(files_pairs):
        raise ValueError("Permutation index exceeds the number of file groups.")
    files_pairs = files_pairs[r_permute]
    train_files = np.array_split(files_pairs, n_folds)
    folds_data = {}
    for fold_id in range(n_folds):
        subject_files = [item for sublist in train_files[fold_id] for item in sublist]
        files_pairs2 = [item for sublist in files_pairs for item in sublist]
        training_files = list(set(files_pairs2) - set(subject_files))
        folds_data[fold_id] = [training_files, subject_files]
    return folds_data


def calc_class_weight(labels_count):
    total = np.sum(labels_count)
    class_weight = {key: round(math.log((1 / len(labels_count) * 1.5) * total / float(labels_count[key])) * (1 / len(labels_count) * 1.5), 2) if math.log((1 / len(labels_count) * 1.5) * total / float(labels_count[key])) > 1.0 else 1.0 for key in range(len(labels_count))}
    return [class_weight[i] for i in range(len(labels_count))]


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
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