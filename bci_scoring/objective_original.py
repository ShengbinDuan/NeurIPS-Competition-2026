"""Objective for the BCI-decoding track (track 2).

Cued mental-command classification from short EEG windows — *epoched*: one
label per window. A submission's model receives torch batches ``(B, C, T)``
and must return one predicted class per window (``predict(X) -> (B,)``).

Ranking metric: **balanced accuracy** (plain accuracy reported alongside).
Data flows as lazy dataloaders — see ``benchmark_utils/data.py``.
"""

import numpy as np
from benchopt import BaseObjective
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from benchmark_utils.data import to_numpy


class Objective(BaseObjective):

    name = "BCI-decoding"
    url = "https://github.com/neural-interfaces26/2026-competition"

    # CPU/GPU variants resolved by ``benchopt install [--gpu]``; the conda
    # metapackages pin the matching torch build (CI installs the cpu one).
    requirements = {
        "cpu": ["scikit-learn", "pytorch-cpu"],
        "gpu": ["scikit-learn", "pytorch-gpu"],
    }

    min_benchopt_version = "1.10.0"

    # Each solver runs once to completion (no convergence curve).
    sampling_strategy = "run_once"

    # training=True lets the solvers' optional ``fit`` run before evaluation
    # (how baselines and participants train); default runs are
    # inference-only, like the platform. Select it with
    #     benchopt run ... -o "<objective>[training=True]"
    parameters = {"training": [False]}

    # ``benchopt test`` exercises the full contract, training included.
    test_config = {"training": True}

    def set_data(self, train_loader, test_loader, n_classes, **meta):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_classes = n_classes
        self.meta = meta  # sfreq, ch_names, chs_info, n_chans, n_times, ...

    def skip(self, train_loader=None, **data):
        if self.training and train_loader is None:
            return True, "training=True needs a dataset with a train split"
        return False, None

    def get_objective(self):
        # Train loader goes to the solver; the test loader stays here so the
        # objective owns evaluation.
        return dict(
            # Inference-only unless training is selected: no loader, no fit.
            train_loader=self.train_loader if self.training else None,
            n_classes=self.n_classes,
            **self.meta,
        )

    def evaluate_result(self, model):
        y_true, y_pred = [], []
        for X, y, _info in self.test_loader:
            y_pred.append(to_numpy(model.predict(X)))
            y_true.append(to_numpy(y))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        return dict(
            balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
            accuracy=accuracy_score(y_true, y_pred),
            n_classes=self.n_classes,
        )

    def get_one_result(self):
        # A trivial constant model, used by ``benchopt test`` to validate the
        # metric computation.
        from benchmark_utils.baselines import ConstantClassifier
        return dict(model=ConstantClassifier())
