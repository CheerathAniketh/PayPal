"""The recovery-propensity model: LightGBM + cross-fitted calibration.

``PropensityModel`` is the object that replaces the rules ``_p_recover``.  Its
public surface is deliberately the same shape as the hook it fills:

    model.predict_proba(record_dict, intervention) -> float in [0, 1]

Calibration
-----------
The stop rule thresholds on the probability, so the number has to *mean* what it
says: among records the model scores 0.30, about 30% should recover.  A raw
gradient-boosting score does not have that property.  We fit a calibrator on
**cross-fitted out-of-fold scores** (grouped by customer), never on the same
rows the booster trained on, so the calibration estimate is honest rather than
optimistic.

Grouped everything
------------------
Every split -- CV folds for calibration, and the caller's train/holdout split --
is by ``customer_id``.  A customer's latent traits persist across their records;
letting the same customer sit in both sides of any split would let the model
memorise the person and report an AUC that would not survive contact with a new
customer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.taxonomy import Intervention
from recoup.ml.dataset import Dataset
from recoup.ml.features import feature_names, feature_vector


# LightGBM params tuned for a small, shallow problem: a few hundred rows means
# depth and leaf count must stay low or the trees memorise noise.  These are
# conservative on purpose; the story is "a fair model", not "a squeezed score".
DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_split_gain": 0.0,
    "verbosity": -1,
}


@dataclass
class PropensityModel:
    """A trained, calibrated scorer for P(recover | record, intervention)."""

    booster: Any                                  # lightgbm.LGBMClassifier
    calibrator: Any                               # sklearn IsotonicRegression | _Platt
    names: List[str] = field(default_factory=feature_names)

    # ------------------------------------------------------------------
    # Inference -- the Phase-4 hook shape
    # ------------------------------------------------------------------
    def _raw(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict_proba(X)[:, 1]

    def predict_proba(self, record: Dict[str, Any], intervention: Intervention) -> float:
        """Calibrated P(recover) for one (record, intervention). In [0, 1]."""
        x = np.asarray([feature_vector(record, intervention)], dtype=float)
        raw = float(self._raw(x)[0])
        cal = float(self.calibrator.predict([raw])[0])
        return max(0.0, min(1.0, cal))

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw(X)
        cal = self.calibrator.predict(raw)
        return np.clip(cal, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "PropensityModel":
        import joblib

        return joblib.load(Path(path))


class _Platt:
    """A tiny logistic (Platt) calibrator: p = sigmoid(a*raw + b).

    A stable fallback to isotonic when the calibration set is small; isotonic can
    fit a jagged step function on a few hundred points, Platt cannot.
    """

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_Platt":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(raw.reshape(-1, 1), y)
        self.a = float(lr.coef_[0, 0])
        self.b = float(lr.intercept_[0])
        return self

    def predict(self, raw) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        z = self.a * raw + self.b
        return 1.0 / (1.0 + np.exp(-z))


def _fit_booster(X: np.ndarray, y: np.ndarray, params: Dict[str, Any]):
    from lightgbm import LGBMClassifier

    clf = LGBMClassifier(**params)
    clf.fit(X, y)
    return clf


def _oof_scores(
    dataset: Dataset, params: Dict[str, Any], n_splits: int, seed: int
) -> np.ndarray:
    """Cross-fitted out-of-fold raw scores, grouped by customer.

    Every row is scored by a booster that never saw that row's customer -- the
    honest input for calibration.
    """
    from sklearn.model_selection import GroupKFold

    X, y, groups = dataset.X, dataset.y, dataset.groups
    oof = np.zeros(len(y), dtype=float)
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=splits)
    for train_idx, test_idx in gkf.split(X, y, groups):
        booster = _fit_booster(X[train_idx], y[train_idx], params)
        oof[test_idx] = booster.predict_proba(X[test_idx])[:, 1]
    return oof


def train_model(
    dataset: Dataset,
    *,
    params: Optional[Dict[str, Any]] = None,
    calibration: str = "isotonic",
    n_calibration_splits: int = 5,
    seed: int = 20260251,
) -> PropensityModel:
    """Fit the booster on all rows, then a calibrator on cross-fitted OOF scores.

    ``calibration`` is ``"isotonic"`` or ``"platt"``.  The booster is refit on the
    full training set (calibration does not waste data); only the *calibration
    map* is learned out-of-fold.
    """
    params = dict(params or DEFAULT_LGBM_PARAMS)

    # 1. calibrator, from grouped OOF scores (honest, never in-fold).
    oof = _oof_scores(dataset, params, n_calibration_splits, seed)
    if calibration == "platt":
        calibrator: Any = _Platt().fit(oof, dataset.y)
    else:
        from sklearn.isotonic import IsotonicRegression

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(oof, dataset.y)

    # 2. booster, refit on everything.
    booster = _fit_booster(dataset.X, dataset.y, params)

    return PropensityModel(booster=booster, calibrator=calibrator, names=list(dataset.feature_names))
