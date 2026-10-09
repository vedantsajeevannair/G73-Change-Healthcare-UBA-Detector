"""Steps 4 and 5 - the three models, the ensemble and the explanation.

    Isolation Forest  - unsupervised, trained on normal training events [33]
    Autoencoder       - unsupervised; reconstruction error of a bottleneck
                        MLP (scikit-learn) trained on normal events
    XGBoost           - supervised, class-weighted for the imbalance [35]

Each score is turned into a percentile against the validation week (normal
events only), so the three are on the same 0-1 scale. The ensemble is a
weighted mean of the percentiles; the weights are picked by grid search on
the training month using out-of-fold XGBoost scores (minimum 0.1 each). Explanations use
XGBoost's built-in TreeSHAP (pred_contribs), the same values the SHAP
library returns for tree models [34].
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["isolation_forest", "autoencoder", "xgboost"]


def _xgb(pos_weight: float, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.08,
                             subsample=0.9, colsample_bytree=0.9,
                             scale_pos_weight=pos_weight, eval_metric="aucpr",
                             random_state=seed, n_jobs=2)


def recall_at_fpr(scores: np.ndarray, y: np.ndarray, fpr: float) -> float:
    neg = np.sort(scores[y == 0])
    thr = neg[min(len(neg) - 1, int(np.ceil((1 - fpr) * len(neg))))]
    return float((scores[y == 1] > thr).mean()) if (y == 1).any() else 0.0


@dataclass
class Detector:
    seed: int = 7
    fpr_budget: float = 0.002
    scaler: StandardScaler = field(default_factory=StandardScaler)
    weights: dict = field(default_factory=dict)
    threshold: float = 1.0

    def fit(self, X_train, y_train, X_val):
        Xs = self.scaler.fit_transform(X_train)
        normal = Xs[y_train == 0]

        self.iforest = IsolationForest(n_estimators=300, contamination="auto",
                                       random_state=self.seed).fit(normal)
        self.ae = MLPRegressor(hidden_layer_sizes=(16, 4, 16), max_iter=300,
                               early_stopping=True, random_state=self.seed)
        self.ae.fit(normal, normal)

        pw = (y_train == 0).sum() / max(1, (y_train == 1).sum())
        self.xgb = _xgb(pw, self.seed).fit(X_train, y_train)

        # calibration reference: validation week, all normal
        self._ref = {k: np.sort(v) for k, v in self._raw(X_val).items()}

        # ensemble weights on the training month with out-of-fold XGBoost
        oof = np.zeros(len(y_train))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=self.seed).split(X_train, y_train):
            oof[te] = _xgb(pw, self.seed).fit(X_train[tr], y_train[tr]).predict_proba(X_train[te])[:, 1]
        raw = self._raw(X_train)
        raw["xgboost"] = oof
        pct = self._percentiles(raw)
        best = (-1.0, None)
        # every model keeps at least 0.1: the unsupervised models are the only
        # cover for attack behaviour that is missing from the labelled data
        grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for w_if, w_ae in itertools.product(grid, grid):
            w_x = 1 - w_if - w_ae
            if w_x < 0.1:
                continue
            s = w_if * pct["isolation_forest"] + w_ae * pct["autoencoder"] + w_x * pct["xgboost"]
            r = recall_at_fpr(s, y_train, self.fpr_budget)
            if r > best[0] + 1e-9:
                best = (r, dict(isolation_forest=w_if, autoencoder=w_ae, xgboost=round(w_x, 2)))
        self.weights = best[1]

        # alert threshold: the FPR budget on the validation week
        val = self.ensemble(self.scores(X_val))
        self.threshold = float(np.quantile(val, 1 - self.fpr_budget))
        self.model_thresholds = {k: float(np.quantile(v, 1 - self.fpr_budget))
                                 for k, v in self.scores(X_val).items()}
        return self

    def _raw(self, X):
        Xs = self.scaler.transform(X)
        return {
            "isolation_forest": -self.iforest.score_samples(Xs),
            "autoencoder": ((self.ae.predict(Xs) - Xs) ** 2).mean(axis=1),
            "xgboost": self.xgb.predict_proba(X)[:, 1],
        }

    def _percentiles(self, raw):
        return {k: np.searchsorted(self._ref[k], v, side="right") / len(self._ref[k])
                for k, v in raw.items()}

    def scores(self, X):
        return self._percentiles(self._raw(X))

    def ensemble(self, pct):
        return sum(self.weights[k] * pct[k] for k in MODEL_NAMES)

    def explain(self, X) -> np.ndarray:
        """TreeSHAP contributions of each feature to the XGBoost log-odds."""
        dm = xgb.DMatrix(X)
        return self.xgb.get_booster().predict(dm, pred_contribs=True)[:, :-1]
