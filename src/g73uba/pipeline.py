"""The six-step workflow of Section 4.10, end to end.

1 data collection   -> simulate.generate (or your own logs in the same shape)
2 session building  -> features.build_features (per-user session state)
3 feature eng.      -> features.build_features (Table 4)
4 modelling         -> models.Detector (Isolation Forest, autoencoder, XGBoost)
5 ensemble + SHAP   -> models.Detector.ensemble / explain, plus rules.run_rules
6 alert + response  -> alert queue with reasons and a suggested action
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURE_LABELS, FEATURES, build_features
from .models import MODEL_NAMES, Detector
from .rules import run_rules
from .simulate import ENCRYPTION, INITIAL_ACCESS, TEST_START, TRAIN_START, SimConfig, generate

VAL_DAYS = 7
WARMUP_DAYS = 7   # history features need a week of baseline before training
PHASES = ["initial_access", "tooling", "credential_abuse", "lateral_movement", "impact"]


def _metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp / max(1, tp + fn),
                fpr=fp / max(1, fp + tn),
                precision=tp / max(1, tp + fp))


def _top_reasons(contrib: np.ndarray, row: pd.Series, k: int = 3) -> str:
    idx = np.argsort(-contrib)[:k]
    parts = []
    for j in idx:
        if contrib[j] <= 0:
            continue
        f = FEATURES[j]
        parts.append(f"{FEATURE_LABELS[f]} = {row[f]:.2f} (+{contrib[j]:.2f})")
    return "; ".join(parts)


def run(seed: int = 7, fpr_budget: float = 0.002, out_dir: str | Path = "output",
        log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log("[1/6] Generating synthetic enterprise logs ...")
    logs = generate(SimConfig(seed=seed))
    log(f"      {len(logs.auth):,} auth events, {len(logs.process):,} process events, "
        f"{len(logs.users)} users")

    log("[2/6] Building per-user sessions and [3/6] Table 4 features ...")
    val_start = pd.Timestamp(TEST_START - timedelta(days=VAL_DAYS))
    ev = build_features(logs, train_end=val_start)
    warm = pd.Timestamp(TRAIN_START + timedelta(days=WARMUP_DAYS))
    train = ev[(ev.time >= warm) & (ev.time < val_start)]
    val = ev[(ev.time >= val_start) & (ev.time < pd.Timestamp(TEST_START))]
    test = ev[ev.time >= pd.Timestamp(TEST_START)].copy()
    log(f"      train {len(train):,} (attack {int(train.label.sum())}) | "
        f"validation {len(val):,} | test {len(test):,} (attack {int(test.label.sum())})")

    log("[4/6] Training Isolation Forest, autoencoder and XGBoost ...")
    det = Detector(seed=seed, fpr_budget=fpr_budget).fit(
        train[FEATURES].to_numpy(), train.label.to_numpy(), val[FEATURES].to_numpy())
    log(f"      ensemble weights {det.weights}, threshold {det.threshold:.4f}")

    log("[5/6] Scoring the test window, rule layer and SHAP explanations ...")
    X = test[FEATURES].to_numpy()
    pct = det.scores(X)
    for k in MODEL_NAMES:
        test[k] = pct[k]
    test["risk"] = det.ensemble(pct)
    test["ml_alert"] = (test.risk > det.threshold).astype(int)

    admins = set(logs.users[logs.users.admin].user)
    rules = run_rules(logs.process[logs.process.time >= pd.Timestamp(TEST_START)], admins)

    y = test.label.to_numpy()
    metrics = {k: _metrics((test[k] > det.model_thresholds[k]).to_numpy().astype(int), y)
               for k in MODEL_NAMES}
    metrics["ensemble"] = _metrics(test.ml_alert.to_numpy(), y)
    rp = rules.label.to_numpy() if len(rules) else np.array([])
    metrics["rule_layer"] = dict(alerts=int(len(rules)), true_alerts=int(rp.sum()),
                                 false_alerts=int(len(rp) - rp.sum()))

    contrib = det.explain(X)
    alerts = test[test.ml_alert == 1].copy()
    alerts["reasons"] = [_top_reasons(contrib[test.index.get_loc(i)], test.loc[i])
                         for i in alerts.index]
    alerts["source"] = "ML ensemble"
    alerts["action"] = np.where(alerts.risk > 0.995, "Disable account and isolate host",
                                "Analyst review")

    rule_alerts = rules.rename(columns={"host": "dst"}).assign(
        risk=1.0, source="Rule " + rules.severity, src="",
        reasons=rules.reason + " (" + rules.technique + ")",
        action=np.where(rules.severity == "high", "Isolate host now", "Analyst review"))
    cols = ["time", "source", "user", "src", "dst", "process", "risk", "reasons",
            "action", "label", "phase"]
    queue = pd.concat([alerts[cols], rule_alerts[cols]]).sort_values("time")

    log("[6/6] Building the alert queue and detection timeline ...")
    enc = pd.Timestamp(ENCRYPTION)
    timeline = []
    for ph in PHASES:
        hit = queue[(queue.label == 1) & (queue.phase == ph)]
        first = hit.time.min() if len(hit) else None
        timeline.append(dict(
            phase=ph,
            events=int(((test.label == 1) & (test.phase == ph)).sum()),
            first_alert=None if first is None else first.strftime("%Y-%m-%d %H:%M"),
            first_alert_by=None if first is None else hit.sort_values("time").source.iloc[0],
            hours_before_encryption=None if first is None else round((enc - first).total_seconds() / 3600, 1),
        ))
    first_true = queue[queue.label == 1].time.min()

    daily = (test.assign(day=test.time.dt.date)
             .groupby("day").agg(max_risk=("risk", "max"), events=("risk", "size"),
                                 alerts=("ml_alert", "sum"), attack=("label", "sum"))
             .reset_index())
    daily["day"] = daily.day.astype(str)

    # global feature importance (mean |SHAP|) on the test window
    imp = np.abs(contrib).mean(axis=0)
    importance = sorted(({"feature": f, "label": FEATURE_LABELS[f], "mean_abs_shap": float(v)}
                         for f, v in zip(FEATURES, imp)), key=lambda d: -d["mean_abs_shap"])

    summary = dict(
        seed=seed, fpr_budget=fpr_budget,
        period=dict(train=f"{warm.date()} to {(val_start - pd.Timedelta(days=1)).date()}",
                    validation=f"{val_start.date()} to {(TEST_START - timedelta(days=1)).date()}",
                    test=f"{TEST_START.date()} to {test.time.max().date()}"),
        counts=dict(train=len(train), validation=len(val), test=len(test),
                    test_attack=int(test.label.sum()), process_events=len(logs.process)),
        initial_access=str(INITIAL_ACCESS), encryption=str(ENCRYPTION),
        weights=det.weights, threshold=det.threshold,
        metrics=metrics, timeline=timeline,
        first_true_alert=None if pd.isna(first_true) else str(first_true),
        lead_time_hours=None if pd.isna(first_true) else round((enc - first_true).total_seconds() / 3600, 1),
        false_alerts_per_day=round(int(((queue.label == 0)).sum()) / max(1, test.time.dt.date.nunique()), 2),
        importance=importance, daily=daily.to_dict(orient="records"),
    )

    queue.to_csv(out / "alert_queue.csv", index=False)
    test.drop(columns=["success"]).to_csv(out / "scored_events.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    return dict(summary=summary, queue=queue, test=test, contrib=contrib, rules=rules)
