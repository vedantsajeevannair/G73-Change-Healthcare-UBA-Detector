"""Steps 2 and 3 - session building and feature engineering.

One row per authentication event. Features follow Table 4 of the report:
the first seven come from Kushwaha et al. [31]; the last three are the
group's proposed extensions (gateway novelty, outbound volume, and - in
rules.py - security-tool tampering).

History features are computed online, in time order, so an event only ever
sees the events that happened before it (no look-ahead).
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta

import numpy as np
import pandas as pd

from .simulate import LogSet

FEATURES = [
    # [31]
    "is_remote",
    "src_logged_on",
    "other_interactive_logins",
    "process_run",
    "process_rarity",
    "prev_login_hour_fraction",
    "failed_logins_last_hour",
    "fraction_from_same_src",
    # proposed by the group
    "gateway_new_device",
    "gateway_no_mfa",
    "outbound_volume_ratio",
]

FEATURE_LABELS = {
    "is_remote": "Remote login (source != destination)",
    "src_logged_on": "Account had a real session on the source host",
    "other_interactive_logins": "Hosts the account is logged in to at once",
    "process_run": "New process started right after the login",
    "process_rarity": "How rare that process is in the baseline",
    "prev_login_hour_fraction": "How usual this hour is for the account",
    "failed_logins_last_hour": "Failed logins from the same source (1 h)",
    "fraction_from_same_src": "How often the account logs in from this source",
    "gateway_new_device": "First time this device / location at the gateway",
    "gateway_no_mfa": "Gateway login without a second factor",
    "outbound_volume_ratio": "Destination outbound bytes vs its baseline",
}

INTERACTIVE = {"Interactive", "RemoteInteractive"}
SESSION_TTL = timedelta(hours=12)


def _process_join(auth: pd.DataFrame, proc: pd.DataFrame, train_end) -> pd.DataFrame:
    """For each login, the first process the same account starts on the
    destination host within five minutes (the 'was process run' feature)."""
    base = proc[proc.time < train_end]
    freq = base.process.value_counts(normalize=True)
    p = proc.rename(columns={"host": "dst", "time": "ptime"})[["ptime", "dst", "user", "process"]]
    left = auth[["time", "dst", "user"]].reset_index()
    j = pd.merge_asof(left.sort_values("time"), p.sort_values("ptime"),
                      left_on="time", right_on="ptime", by=["user", "dst"],
                      direction="forward", tolerance=pd.Timedelta(minutes=5))
    j = j.set_index("index").sort_index()
    run = j.process.notna().astype(int)
    rarity = j.process.map(lambda n: -np.log10(freq.get(n, 1e-6)) if isinstance(n, str) else 0.0)
    return pd.DataFrame({"process_run": run, "process_rarity": rarity,
                         "process": j.process})


def build_features(logs: LogSet, train_end) -> pd.DataFrame:
    auth = logs.auth.copy()
    n = len(auth)
    out = {k: np.zeros(n) for k in FEATURES}

    sessions: dict[str, dict[str, object]] = defaultdict(dict)   # user -> host -> last time
    src_count: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hour_count: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(24))
    total: dict[str, int] = defaultdict(int)
    failed: dict[tuple, deque] = defaultdict(deque)
    devices: dict[str, set] = defaultdict(set)

    for i, r in enumerate(auth.itertuples(index=False)):
        t, u = r.time, r.user
        sess = sessions[u]
        for h in [h for h, ts in sess.items() if t - ts > SESSION_TTL]:
            del sess[h]

        gateway = r.logon_type == "Gateway"
        out["is_remote"][i] = float(r.src != r.dst)
        out["src_logged_on"][i] = 1.0 if (gateway or r.src in sess or r.src == r.dst) else 0.0
        out["other_interactive_logins"][i] = len(sess)

        hc = hour_count[u]
        tot = total[u]
        if tot:
            hrs = [(t.hour + d) % 24 for d in (-1, 0, 1)]
            out["prev_login_hour_fraction"][i] = hc[hrs].sum() / tot
            out["fraction_from_same_src"][i] = src_count[u][r.src] / tot
        else:
            out["prev_login_hour_fraction"][i] = 1.0
            out["fraction_from_same_src"][i] = 1.0

        q = failed[(u, r.src)]
        while q and t - q[0] > timedelta(hours=1):
            q.popleft()
        out["failed_logins_last_hour"][i] = len(q)

        if gateway:
            key = f"{r.device}|{r.geo}"
            out["gateway_new_device"][i] = float(key not in devices[u])  # includes first-ever portal use
            out["gateway_no_mfa"][i] = float(r.mfa is False)

        # update history with this event
        if not r.success:
            q.append(t)
            continue
        if gateway:
            devices[u].add(f"{r.device}|{r.geo}")
        if r.logon_type in INTERACTIVE:
            sess[r.dst] = t
        hc[t.hour] += 1
        src_count[u][r.src] += 1
        total[u] += 1

    feats = pd.DataFrame(out)
    pj = _process_join(auth, logs.process, train_end)
    feats["process_run"] = pj.process_run.values
    feats["process_rarity"] = pj.process_rarity.values

    # outbound volume of the destination host that day vs its training median
    flow = logs.netflow
    med = flow[pd.to_datetime(flow.day) < train_end].groupby("host").bytes_out.median()
    vol = flow.set_index(["day", "host"]).bytes_out
    keys = list(zip(auth.time.dt.date, auth.dst))
    v = vol.reindex(keys).to_numpy()
    m = med.reindex(auth.dst).to_numpy()
    feats["outbound_volume_ratio"] = np.nan_to_num(v / m, nan=1.0)

    meta = auth[["time", "user", "src", "dst", "auth_type", "logon_type",
                 "success", "label", "phase"]].reset_index(drop=True)
    meta["process"] = pj.process.values
    return pd.concat([meta, feats[FEATURES]], axis=1)
