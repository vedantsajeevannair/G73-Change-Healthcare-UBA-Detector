import numpy as np
import pandas as pd

from g73uba.features import FEATURES, build_features
from g73uba.models import recall_at_fpr
from g73uba.rules import match
from g73uba.simulate import INITIAL_ACCESS, TEST_START, SimConfig, generate


def small_logs():
    return generate(SimConfig(seed=1, n_users=40, n_servers=30, n_admins=4, no_mfa_fraction=0.3))


def test_simulation_has_both_campaigns():
    logs = small_logs()
    attack = logs.auth[logs.auth.label == 1]
    assert (attack.time < pd.Timestamp(TEST_START)).any(), "red-team campaign missing"
    first_chc = attack[attack.time >= pd.Timestamp(TEST_START)].time.min()
    assert first_chc == pd.Timestamp(INITIAL_ACCESS)


def test_features_have_no_lookahead():
    logs = small_logs()
    full = build_features(logs, train_end=pd.Timestamp(TEST_START))
    cut = pd.Timestamp("2024-01-20")
    logs.auth = logs.auth[logs.auth.time < cut].reset_index(drop=True)
    part = build_features(logs, train_end=pd.Timestamp(TEST_START))
    hist = ["src_logged_on", "other_interactive_logins", "prev_login_hour_fraction",
            "failed_logins_last_hour", "fraction_from_same_src", "gateway_new_device"]
    np.testing.assert_allclose(full.loc[:len(part) - 1, hist].to_numpy(), part[hist].to_numpy())


def test_initial_access_looks_unusual():
    logs = small_logs()
    ev = build_features(logs, train_end=pd.Timestamp(TEST_START))
    row = ev[(ev.time == pd.Timestamp(INITIAL_ACCESS)) & (ev.label == 1)].iloc[0]
    assert row.gateway_no_mfa == 1 and row.gateway_new_device == 1
    assert set(FEATURES) <= set(ev.columns)


def test_rules():
    assert match("vssadmin.exe delete shadows /all", True)[0] == "high"
    assert match("anydesk.exe", True) is None          # admins use it legitimately
    assert match("anydesk.exe", False)[0] == "medium"
    assert match(float("nan"), False) is None


def test_recall_at_fpr():
    s = np.r_[np.linspace(0, 1, 1000), [2.0, 2.0, 0.1]]
    y = np.r_[np.zeros(1000), [1, 1, 1]]
    assert abs(recall_at_fpr(s, y, 0.01) - 2 / 3) < 1e-9
