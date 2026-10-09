"""Step 1 - data collection (simulated).

The Change Healthcare logs are not public, so this module generates a
synthetic enterprise log set with the same shape as the five log sources in
Figure 3 of the report:

    auth      - remote-access gateway (Citrix) logins + Windows logons
    process   - process / EDR events
    netflow   - daily outbound bytes per host

Two labelled intrusions are injected:

    * a red-team exercise in the training month (gives XGBoost its labels,
      the same role the LANL red-team file plays in [31]);
    * a replay of the Change Healthcare timeline in the test window:
      stolen-credential login on 12 Feb 2024 through a portal without MFA,
      remote tools, Kerberos abuse, nine days of lateral movement and data
      theft, security-tool tampering and encryption on 21 Feb 2024.

Everything here is synthetic. Host and user names are invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

TRAIN_START = datetime(2024, 1, 8)
TEST_START = datetime(2024, 2, 5)      # first day of the evaluation window
END = datetime(2024, 2, 23)            # exclusive
INITIAL_ACCESS = datetime(2024, 2, 12, 2, 13)
ENCRYPTION = datetime(2024, 2, 21, 4, 40)

GATEWAY = "CTX-GW-01"
VDI_HOSTS = [f"CTX-VDI-{i:02d}" for i in range(1, 5)]

NORMAL_PROCS = ["outlook.exe", "chrome.exe", "excel.exe", "winword.exe",
                "teams.exe", "explorer.exe", "acrord32.exe", "claimsclient.exe"]
ADMIN_PROCS = ["powershell.exe", "mmc.exe", "psexec.exe", "cmd.exe",
               "anydesk.exe", "sqlcmd.exe", "robocopy.exe"]


@dataclass
class SimConfig:
    seed: int = 7
    n_users: int = 220
    n_servers: int = 60
    n_admins: int = 12
    no_mfa_fraction: float = 0.12   # legacy portal users without MFA


@dataclass
class LogSet:
    auth: pd.DataFrame
    process: pd.DataFrame
    netflow: pd.DataFrame
    users: pd.DataFrame
    meta: dict


class _Builder:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.auth: list[dict] = []
        self.proc: list[dict] = []
        self.flow_extra: list[dict] = []

    # -- helpers -----------------------------------------------------------
    def a(self, t, user, src, dst, kind, logon, ok=True, mfa=None,
          device=None, geo=None, label=0, phase=""):
        self.auth.append(dict(time=t, user=user, src=src, dst=dst,
                              auth_type=kind, logon_type=logon, success=ok,
                              mfa=mfa, device=device, geo=geo,
                              label=label, phase=phase))

    def p(self, t, host, user, name, label=0, phase=""):
        self.proc.append(dict(time=t, host=host, user=user, process=name,
                              label=label, phase=phase))


def _population(cfg: SimConfig, rng) -> pd.DataFrame:
    servers = [f"SRV-{i:03d}" for i in range(cfg.n_servers)]
    rows = []
    for i in range(cfg.n_users):
        admin = i < cfg.n_admins
        k = rng.integers(8, 20) if admin else rng.integers(2, 6)
        rows.append(dict(
            user=f"U{i:04d}",
            workstation=f"WS-{i:04d}",
            admin=admin,
            servers=list(rng.choice(servers, size=k, replace=False)),
            start_hour=int(rng.choice([7, 8, 9, 10, 20], p=[.2, .35, .3, .1, .05])),
            remote_worker=bool(rng.random() < 0.35),
            mfa=bool(rng.random() > cfg.no_mfa_fraction),
            devices=[f"DEV-{i:04d}-{j}" for j in range(rng.integers(1, 3))],
            geo=str(rng.choice(["Minneapolis", "Nashville", "Eden Prairie",
                                "Dallas", "Boston"])),
        ))
    return pd.DataFrame(rows)


def _normal_day(b: _Builder, u, day: datetime):
    rng = b.rng
    if day.weekday() >= 5 and rng.random() > 0.15:
        return
    if rng.random() < 0.06:  # leave / sick day
        return
    start = day + timedelta(hours=u.start_hour, minutes=int(rng.integers(0, 50)))
    hours = rng.uniform(7, 9.5)

    if u.remote_worker and rng.random() < 0.7:
        # Citrix gateway login from a known home device, then a VDI desktop
        dev = rng.choice(u.devices)
        mfa = bool(u.mfa)
        if rng.random() < 0.02:     # a genuine new laptop now and then
            dev = f"DEV-NEW-{rng.integers(1e6)}"
        for _ in range(int(rng.poisson(0.15))):  # typo'd passwords
            b.a(start - timedelta(seconds=40), u.user, "EXTERNAL", GATEWAY,
                "Citrix", "Gateway", ok=False, mfa=mfa, device=dev, geo=u.geo)
        b.a(start, u.user, "EXTERNAL", GATEWAY, "Citrix", "Gateway",
            mfa=mfa, device=dev, geo=u.geo)
        desk = VDI_HOSTS[int(u.user[1:]) % len(VDI_HOSTS)]
        b.a(start + timedelta(seconds=20), u.user, GATEWAY, desk,
            "Kerberos", "RemoteInteractive")
    else:
        desk = u.workstation
        b.a(start, u.user, desk, desk, "Kerberos", "Interactive")

    # working-day activity: network logons from the desktop to usual servers
    n = int(rng.poisson(9 if u.admin else 5))
    for _ in range(n):
        t = start + timedelta(minutes=float(rng.uniform(5, hours * 60)))
        srv = rng.choice(u.servers)
        if rng.random() < 0.03:
            srv = f"SRV-{rng.integers(b.cfg.n_servers):03d}"  # occasional new server
        b.a(t, u.user, desk, srv, "Kerberos" if rng.random() < .9 else "NTLM",
            "Network")
        if u.admin and rng.random() < 0.35:
            b.p(t + timedelta(seconds=int(rng.integers(5, 120))), srv, u.user,
                str(rng.choice(ADMIN_PROCS)))
        elif rng.random() < 0.05:
            b.p(t + timedelta(seconds=int(rng.integers(5, 200))), srv, u.user,
                "claimsclient.exe")

    for _ in range(int(rng.poisson(4))):
        t = start + timedelta(minutes=float(rng.uniform(0, hours * 60)))
        b.p(t, desk, u.user, str(rng.choice(NORMAL_PROCS)))


def _campaign(b: _Builder, users: pd.DataFrame, victim: str, admin: str,
              t0: datetime, days: int, impact: datetime | None, tag: str):
    """Inject one intrusion. `tag` is 'redteam' or 'chc'."""
    rng = b.rng
    v = users.set_index("user").loc[victim]
    attacker_dev, attacker_geo = f"DEV-ATK-{tag}", "Unknown-VPS"
    vdi = VDI_HOSTS[0]

    # Delivery + exploitation: valid login, no second factor (T1078/T1133)
    b.a(t0, victim, "EXTERNAL", GATEWAY, "Citrix", "Gateway", mfa=False,
        device=attacker_dev, geo=attacker_geo, label=1, phase="initial_access")
    b.a(t0 + timedelta(seconds=25), victim, GATEWAY, vdi, "Kerberos",
        "RemoteInteractive", label=1, phase="initial_access")

    servers = [f"SRV-{i:03d}" for i in range(b.cfg.n_servers)]
    unseen = [s for s in servers if s not in v.servers]
    foothold = list(rng.choice(unseen, size=6, replace=False))
    file_servers = foothold[:2]

    # Installation: remote tools on the first footholds (CISA AA23-353A)
    t = t0 + timedelta(hours=1)
    for host, tool in zip(foothold[:3], ["anydesk.exe", "screenconnect.windowsclient.exe",
                                          "splashtop.exe"]):
        t += timedelta(minutes=float(rng.uniform(10, 40)))
        b.a(t, victim, vdi, host, "NTLM", "Network", label=1, phase="tooling")
        b.p(t + timedelta(seconds=30), host, victim, tool, label=1, phase="tooling")

    # Credential abuse: Kerberos ticket burst, then a second, privileged account
    t += timedelta(hours=3)
    for host in rng.choice(servers, size=14, replace=False):
        t += timedelta(seconds=float(rng.uniform(2, 8)))
        b.a(t, victim, foothold[0], host, "Kerberos", "Network", label=1,
            phase="credential_abuse")
    b.p(t, foothold[0], victim, "rubeus.exe", label=1, phase="credential_abuse")

    # Lateral movement over the remaining days, partly with the stolen admin
    current = foothold[0]
    for d in range(1, days):
        day = (t0 + timedelta(days=d)).replace(hour=0, minute=0)
        for _ in range(int(rng.integers(4, 10))):
            ts = day + timedelta(hours=float(rng.choice([1, 2, 3, 13, 22])) +
                                 float(rng.uniform(0, 1)))
            acct = admin if (d > 2 and rng.random() < 0.5) else victim
            dst = str(rng.choice(foothold + list(rng.choice(servers, 3))))
            b.a(ts, acct, current, dst, "Kerberos", "Network", label=1,
                phase="lateral_movement")
            if rng.random() < 0.6:
                b.p(ts + timedelta(seconds=int(rng.integers(5, 60))), dst, acct,
                    str(rng.choice(["cmd.exe", "powershell.exe", "rclone.exe",
                                    "7z.exe"])), label=1, phase="lateral_movement")
            current = dst
        # Exfiltration from the file servers (double extortion)
        if d >= 2:
            for fs in file_servers:
                b.flow_extra.append(dict(day=day.date(), host=fs,
                                         bytes_out=float(rng.uniform(80, 220)) * 1e9,
                                         label=1))

    # Pre-encryption: tampering and the encryptor (T1490, T1486)
    if impact is not None:
        t = impact - timedelta(minutes=35)
        for host in foothold:
            b.a(t, admin, current, host, "Kerberos", "Network", label=1,
                phase="impact")
            for name in ["wevtutil.exe cl security", "vssadmin.exe delete shadows /all",
                         "bcdedit.exe /set recoveryenabled no", "killav.exe"]:
                t += timedelta(seconds=20)
                b.p(t, host, admin, name, label=1, phase="impact")
            b.p(impact, host, admin, "encryptor.exe", label=1, phase="impact")
    return foothold


def generate(cfg: SimConfig | None = None) -> LogSet:
    cfg = cfg or SimConfig()
    b = _Builder(cfg)
    users = _population(cfg, b.rng)

    day = TRAIN_START
    while day < END:
        for u in users.itertuples():
            _normal_day(b, u, day)
        day += timedelta(days=1)

    no_mfa = users[(~users.admin) & (~users.mfa)].user.tolist()
    admins = users[users.admin].user.tolist()
    rt_victim, chc_victim = no_mfa[0], no_mfa[1]

    _campaign(b, users, rt_victim, admins[0], datetime(2024, 1, 23, 23, 5),
              days=3, impact=datetime(2024, 1, 26, 3, 0), tag="redteam")
    _campaign(b, users, chc_victim, admins[1], INITIAL_ACCESS,
              days=(ENCRYPTION - INITIAL_ACCESS).days + 1, impact=ENCRYPTION,
              tag="chc")

    auth = pd.DataFrame(b.auth).sort_values("time", kind="stable").reset_index(drop=True)
    proc = pd.DataFrame(b.proc).sort_values("time", kind="stable").reset_index(drop=True)

    # Netflow: baseline outbound bytes per host per day, plus exfil spikes
    hosts = sorted(set(auth.dst) - {"EXTERNAL"})
    days = pd.date_range(TRAIN_START, END - timedelta(days=1)).date
    base = {h: b.rng.lognormal(mean=21.5, sigma=0.6) for h in hosts}
    flow = pd.DataFrame([dict(day=d, host=h,
                              bytes_out=base[h] * b.rng.lognormal(0, 0.35), label=0)
                         for d in days for h in hosts])
    if b.flow_extra:
        extra = pd.DataFrame(b.flow_extra)
        flow = flow.merge(extra, on=["day", "host"], how="left", suffixes=("", "_x"))
        flow["bytes_out"] += flow.bytes_out_x.fillna(0)
        flow["label"] = flow.label_x.fillna(flow.label).astype(int)
        flow = flow.drop(columns=["bytes_out_x", "label_x"])

    meta = dict(seed=cfg.seed, redteam_victim=rt_victim, chc_victim=chc_victim,
                initial_access=INITIAL_ACCESS, encryption=ENCRYPTION,
                test_start=TEST_START)
    return LogSet(auth=auth, process=proc, netflow=flow, users=users, meta=meta)
