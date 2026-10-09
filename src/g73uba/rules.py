"""Rule layer: known ALPHV/BlackCat indicators (CISA AA23-353A, AttackIQ).

Some behaviours are rare enough in normal use that a rule beats a model
(report, Section 4.6). Rules run on process events; each hit carries a
severity and the ATT&CK technique it points to.
"""

from __future__ import annotations

import pandas as pd

HIGH = [
    ("vssadmin.exe delete shadows", "T1490", "Shadow copies deleted"),
    ("bcdedit.exe /set recoveryenabled no", "T1490", "Windows recovery disabled"),
    ("wevtutil.exe cl", "T1070.001", "Windows event log cleared"),
    ("killav", "T1562.001", "Security tool killer"),
    ("encryptor", "T1486", "Encryptor binary"),
    ("rubeus", "T1558", "Kerberos ticket tool"),
]
# Legitimate admins use these, so they only fire for non-admin accounts
MEDIUM = [
    ("anydesk", "T1219", "Remote-access tool"),
    ("screenconnect", "T1219", "Remote-access tool"),
    ("splashtop", "T1219", "Remote-access tool"),
    ("rclone", "T1567", "File-sync / exfiltration tool"),
]


def match(process: str | float, is_admin: bool) -> tuple[str, str, str] | None:
    if not isinstance(process, str):
        return None
    p = process.lower()
    for needle, tech, why in HIGH:
        if needle in p:
            return "high", tech, why
    if not is_admin:
        for needle, tech, why in MEDIUM:
            if needle in p:
                return "medium", tech, why
    return None


def run_rules(process: pd.DataFrame, admins: set[str]) -> pd.DataFrame:
    rows = []
    for r in process.itertuples(index=False):
        m = match(r.process, r.user in admins)
        if m:
            rows.append(dict(time=r.time, host=r.host, user=r.user,
                             process=r.process, severity=m[0], technique=m[1],
                             reason=m[2], label=r.label, phase=r.phase))
    return pd.DataFrame(rows, columns=["time", "host", "user", "process", "severity",
                                       "technique", "reason", "label", "phase"])
