"""Self-contained HTML report (inline SVG charts, no plotting library)."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

BLUE, LIGHT, INK, GREY = "#1f3a93", "#c9d4f2", "#111111", "#666666"


def _bar_chart(items, width=640, bar_h=22, fmt="{:.3f}"):
    vmax = max(v for _, v in items) or 1
    label_w = 300
    h = bar_h * len(items) + 10
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">']
    for i, (name, v) in enumerate(items):
        y = i * bar_h + 4
        w = (width - label_w - 70) * v / vmax
        out.append(f'<text x="{label_w - 8}" y="{y + 14}" text-anchor="end" font-size="12" fill="{INK}">{html.escape(name)}</text>')
        out.append(f'<rect x="{label_w}" y="{y + 2}" width="{w:.1f}" height="{bar_h - 8}" fill="{BLUE}"/>')
        out.append(f'<text x="{label_w + w + 6:.1f}" y="{y + 14}" font-size="11" fill="{GREY}">{fmt.format(v)}</text>')
    out.append("</svg>")
    return "".join(out)


def _risk_timeline(daily, initial, encryption, threshold, width=760, height=260):
    pad_l, pad_b, pad_t = 50, 50, 20
    n = len(daily)
    step = (width - pad_l - 20) / max(1, n - 1)
    lo = min(0.9, min(d["max_risk"] for d in daily))
    def y(v):
        return pad_t + (1 - (v - lo) / (1 - lo)) * (height - pad_t - pad_b)
    pts = " ".join(f"{pad_l + i * step:.1f},{y(d['max_risk']):.1f}" for i, d in enumerate(daily))
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">',
           f'<line x1="{pad_l}" x2="{width - 20}" y1="{y(threshold):.1f}" y2="{y(threshold):.1f}" stroke="{GREY}" stroke-dasharray="4 3"/>',
           f'<text x="{width - 22}" y="{y(threshold) - 4:.1f}" text-anchor="end" font-size="11" fill="{GREY}">alert threshold</text>']
    for i, d in enumerate(daily):
        x = pad_l + i * step
        if d["attack"]:
            out.append(f'<rect x="{x - step / 2:.1f}" y="{pad_t}" width="{step:.1f}" height="{height - pad_t - pad_b}" fill="{LIGHT}" opacity="0.6"/>')
        out.append(f'<text x="{x:.1f}" y="{height - pad_b + 16}" font-size="10" text-anchor="middle" fill="{GREY}">{d["day"][5:]}</text>')
        if d["day"] in (initial[:10], encryption[:10]):
            lab = "initial access" if d["day"] == initial[:10] else "encryption"
            out.append(f'<text x="{x:.1f}" y="{height - pad_b + 32}" font-size="10" text-anchor="middle" fill="{BLUE}">{lab}</text>')
    out.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    for i, d in enumerate(daily):
        out.append(f'<circle cx="{pad_l + i * step:.1f}" cy="{y(d["max_risk"]):.1f}" r="3" fill="{BLUE}"/>')
    for v in (lo, (lo + 1) / 2, 1.0):
        out.append(f'<text x="{pad_l - 6}" y="{y(v) + 4:.1f}" font-size="10" text-anchor="end" fill="{GREY}">{v:.2f}</text>')
    out.append(f'<text x="12" y="{height / 2}" font-size="11" fill="{GREY}" transform="rotate(-90 12 {height / 2})" text-anchor="middle">max daily risk</text>')
    out.append("</svg>")
    return "".join(out)


def _table(rows, cols, heads):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in heads)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
                   for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def write_report(result: dict, path: str | Path) -> Path:
    s = result["summary"]
    m = s["metrics"]
    mrows = []
    for k, label in [("isolation_forest", "Isolation Forest"), ("autoencoder", "Autoencoder"),
                     ("xgboost", "XGBoost"), ("ensemble", "Ensemble (weighted)")]:
        r = m[k]
        mrows.append(dict(model=label, recall=f"{r['recall']:.1%}", fpr=f"{r['fpr']:.4%}",
                          precision=f"{r['precision']:.1%}", tp=r["tp"], fp=r["fp"], fn=r["fn"]))
    rl = m["rule_layer"]
    tl = [dict(phase=t["phase"].replace("_", " "), events=t["events"],
               first=t["first_alert"] or "not detected", by=t["first_alert_by"] or "-",
               lead="-" if t["hours_before_encryption"] is None else f"{t['hours_before_encryption']} h")
          for t in s["timeline"]]
    q = result["queue"]
    top = q[q.label == 1].head(12)
    qrows = [dict(time=str(r.time)[:16], source=r.source, user=r.user, dst=r.dst,
                  risk=f"{r.risk:.3f}", why=r.reasons, action=r.action) for r in top.itertuples()]
    fp = q[q.label == 0].head(6)
    fprows = [dict(time=str(r.time)[:16], source=r.source, user=r.user, dst=r.dst,
                   risk=f"{r.risk:.3f}", why=r.reasons) for r in fp.itertuples()]

    css = f"""
    body{{font-family:'Times New Roman',Times,serif;color:{INK};background:#fff;max-width:900px;margin:0 auto;padding:24px 16px;line-height:1.45}}
    h1{{color:{BLUE};font-size:26px;margin-bottom:4px}} h2{{color:{BLUE};font-size:19px;margin-top:32px;border-bottom:1px solid {LIGHT};padding-bottom:4px}}
    .note{{border-left:3px solid {BLUE};padding:8px 12px;background:#f3f6fd;font-size:14px}}
    table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}} th{{background:{LIGHT};text-align:left}}
    th,td{{border:1px solid #b9c3e0;padding:4px 6px;vertical-align:top}}
    .kpi{{display:flex;gap:12px;flex-wrap:wrap}} .kpi div{{flex:1;min-width:150px;border:1px solid {LIGHT};padding:10px}}
    .kpi b{{display:block;font-size:24px;color:{BLUE}}} small{{color:{GREY}}}
    """
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>G73 UBA Detection Run</title>
<style>{css}</style></head><body>
<h1>G73 &middot; Compromised-Credential Intrusion Detector</h1>
<small>Prototype of the framework in Section 4 of the G73 case study &middot; generated {datetime.now():%d %b %Y %H:%M} &middot; seed {s['seed']}</small>
<p class="note"><b>Synthetic data.</b> These numbers come from a simulated network that replays the public
Change Healthcare timeline. They show that the pipeline works end to end. They are not a measurement of how
it would perform on real Change Healthcare or LANL logs, and the simulation was written by the same group
that built the detector, so the task is easier than a real network.</p>

<h2>Headline</h2>
<div class="kpi">
<div><b>{s['lead_time_hours'] if s['lead_time_hours'] is not None else '-'} h</b>first true alert before encryption</div>
<div><b>{m['ensemble']['recall']:.0%}</b>ensemble recall on attack logins</div>
<div><b>{m['ensemble']['fpr']:.3%}</b>ensemble false-positive rate</div>
<div><b>{s['false_alerts_per_day']}</b>false alerts per day (all layers)</div>
</div>
<p>Data split: train {s['period']['train']}, validation {s['period']['validation']}, test {s['period']['test']}.
Test window: {s['counts']['test']:,} authentication events, of which {s['counts']['test_attack']} belong to the
intrusion. Initial access {s['initial_access'][:16]}, encryption {s['encryption'][:16]}.
Ensemble weights {s['weights']}; FPR budget {s['fpr_budget']:.2%}.</p>

<h2>1. Detection timeline (per attack phase)</h2>
{_table(tl, ['phase', 'events', 'first', 'by', 'lead'], ['Phase', 'Attack events', 'First alert', 'Raised by', 'Before encryption'])}
{_risk_timeline(s['daily'], s['initial_access'], s['encryption'], s['threshold'])}
<small>Shaded days contain attack activity. The line is the highest ensemble risk score seen that day.</small>

<h2>2. Model comparison (test window, event level)</h2>
{_table(mrows, ['model', 'recall', 'fpr', 'precision', 'tp', 'fp', 'fn'], ['Model', 'Recall', 'FPR', 'Precision', 'TP', 'FP', 'FN'])}
<p>Rule layer (CISA AA23-353A indicators on process events): {rl['alerts']} alerts, {rl['true_alerts']} true, {rl['false_alerts']} false.</p>

<h2>3. What drives the score (mean |SHAP|, XGBoost)</h2>
{_bar_chart([(d['label'], d['mean_abs_shap']) for d in s['importance']])}

<h2>4. Alert queue - first true alerts</h2>
{_table(qrows, ['time', 'source', 'user', 'dst', 'risk', 'why', 'action'], ['Time', 'Source', 'Account', 'Host', 'Risk', 'Top reasons (SHAP)', 'Suggested action'])}

<h2>5. Sample false alerts</h2>
{_table(fprows, ['time', 'source', 'user', 'dst', 'risk', 'why'], ['Time', 'Source', 'Account', 'Host', 'Risk', 'Top reasons (SHAP)'])}
<p><small>Full outputs: alert_queue.csv, scored_events.csv, metrics.json in the same folder.</small></p>
</body></html>"""
    path = Path(path)
    path.write_text(doc, encoding="utf-8")
    return path
