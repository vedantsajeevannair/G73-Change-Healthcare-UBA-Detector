# G73 – Compromised-Credential Intrusion Detector (prototype)

Implementation for the Cyber Security CA-2 case study of **Group 73**:
*"Detecting a Compromised-Credential Ransomware Intrusion: A Case Study of the
2024 Change Healthcare Attack Using User Behaviour Analytics and Machine Learning"*.

| Name | PRN |
|---|---|
| Vedant Nair | 23070122230 |
| Samartha Shreshta | 23070122189 |
| Deepak Raju Rathod | 23070122175 |

**GitHub:** https://github.com/vedantsajeevannair/G73-Change-Healthcare-UBA-Detector

This program is a working prototype of the detection framework in Section 4 of
the report (Figure 3 and the six-step workflow of Section 4.10). It runs the
full pipeline: log generation, per-user sessions, the Table 4 behavioural
features, three models, an ensemble, a rule layer, SHAP explanations and an
alert queue for a SOC analyst.

> **Important – synthetic data.** The Change Healthcare logs are not public.
> The program therefore generates a simulated enterprise network and injects an
> intrusion that follows the public Change Healthcare timeline (12 Feb 2024
> stolen-credential login through a Citrix portal without MFA → remote tools →
> Kerberos abuse → nine days of lateral movement and data theft → security-tool
> tampering and encryption on 21 Feb 2024). The numbers it prints show that the
> pipeline works end to end. They are **not** a measurement of how it would
> perform on real Change Healthcare or LANL data, and they are easier than a real
> network because the same group wrote the simulation and the detector.

---

## 1. Quick start – executable (no Python needed)

The zip contains ready-built executables in `exe/`:

| OS | File |
|---|---|
| Windows 10/11 (64-bit) | `exe/windows/g73-uba.exe` |
| macOS (Apple Silicon) | `exe/macos/g73-uba` |

**Windows:** open *Command Prompt* in the `exe\windows` folder and run

```
g73-uba.exe --open
```

**macOS:** open *Terminal* in the `exe/macos` folder and run

```
chmod +x g73-uba
./g73-uba --open
```

(If macOS blocks it the first time: System Settings → Privacy & Security →
*Open Anyway*.)

The run takes about 10–30 seconds (the first launch can take a minute while the
executable unpacks and the OS scans it). It creates an `output/` folder next to the
executable and, with `--open`, opens `output/report.html` in the browser.

## 2. Run from source (Python 3.10 – 3.13)

```bash
cd G73_Implementation
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
g73-uba --open                # or: python -m g73uba --open
```

On macOS, XGBoost needs the OpenMP runtime: `brew install libomp`.

Run the tests:

```bash
pip install pytest
python -m pytest -q
```

Build the executable yourself:

```bash
pip install pyinstaller
python build_exe.py
# result: dist/g73-uba(.exe)
```

The GitHub Actions workflow in `.github/workflows/build.yml` runs the tests and
builds the Windows, macOS and Linux executables on every push.

### Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--seed N` | 7 | Random seed of the simulated network |
| `--fpr X` | 0.002 | False-positive budget that sets the alert threshold (0.2 %) |
| `--out DIR` | `output` | Output folder |
| `--open` | off | Open the HTML report when finished |

## 3. Outputs

| File | Content |
|---|---|
| `report.html` | Self-contained report: detection timeline per attack phase, model comparison, SHAP feature importance, first true alerts with reasons, sample false alerts |
| `alert_queue.csv` | Every alert (ML ensemble and rule layer) with risk, top SHAP reasons and suggested SOC action |
| `scored_events.csv` | Every test-window login with its features and the score of each model |
| `metrics.json` | Recall, FPR, precision per model, ensemble weights, lead time before encryption |

## 4. How it maps to the report

| Report | Code |
|---|---|
| Step 1 – data collection (gateway, Windows auth, process/EDR, netflow) | `src/g73uba/simulate.py` |
| Steps 2–3 – session building, Table 4 features | `src/g73uba/features.py` |
| Step 4 – Isolation Forest, autoencoder, XGBoost (Sections 2.4, 3.4) | `src/g73uba/models.py` |
| Step 5 – ensemble risk score + SHAP | `models.py` (`ensemble`, `explain`) |
| Rule layer – CISA AA23-353A ALPHV indicators (Section 4.6) | `src/g73uba/rules.py` |
| Step 6 – alert queue and suggested response | `src/g73uba/pipeline.py` |
| Recall and FPR (Section 3.3) | `pipeline.py` (`_metrics`) |
| HTML report | `src/g73uba/report.py` |
| Command line | `src/g73uba/cli.py` |

**Data split.** Training month 15 Jan – 28 Jan 2024 (after a one-week
warm-up so the history features have a baseline; contains a labelled
red-team exercise that gives XGBoost its positive examples, the same role the
LANL red-team file plays in Kushwaha et al.), validation week 29 Jan – 4 Feb
(normal activity only; used to calibrate scores and set the alert threshold),
test window 5 Feb – 22 Feb (contains the Change Healthcare replay).

**Features (Table 4).** Remote login, was source logged on, other interactive
logins, was process run (+ process rarity), previous login-hour fraction,
failed logins in the last hour, fraction from same source — from Kushwaha et
al. — plus the group's proposed gateway new-device / no-MFA flags and outbound
volume versus baseline. Security-tool tampering is handled by the rule layer.

**Models.** Isolation Forest and the autoencoder are trained on normal
training events only. The autoencoder is a 16-4-16 bottleneck MLP from
scikit-learn (the report's stack lists PyTorch; scikit-learn keeps the
executable small and gives the same reconstruction-error score). XGBoost uses
`scale_pos_weight` for the class imbalance. Each score is converted to a
percentile against the validation week, and the ensemble is a weighted mean;
the weights are chosen by grid search on the training month using out-of-fold
XGBoost scores. SHAP values come from XGBoost's built-in TreeSHAP
(`pred_contribs=True`), which gives the same values as the `shap` library for
tree models.

## 5. Project structure

```
G73_Implementation/
├── README.md
├── requirements.txt
├── pyproject.toml
├── run_g73.py                 # PyInstaller entry point
├── build_exe.py               # builds the executable
├── src/g73uba/
│   ├── simulate.py            # step 1
│   ├── features.py            # steps 2–3
│   ├── models.py              # steps 4–5
│   ├── rules.py               # rule layer
│   ├── pipeline.py            # six-step workflow
│   ├── report.py              # HTML report
│   └── cli.py                 # command line
├── tests/test_pipeline.py
├── sample_output/             # output of the default run (seed 7)
├── exe/                       # built executables
└── .github/workflows/build.yml
```

## 6. Limitations

- Synthetic data only; host and user names are invented. No real Change
  Healthcare data was used.
- The intrusion follows the public timeline; details that are not public
  (source of the credentials, exact tools, exfiltration method) are assumed
  from the CISA advisory on ALPHV/BlackCat.
- The features that separate attack from normal behaviour were designed by the
  same group that wrote the simulation, so detection here is easier than it
  would be on a real network.
- It does not connect to Wazuh, Zeek or Suricata; those parts of the proposed
  stack (Table 6) are not implemented.

## 7. References (from the report)

- [1] CISA, FBI & HHS, *#StopRansomware: ALPHV Blackcat*, Advisory AA23-353A, 2024.
- [3] A. Witty, Testimony before the U.S. Senate Committee on Finance, 1 May 2024.
- [31] D. Kushwaha et al., "Lateral movement detection using user behavioral analysis", arXiv:2208.13524, 2022.
- [32] A. D. Kent, *Comprehensive, multi-source cyber-security events*, LANL, 2015.
- [33] F. T. Liu, K. M. Ting, Z.-H. Zhou, "Isolation forest", IEEE ICDM, 2008.
- [34] S. M. Lundberg, S.-I. Lee, "A unified approach to interpreting model predictions", NeurIPS, 2017.
- [35] T. Chen, C. Guestrin, "XGBoost: A scalable tree boosting system", ACM KDD, 2016.
