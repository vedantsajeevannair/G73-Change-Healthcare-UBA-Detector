"""Build the one-file g73-uba executable for the current OS (python build_exe.py)."""
import PyInstaller.__main__

PyInstaller.__main__.run([
    "run_g73.py",
    "--noconfirm", "--onefile", "--name", "g73-uba",
    "--paths", "src",
    "--collect-binaries", "xgboost", "--collect-data", "xgboost",
    "--exclude-module", "tkinter", "--exclude-module", "matplotlib",
    "--exclude-module", "pytest", "--exclude-module", "hypothesis",
    "--exclude-module", "xgboost.testing",
])
