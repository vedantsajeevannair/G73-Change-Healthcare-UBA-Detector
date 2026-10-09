"""Entry script used by PyInstaller to build the g73-uba executable."""
import multiprocessing
import sys

from g73uba.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
