import importlib
import os

import numpy as np


def _numpy_major_version():
    version = np.__version__.split(".", 1)[0]
    return int(version)


def ensure_pinocchio_compatible():
    if _numpy_major_version() < 2:
        return
    if os.environ.get("PINOCCHIO_NUMPY2_OK") == "1":
        return

    raise RuntimeError(
        "Detected numpy "
        f"{np.__version__}, but this project's Pinocchio stack is not compatible with NumPy 2.x "
        "and may segfault during import.\n"
        "Fix the environment with:\n"
        "  /home/unitree/miniforge3/bin/conda install -n hand_eye_calib --offline --yes --force-reinstall "
        "numpy=1.26.4 pinocchio=3.1.0 eigenpy=3.8.0 hpp-fcl=2.4.5\n"
        "If you have rebuilt Pinocchio against NumPy 2.x already, set PINOCCHIO_NUMPY2_OK=1 to skip this guard."
    )


def import_pinocchio():
    ensure_pinocchio_compatible()
    return importlib.import_module("pinocchio")


def import_pinocchio_casadi():
    ensure_pinocchio_compatible()
    return importlib.import_module("pinocchio.casadi")


def import_meshcat_visualizer():
    ensure_pinocchio_compatible()
    visualize = importlib.import_module("pinocchio.visualize")
    return visualize.MeshcatVisualizer
