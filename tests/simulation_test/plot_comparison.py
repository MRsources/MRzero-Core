"""
Plot reference (main branch) vs. actual (current branch) simulation results.

Usage: python plot_comparison.py <notebook_path> <output_folder>

Writes <name>.png and <name>.json (summary used for the PR comment) into the
output folder. Tolerates missing data so it can run after failed steps.
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_reference
from test_simulation import compare_seq_parameters, compute_metrics


def plot_comparison(name, actual, reference, output_folder):
    mag_rmse, phase_rmse, dt_diff = compute_metrics(actual, reference)
    # Show the accuracy level where the signals differ the most
    i = int(np.argmax(mag_rmse + phase_rmse))
    acc = config.ACC_ARRAY[i]
    ref = np.asarray(reference["signal"][i]).reshape(-1)
    act = np.asarray(actual["signal"][i]).reshape(-1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(f"{name}  (acc={acc:g}, worst level)")

    ax = axes[0, 0]
    ax.plot(np.abs(ref), lw=1, label="main (before)")
    ax.plot(np.abs(act), lw=1, ls="--", label="PR (after)")
    ax.set_title("Magnitude")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(np.angle(ref), lw=1, label="main (before)")
    ax.plot(np.angle(act), lw=1, ls="--", label="PR (after)")
    ax.set_title("Phase")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(np.abs(act - ref), lw=1, color="C3", label="|after - before|")
    ax.plot(np.abs(act) - np.abs(ref), lw=1, color="C2", alpha=0.7, label="|after| - |before|")
    ax.set_title("Difference")
    ax.set_xlabel("sample")
    ax.legend()

    ax = axes[1, 1]
    ax.loglog(config.ACC_ARRAY, np.maximum(mag_rmse, 1e-12), "o-", label="mag NRMSE")
    ax.loglog(config.ACC_ARRAY, np.maximum(phase_rmse, 1e-12), "s-", label="phase RMSE")
    ax.axhline(config.MAG_NRMSE, color="gray", ls=":", label="threshold")
    ax.invert_xaxis()
    ax.set_title("Error vs. accuracy")
    ax.set_xlabel("accuracy")
    ax.legend()

    fig.savefig(os.path.join(output_folder, f"{name}.png"), dpi=90)
    plt.close(fig)

    passed, message = compare_seq_parameters(actual, reference)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "message": message,
        "image": f"{name}.png",
        "max_mag_nrmse": float(mag_rmse.max()),
        "max_phase_rmse": float(phase_rmse.max()),
        "max_dt_percent": float(dt_diff.max()),
    }


def main():
    notebook_path, output_folder = sys.argv[1:3]
    os.makedirs(output_folder, exist_ok=True)
    name = os.path.splitext(os.path.basename(notebook_path))[0]
    npz_name = f"{name}.seq.npz"

    try:
        reference = load_reference(os.path.join(config.REF_FOLDER, npz_name))
        actual = load_reference(os.path.join(config.ACTUAL_FOLDER, npz_name))
        summary = plot_comparison(name, actual, reference, output_folder)
    except Exception as e:
        summary = {"name": name, "status": "error", "message": f"No comparison possible: {e}"}

    with open(os.path.join(output_folder, f"{name}.json"), "w") as f:
        json.dump(summary, f)
    print(f"{name}: {summary['status']} - {summary['message']}")


if __name__ == "__main__":
    main()
