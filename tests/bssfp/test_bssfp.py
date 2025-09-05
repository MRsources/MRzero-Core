import numpy as np
import pathlib
from typing import Dict, Any,Tuple
import os
import sys
from config import NOTE_BOOK_PATH,REF_FILE,MAG_NRMSE,PHASE_NRMSE,DT_PERCENT

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import exec_notebook,load_reference

def compare_bssfp_outputs(actual: Dict[str, Any], reference: Dict[str, Any], thresholds: Dict[str, float]) -> Tuple[bool, str]:
    """
    Compare bSSFP simulation outputs from a notebook against reference data.

    Parameters:
        actual (Dict[str, Any]): Dictionary of variables from the executed notebook.
        reference (Dict[str, Any]): Dictionary of reference data (from .npz file).
        thresholds (Dict[str, float]): Thresholds for normalized magnitude rmse, phase rmse, and runtime deviation percentage.
            Expected keys: "mag_nrmse", "phase_nrmse", "dt_percent".

    Returns:
        Tuple[bool, str]: (passed, message), where `passed` indicates whether the comparison was successful,
                          and `message` provides diagnostic information.
    """
    acc_array = actual.get("acc_array")
    dt = actual.get("dt")
    space_3d = actual.get("space_3d")

    ref_acc_array = reference.get("acc_array")
    ref_dt = reference.get("dt")
    ref_space_3d = reference.get("space_3d")

    if((acc_array is None) or len(acc_array) == 0) : return False, "Missing required variable acc_array in notebook output."
    if((dt is None) or len(dt) == 0) : return False, "Missing required variable dt in notebook output."
    if((space_3d is None) or len(space_3d) == 0) : return False, "Missing required variable space_3d in notebook output."

    # Check lengths match
    if len(acc_array) != len(ref_acc_array):
        return False, "acc_array length mismatch."

    for i in range(len(acc_array)):
        acc = acc_array[i]
        mag_rmse = np.sqrt(np.mean((np.abs(space_3d[i]) - np.abs(ref_space_3d[i])) ** 2)) / np.mean(np.abs(space_3d[i]))
        phase_rmse = np.sqrt(np.mean((np.angle(space_3d[i]) - np.angle(ref_space_3d[i])) ** 2))
        dt_diff_percent = ((dt[i] - ref_dt[i]) / ref_dt[i]) * 100

        if mag_rmse > thresholds["mag_nrmse"]:
            return False, f"Mag RMSE too high ({mag_rmse:.5f}) at acc={acc:.5f}"
        
        if phase_rmse > thresholds["phase_nrmse"]:
            return False, f"Phase RMSE too high ({phase_rmse:.5f}) at acc={acc:.5f}"
        
        if dt_diff_percent > thresholds["dt_percent"]:
            return False, f"Runtime deviation too high ({dt_diff_percent:.2f}%) at acc={acc:.5f}"

    return True, "All tests passed."

def test_bssfp_simulation() -> None:
    """
    Run an automated test for bSSFP simulation by executing a notebook and comparing its output 
    to a reference dataset using defined accuracy and timing thresholds.
    
    Raises:
        AssertionError: If the notebook is not found,
                        or if the output comparison fails the defined thresholds.
    """
    notebook_path = "documentation/playground_mr0/mr0_bSSFP_2D_seq.ipynb"
    assert pathlib.Path(notebook_path).is_file(), f"Notebook file not found: {notebook_path}" 

    assert pathlib.Path(REF_FILE).is_file(), f"Reference file not found: {REF_FILE}" 

    thresholds = {
        "mag_nrmse": MAG_NRMSE,
        "phase_nrmse": PHASE_NRMSE,
        "dt_percent": DT_PERCENT,
    }

    actual = exec_notebook(NOTE_BOOK_PATH)
    reference = load_reference(REF_FILE)
    passed, message = compare_bssfp_outputs(actual, reference, thresholds)
    assert passed, message 

if __name__ == "__main__":
    test_bssfp_simulation()