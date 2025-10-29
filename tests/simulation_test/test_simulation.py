import numpy as np
from typing import Dict, Any,Tuple
import os
import sys
import config

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_reference

def compare_seq_parameters(actual: Dict[str, Any], reference: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Compare MR sequence simulation outputs against reference data across multiple accuracy levels.
    
    This function validates that MR sequence simulations produce consistent results by comparing
    actual simulation outputs with reference data. It tests multiple accuracy levels defined in
    the configuration to ensure robust validation across different simulation precision settings.

    Parameters:
        actual (Dict[str, Any]): Dictionary containing actual simulation results from .npz file.
                                Expected keys:
                                - 'signal': Complex signal data array for each accuracy level
                                - 'timing_results': Execution time for each accuracy level
        reference (Dict[str, Any]): Dictionary containing reference simulation results from .npz file.
                                  Must have same structure as 'actual' parameter.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if all comparisons pass thresholds, False otherwise
            - str: Diagnostic message indicating pass status or specific failure details
    """
    dt = actual.get("timing_results")
    signal = actual.get("signal")

    ref_dt = reference.get("timing_results")
    ref_signal = reference.get("signal")

    for i in range(len(config.ACC_ARRAY)):
        acc = config.ACC_ARRAY[i]
        mag_rmse = np.sqrt(np.mean((np.abs(signal[i]) - np.abs(ref_signal[i])) ** 2)) / np.mean(np.abs(signal[i]))
        phase_rmse = np.sqrt(np.mean((np.angle(signal[i]) - np.angle(ref_signal[i])) ** 2))
        dt_diff_percent = ((dt[i] - ref_dt[i]) / ref_dt[i]) * 100

        if mag_rmse > config.MAG_NRMSE:
            return False, f"Mag RMSE too high ({mag_rmse:.5f}) at acc={acc:.5f}"
        
        if phase_rmse > config.PHASE_NRMSE:
            return False, f"Phase RMSE too high ({phase_rmse:.5f}) at acc={acc:.5f}"

        if dt_diff_percent > config.DT_PERCENT:
            return False, f"Runtime deviation too high ({dt_diff_percent:.2f}%) at acc={acc:.5f}"

    return True, "All tests passed."

def test_all_sequence_simulations() -> None:
    """
    Run comprehensive automated tests for all MR sequence simulations.
    """
    seq_files = config.GetSeqFiles()
    for seq_file in seq_files:
        seq_base_name = os.path.basename(seq_file)
        ref_file = os.path.join(config.REF_FOLDER, f"{seq_base_name}.npz")
        actual_file = os.path.join(config.ACTUAL_FOLDER, f"{seq_base_name}.npz")

        ref_data = load_reference(ref_file)
        act_data = load_reference(actual_file)

        passed, message = compare_seq_parameters(act_data, ref_data)

        assert passed, message + f" (Sequence: {seq_base_name} Failed ❌)"
        print(f"Sequence: {seq_base_name} Passed ✅")
        
    print(f"All sequences passed ✅")
if __name__ == "__main__":
    test_all_sequence_simulations()