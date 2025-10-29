# MRzero-Core Tests

## Simulation Test

GitHub workflow that validates changes to the MRzero-Core simulation engine don't break existing functionality. It compares current simulation results against the main branch baseline across multiple accuracy levels. Can also be run locally.

**What it tests:**
- Signal magnitude, phase, and timing accuracy
- Multiple MR sequence types (FID, SE, STE, FLASH, EPI, DWI, RARE, TSE, etc.)
- 11 different accuracy levels (0.6 to 0.00001)

## Running Tests Locally

Use the PowerShell script in `simulation_test/` folder (run from project root directory):

```powershell
# Test all notebooks from config
.\tests\simulation_test\test_simulation.ps1

# Test specific notebook
.\tests\simulation_test\test_simulation.ps1 documentation/playground_mr0/mr0_FLASH_2D_seq.ipynb
```

**Note**: The script uses a virtual environment (`.test_venv`) to isolate test dependencies from your global packages. The first run will create the virtual environment and install dependencies. Subsequent runs will reuse the existing environment for faster startup.

**Virtual Environment Management:**
- The virtual environment (`.test_venv`) is created in your project root directory
- You can safely cancel tests (Ctrl+C) without affecting your global Python packages
- To recreate the virtual environment, delete the `.test_venv` folder and run the script again

**Requirements for testing all notebooks:**
- Notebooks must be listed in `config.py` NOTEBOOKS_TO_TEST
- Each notebook must contain a `seq_` function
- Function must generate a valid MR sequence
- All parameters in the `seq_` function should have default values

**Requirements for testing specific notebook:**
- Notebook must contain a `seq_` function
- Function must generate a valid MR sequence
- All parameters in the `seq_` function should have default values
- Notebook path must be valid (doesn't need to be in config)

## Configuration

The `config.py` file controls test behavior:
- **NOTEBOOKS_TO_TEST**: List of notebooks to test when running all tests
- **ACC_ARRAY**: Accuracy levels tested (0.6 to 0.00001)
- **Thresholds**: 
  - **MAG_NRMSE (0.01)**: Maximum allowed normalized root mean square error for signal magnitude (1% tolerance)
  - **PHASE_NRMSE (0.01)**: Maximum allowed normalized root mean square error for signal phase (1% tolerance)
  - **DT_PERCENT (20%)**: Maximum allowed execution time deviation from reference (20% tolerance)
- **Folders**: Reference and actual data storage locations

**Threshold Purpose**: These thresholds ensure simulation accuracy and performance remain within acceptable bounds when comparing current implementation against the main branch baseline.

**Note**: If a test fails due to timing deviation and the sequence is very fast(FID for example), consider running the test again. Fast sequences can have timing fluctuations between runs that may cause false failures.

Modify `config.py` to add/remove notebooks or adjust test parameters.

## Playground Test

GitHub workflow that tests Jupyter notebook execution in a Colab-like environment to ensure notebooks work online.
