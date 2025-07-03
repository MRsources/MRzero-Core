import os
import numpy as np
from config import NOTE_BOOK_PATH,REF_FILE
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import exec_notebook

def generate_ref_file() -> None :
    """
    Executes the specified Jupyter notebook and extracts required variables.
    Validates that the notebook produces 'space_3d', 'dt', and 'acc_array'.
    Saves these variables to a reference .npz file for future testing/comparison.
    
    If any of the required variables are missing, the script exits with an error message.
    """

    notebook_name_space = exec_notebook(NOTE_BOOK_PATH)

    required_vars = ["space_3d", "dt", "acc_array"]
    missing = [var for var in required_vars if notebook_name_space.get(var) is None]

    if missing:
        sys.exit(f"Notebook did not produce required variables: {', '.join(missing)}")

    # Extract arrays
    space_3d = notebook_name_space.get("space_3d")
    dt = notebook_name_space.get("dt")
    acc_array = notebook_name_space.get("acc_array")
    
    np.savez(REF_FILE, space_3d=space_3d, dt=dt, acc_array=acc_array)

if __name__ == "__main__":
    generate_ref_file()