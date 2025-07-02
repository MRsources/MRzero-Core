import numpy as np
import glob
import os
import shutil
from nbformat import read
import matplotlib
from typing import Dict, Any

def exec_notebook(nb_path : str) -> Dict[str, Any]: 
    """
    Load and execute a Jupyter notebook, returning a dictionary representing 
    the namespace (variables and functions) defined within the notebook.

    Parameters:
        nb_path (str): Path to the Jupyter notebook file (.ipynb).

    Returns:
        Dict[str, Any]: A dictionary containing the variables defined during notebook execution.
    """

    matplotlib.use('Agg') # Silence matplotlib outputs

    # Load notebook
    with open(nb_path) as f:
        nb = read(f, as_version=4)

    namespace = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            try:
                exec(cell.source, namespace, namespace)
            except Exception as e:
                print("Error executing cell:", e)

    cleanup_generated_files()
    return namespace

def cleanup_generated_files() -> None:
    """
    Remove temporary or generated files matching specific patterns 
    (e.g., images, data files, checkpoints) from the current directory.
    """

    patterns = [
        "*.png", "*.jpg", "*.jpeg", "*.svg", "*.gif",
        "*.json", "*.npz", "*.npy", "*.mat",
        ".ipynb_checkpoints",".seq"
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass

def load_reference(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Load a numpy .npz file and return its contents as a dictionary.

    Parameters:
        npz_path (str): Path to the .npz reference file.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping variable names to arrays stored in the .npz file.
    """
    with np.load(npz_path) as data:
        return dict(data)