import warnings
import numpy as np
import glob
import os
import shutil
from nbformat import read
import matplotlib
from typing import Dict, Any
import sys
import io
from contextlib import contextmanager

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
    cell_count = 0
    for cell in nb.cells:
        if cell.cell_type == "code":
            # Filter out lines starting with "!"
            filtered_lines = [
                line for line in cell.source.splitlines()
                if not line.strip().startswith("!")
            ]
            filtered_source = "\n".join(filtered_lines)

            # Redirect stdout to suppress prints
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with suppress_output():  # your context manager to silence prints & native output
                        exec(filtered_source, namespace, namespace)
            except Exception as e:
                # Restore stdout before printing errors
                sys.stdout = old_stdout
                sys.exit(f"Error executing cell {cell_count}: {e} at {nb_path}")
            else :
                sys.stdout = old_stdout
            cell_count += 1

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
        ".ipynb_checkpoints","*.seq"
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
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}

@contextmanager
def suppress_output():
    try:
        fd_stdout = sys.__stdout__.fileno()
        fd_stderr = sys.__stderr__.fileno()
    except Exception:
        # If fileno() is not supported, just yield without suppressing
        yield
        return

    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    saved_stdout = os.dup(fd_stdout)
    saved_stderr = os.dup(fd_stderr)

    try:
        os.dup2(null_fds[0], fd_stdout)
        os.dup2(null_fds[1], fd_stderr)
        yield
    finally:
        os.dup2(saved_stdout, fd_stdout)
        os.dup2(saved_stderr, fd_stderr)
        os.close(null_fds[0])
        os.close(null_fds[1])
        os.close(saved_stdout)
        os.close(saved_stderr)