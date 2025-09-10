import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
import argparse
import sys

def run_notebook(notebook_path):
    """
    Runs a Jupyter notebook and checks if it executes without errors.
    
    Args:
        notebook_path (str): Path to the notebook file.
        
    Returns:
        bool: True if notebook ran successfully, False if there was an error.
    """
    try:
        # Load notebook
        with open(notebook_path) as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Set up the executor
        executor = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        # Execute notebook
        executor.preprocess(notebook, {'metadata': {'path': './'}})
        
        print(f"[SUCCESS] Notebook '{notebook_path}' ran successfully.")
        return True

    except CellExecutionError as e:
        print(f"[ERROR] Error executing the notebook '{notebook_path}':\n{e}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error with notebook '{notebook_path}':\n{e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Jupyter notebook and check if it executes correctly.")
    parser.add_argument("notebook_path", type=str, help="Path to the Jupyter notebook file")
    args = parser.parse_args()

    success = run_notebook(args.notebook_path)
    if not success:
        sys.exit(1)  # Exit with error code if notebook fails