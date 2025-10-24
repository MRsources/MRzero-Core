import json
import sys
import os
from pathlib import Path
import subprocess
import tempfile
import config

def extract_and_run_seq_function(notebook_path):
    """
    Extract the seq_ function from a Jupyter notebook and run it.
    
    Args:
        notebook_path (str): Path to the Jupyter notebook file
    """
    notebook_file = Path(notebook_path)
    
    if not notebook_file.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return
    
    if not notebook_file.suffix == '.ipynb':
        print(f"❌ Not a Jupyter notebook: {notebook_path}")
        return
    
    print(f"📓 Processing notebook: {notebook_file.name}")
    
    try:
        with open(notebook_file, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
        
        # Find the sequence function
        seq_function_name = None
        seq_function_code = None
        
        for cell in notebook_data.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ''.join(cell.get('source', []))
                
                if 'def seq_' in source:
                    # Extract function name
                    start = source.find('def seq_')
                    end = source.find('(', start)
                    if end != -1:
                        seq_function_name = source[start+4:end].strip()
                        seq_function_code = source
                        break
        
        if not seq_function_name:
            print("❌ No sequence function found in this notebook")
            return
        
        # Create a temporary Python file to run the function
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            # Write the function code
            temp_file.write(f"import pypulseq as pp\nimport numpy as np\nimport pickle\nimport torch\n")
            temp_file.write(seq_function_code)
            temp_file.write(f"\n\n# Test the function\nif __name__ == '__main__':\n")
            temp_file.write(f"    # You can add test parameters here\n")
            temp_file.write(f"    seq = {seq_function_name}()\n")
            temp_file.write(f"    seq.write(f'tests/simulation_test/seq_files/{notebook_file.name.replace('.ipynb', '.seq')}')\n")
            
        
        temp_file_path = temp_file.name
        
        # Run the Python file
        try:
            result = subprocess.run([sys.executable, temp_file_path], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Function executed successfully!")
            else:
                print("❌ Function execution failed!")
                if result.stderr:
                    print("📤 Error:")
                    print(result.stderr)
                    
        except subprocess.TimeoutExpired:
            print("⏰ Function execution timed out")
        except Exception as e:
            print(f"❌ Error running function: {e}")
        
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass
            
    except Exception as e:
        print(f"❌ Error processing notebook: {e}")

def main():
    args = sys.argv[1:]
    notebook_paths = config.NOTEBOOKS_TO_TEST
    
    if args:
        notebook_paths = args

    for notebook_path in notebook_paths:
        if not os.path.exists(notebook_path):
            print(f"❌ Notebook not found: {notebook_path}")
            continue
        extract_and_run_seq_function(notebook_path)

if __name__ == "__main__":
    main()