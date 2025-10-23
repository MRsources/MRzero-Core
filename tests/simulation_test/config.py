import os

MAG_NRMSE = 0.01
PHASE_NRMSE = 0.01
DT_PERCENT = 50
REF_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_files")
ACTUAL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "actual_files")
ACC_ARRAY = [0.6, 0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.00001]

def GetSeqFiles():
    directory = "tests/simulation_test/seq_files"
    files = [os.path.join(directory, entry.name) for entry in os.scandir(directory) if entry.is_file()]
    return files