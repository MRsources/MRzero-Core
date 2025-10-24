import os

MAG_NRMSE = 0.01
PHASE_NRMSE = 0.01
DT_PERCENT = 20
REF_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_files")
ACTUAL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "actual_files")
ACC_ARRAY = [0.6, 0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.00001]

# Centralized list of notebooks to test
NOTEBOOKS_TO_TEST = [
    "documentation/playground_mr0/mr0_FID_seq.ipynb",
    "documentation/playground_mr0/mr0_SE_CPMG_seq.ipynb",
    "documentation/playground_mr0/mr0_STE_3pulses_5echoes_seq.ipynb",
    "documentation/playground_mr0/mr0_FLASH_2D_seq.ipynb",
    "documentation/playground_mr0/mr0_EPI_2D_seq.ipynb",
    "documentation/playground_mr0/mr0_DWI_SE_EPI.ipynb",
    "documentation/playground_mr0/mr0_diffusion_prep_STEAM_2D_seq.ipynb",
    "documentation/playground_mr0/mr0_RARE_2D_seq.ipynb",
    "documentation/playground_mr0/mr0_TSE_2D_multi_shot_seq.ipynb",
    "documentation/playground_mr0/mr0_GRE_to_FLASH.ipynb",
    "documentation/playground_mr0/mr0_bSSFP_2D_seq.ipynb",
    "documentation/playground_mr0/mr0_DREAM_STE_seq.ipynb",
]

def GetSeqFiles():
    directory = "tests/simulation_test/seq_files"
    files = [os.path.join(directory, entry.name) for entry in os.scandir(directory) if entry.is_file()]
    return files