import os
from timeit import timeit
import numpy as np
import sys
import MRzeroCore as mr0
import config
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import suppress_output


class SimulationParametersComputer:
    def __init__(self, seq_file):
        self.__seq = mr0.Sequence.import_file(seq_file)
        phantom = self.__load_phantom()
        self.__data = phantom.build()
        with suppress_output():
            self.__graph = mr0.compute_graph(self.__seq, self.__data, 2000, 1e-5)

    def __load_phantom(self):
        return mr0.util.load_phantom()

    def generate_simulation_parameters(self):
        timing_results = np.zeros(len(config.ACC_ARRAY))
        signal_list = np.empty(len(config.ACC_ARRAY), dtype=object)
        for i, acc in enumerate(tqdm(config.ACC_ARRAY, desc="Generating simulation parameters")):
            signal, total_time = self.__get_seq_parameters(acc)
            timing_results[i] = total_time
            signal_list[i] = signal.numpy()
        return timing_results, signal_list
    
    def __get_seq_parameters(self, accuracy):
        signal = self.__execute_graph(accuracy)
        total_time = timeit(lambda:  self.__execute_graph(accuracy), number=1)
        return signal, total_time

    def __execute_graph(self, accuracy):
        if torch.cuda.is_available():
            return mr0.execute_graph(
                self.__graph, self.__seq.cuda(), self.__data.cuda(), accuracy, accuracy,
                print_progress=False
            ).cpu()
        else:
            return mr0.execute_graph(
            self.__graph, self.__seq, self.__data, accuracy, accuracy,
            print_progress=False
        )
        
def generate_files(output_folder, seq_files, description="data"):
    for seq_file in seq_files:
        seq_base_name = os.path.basename(seq_file)
        print(f"Generating {description} for {seq_base_name}")
        
        _, ext = os.path.splitext(seq_file)
        if(ext == ".ipynb"): # mr0_diffusion_prep_STEAM_2D_seq.ipynb
            seq_file = "tests/simulation_test/seq_files/"+os.path.splitext(seq_base_name)[0]+".seq"

        sim_params_comp = SimulationParametersComputer(seq_file)
        timing_results, signal = sim_params_comp.generate_simulation_parameters()
        
        np.savez(
            os.path.join(output_folder, f"{seq_base_name}.npz"),
            timing_results=timing_results,
            signal=signal
        )