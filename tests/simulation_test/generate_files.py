import os
from timeit import Timer
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
            signal, avg_time = self.__get_seq_parameters(acc)
            timing_results[i] = avg_time
            signal_list[i] = signal.numpy()
        return timing_results, signal_list
    
    def __get_seq_parameters(self, accuracy):
        signal = self.__execute_graph(accuracy)
        timer = Timer(lambda: self.__execute_graph(accuracy))
        number_of_runs, total_time = self.__autorange_custom(timer)
        return signal, total_time / number_of_runs

    def __autorange_custom(self, timer, min_time=1.0):
        """
        Run timeit() repeatedly, increasing number of loops until total time >= min_time seconds.
        Returns (number, total_time).
        """
        # same loop progression as timeit.autorange()
        total_time = 0
        number_of_runs = 1
        while total_time < min_time:
            time = timer.timeit(1)
            total_time += time
            number_of_runs += 1
        return number_of_runs, total_time

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