import os
from timeit import timeit
import numpy as np
import sys
import MRzeroCore as mr0
import config

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import suppress_output


class SimulationParametersComputer:
    def __init__(self, seq_file):
        self.seq = mr0.Sequence.import_file(seq_file)
        self.phantom = mr0.util.load_phantom()
        self.data = self.phantom.build()
        with suppress_output():
            self.graph = mr0.compute_graph(self.seq, self.data, 2000, 1e-5)

    def generate_simulation_parameters(self):
        timing_results = np.zeros(len(config.ACC_ARRAY))
        signal_list = np.empty(len(config.ACC_ARRAY), dtype=object)
        for i, acc in enumerate(config.ACC_ARRAY):
            signal, avrg_time = self.execute_signal(self.seq, self.data, self.graph, acc)
            timing_results[i] = avrg_time
            signal_list[i] = signal.numpy()
        return timing_results, signal_list
    
    def execute_signal(self, seq, data, graph, accuracy):
        signal =  mr0.execute_graph(graph, seq, data, accuracy, accuracy, print_progress=False)
        avrg_time = timeit(lambda:  mr0.execute_graph(graph, seq, data, accuracy, accuracy, print_progress=False), number=10)
        return signal, avrg_time

def generate_files(output_folder, seq_files, description="data"):
    for seq_file in seq_files:
        seq_base_name = os.path.basename(seq_file)
        print(f"Generating {description} for {seq_base_name}")

        sim_params_comp = SimulationParametersComputer(seq_file)
        timing_results, signal = sim_params_comp.generate_simulation_parameters()
        
        np.savez(
            os.path.join(output_folder, f"{seq_base_name}.npz"),
            timing_results=timing_results,
            signal=signal
        )