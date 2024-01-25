from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt
from ..sequence import Sequence
from ..phantom.sim_data import SimData
from MRzeroCore import _prepass


def compute_graph(
    seq: Sequence,
    data: SimData,
    max_state_count: int = 200,
    min_state_mag: float = 1e-4
) -> Graph:
    """Like :func:`pre_pass.compute_graph_ext`, but computes some args from :attr:`data`."""
    return compute_graph_ext(
        seq,
        float(torch.mean(data.T1)),
        float(torch.mean(data.T2)),
        float(torch.mean(data.T2dash)),
        float(torch.mean(data.D)),
        max_state_count,
        min_state_mag,
        data.nyquist.tolist(),
        data.size.tolist(),
        data.avg_B1_trig
    )


def compute_graph_ext(
    seq: Sequence,
    T1: float,
    T2: float,
    T2dash: float,
    D: float,
    max_state_count: int = 200,
    min_state_mag: float = 1e-4,
    nyquist: tuple[float, float, float] = (float('inf'), float('inf'), float('inf')),
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    avg_b1_trig: torch.Tensor | None = None,
) -> Graph:
    """Compute the PDG from the sequence and phantom data provided.

    Parameters
    ----------
    seq : Sequence
        The sequence that produces the returned PDG
    T1 : float
        Simulated T1 relaxation time [s]
    T2 : float
        Simulated T2 relaxation time [s]
    T2' : float
        Simulated T2' relaxation time [s]
    D : float
        Simulated diffusion coefficient [$10^{-3} mm^2 / s$]
    max_state_count : int
        Maximum state count. If more states are produced, the weakest are omitted.
    min_state_mag : float
        Minimum magnetization of a state to be simulated.
    nyquist : (float, float, float)
        Nyquist frequency of simulated data. Signal is cut off for higher frequencies.
    size : (float, float, float)
        Size of the simulated phantom. Used for scaling grads for normalized seqs.
    avg_b1_trig : torch.Tensor | None
        Tensor containing the B1-averaged trigonometry used in the rotation matrix.
        Default values are used if `None` is passed.
    """
    if min_state_mag < 0:
        min_state_mag = 0

    if avg_b1_trig is None:
        angle = torch.linspace(0, 2*np.pi, 361)
        avg_b1_trig = torch.stack([
            torch.sin(angle),
            torch.cos(angle),
            torch.sin(angle/2)**2
        ], dim=1).type(torch.float32)

    return Graph(_prepass.compute_graph(
        seq,
        T1, T2, T2dash, D,
        max_state_count, min_state_mag,
        nyquist, size, seq.normalized_grads,
        avg_b1_trig
    ))


class Graph(list):
    """:class:`Graph` is a wrapper around the list of states returned by the prepass."""
    def __init__(self, graph: list[list[_prepass.PyDistribution]]) -> None:
        super().__init__(graph)

    def plot(self,
             transversal_mag: bool = True,
             dephasing: str = "tau",
             color: str = "latent signal",
             log_color: bool = True):
        """Visualize the graph.

        Parameters
        ----------
        transversal_mag : bool
            If true, show only + states, otherwise z(0)
        dephasing : str
            Use one of ``['k_x', 'k_y', 'k_z', 'tau']`` dephasing as the
            y-position of a state in the scatter plot
        color : str
            Use one of ``['abs(mag)', 'phase(mag)', 'latent signal', 'signal',
            'emitted signal']`` as color of a state in the scatter plot
        log_color : bool
            If true, use the logarithm of the chosen property for coloring
        """
        data = []
        kt_idx = {"k_x": 0, "k_y": 1, "k_z": 2, "tau": 3}[dephasing]

        def extract(state: _prepass.PyDistribution):
            if color == "abs(mag)":
                value = np.abs(state.prepass_mag)
            elif color == "phase(mag)":
                value = np.angle(state.prepass_mag)
            elif color == "latent signal":
                value = state.latent_signal
            elif color == "signal":
                value = state.signal
            elif color == "emitted signal":
                value = state.emitted_signal
            if log_color:
                value = np.log10(np.abs(value) + 1e-7)
            return value

        for r, rep in enumerate(self):
            for state in rep:
                if transversal_mag == (state.dist_type == "+"):
                    data.append((
                        r,
                        state.prepass_kt_vec[kt_idx],
                        extract(state),
                    ))

        data.sort(key=lambda d: d[2])
        data = np.asarray(data)

        plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=20)
        plt.xlabel("Repetition")
        plt.ylabel(f"${dephasing}$ - Dephasing")
        if log_color:
            plt.colorbar(label="log. " + color)
        else:
            plt.colorbar(label=color)
