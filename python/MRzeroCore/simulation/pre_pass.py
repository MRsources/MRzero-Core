from __future__ import annotations
import json
from concurrent.futures import ThreadPoolExecutor
from warnings import warn
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
    min_state_mag: float = 1e-4,
    tissues: dict | str | None = None,
    threaded: bool = True,
) -> Graph:
    """Like :func:`compute_graph_ext`, but computes some args from :attr:`data`.

    The prepass uses one T1, T2, T2' and D. By default those are the means of
    ``data``, which is the graph of a phantom made entirely of average tissue.
    On a heterogeneous object that graph is missing states other tissues keep,
    and the main pass can only simulate states the graph holds.

    Pass ``tissues`` with more than one entry to run that same prepass once per
    tissue and union the results with the mean-tissue graph. Omit it, or pass a
    single tissue, and only the mean prepass runs.

    Parameters
    ----------
    tissues : dict or str, optional
        Probe set ``{name: {"T1", "T2", "T2dash", "D"}}``, or a NIfTI-phantom
        JSON path (fields ``T2'`` and ``ADC``). There is no built-in table.
    threaded : bool
        Run the per-tissue prepasses concurrently. The prepass releases the GIL.
    """
    params = {} if tissues is None else _tissue_params(tissues)
    if len(params) <= 1:
        return _mean_graph(seq, data, max_state_count, min_state_mag)
    graph, note = _tissue_merged_graph(
        seq, data, max_state_count, min_state_mag, params, threaded
    )
    if note:
        warn(note, RuntimeWarning, stacklevel=2)
    return graph


def _mean_graph(seq, data, max_state_count, min_state_mag) -> Graph:
    """The conventional prepass, on the mean relaxation of ``data``."""
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
    
    if any(rep.pulse.angle > 2*np.pi for rep in seq):
        warn("Some flip angles are > 360°, inhomogeneities produced by extra rotations are ignored by the pre-pass B1 estimation")

    return Graph(_prepass.compute_graph(
        seq,
        T1, T2, T2dash, D,
        max_state_count, min_state_mag,
        nyquist, size, seq.normalized_grads,
        avg_b1_trig
    ))


def _tissue_params(source: dict | str) -> dict:
    """``{name: {T1, T2, T2dash, D}}`` from a dict or a NIfTI-phantom JSON path."""
    if isinstance(source, dict):
        return {
            name: {key: float(p[key]) for key in ("T1", "T2", "T2dash", "D")}
            for name, p in source.items()
        }
    with open(source, encoding="utf-8") as f:
        spec = json.load(f)
    return {
        name: {
            "T1": float(t["T1"]),
            "T2": float(t["T2"]),
            "T2dash": float(t["T2'"]),
            "D": float(t["ADC"]),
        }
        for name, t in spec["tissues"].items()
    }


def _tissue_graph(seq, data, params, max_state_count, min_state_mag) -> Graph:
    """One tissue's prepass, with ``data``'s geometry and B1 LUT."""
    return compute_graph_ext(
        seq,
        params["T1"], params["T2"], params["T2dash"], params["D"],
        max_state_count, min_state_mag,
        data.nyquist.tolist(), data.size.tolist(), data.avg_B1_trig,
    )


def _tissue_merged_graph(
    seq, data, max_state_count, min_state_mag, tissues: dict, threaded: bool,
):
    """Mean prepass plus one per tissue, unioned.

    Returns ``(graph, note)``. ``note`` is set when the union is smaller than
    its largest input, which happens when one tissue's prepass fused states
    another kept apart.
    """
    probes = {
        "mean": {
            "T1": float(torch.mean(data.T1)),
            "T2": float(torch.mean(data.T2)),
            "T2dash": float(torch.mean(data.T2dash)),
            "D": float(torch.mean(data.D)),
        }
    }
    probes.update(tissues)

    def one(item):
        name, params = item
        return name, _tissue_graph(
            seq, data, params, max_state_count, min_state_mag
        )

    if threaded and len(probes) > 1:
        with ThreadPoolExecutor(max_workers=len(probes)) as pool:
            graphs = dict(pool.map(one, probes.items()))
    else:
        graphs = dict(one(item) for item in probes.items())

    sizes = {name: _state_count(g) for name, g in graphs.items()}
    merged = merge_graphs([graphs.pop("mean")] + list(graphs.values()))
    n_merged = _state_count(merged)
    note = None
    if n_merged < max(sizes.values()):
        biggest = max(sizes, key=sizes.get)
        note = (
            f"merged graph at min_state_mag {min_state_mag:.0e} holds {n_merged} "
            f"states, {sizes[biggest] - n_merged} fewer than its {biggest} input "
            f"({sizes[biggest]}): one tissue's prepass merged states that another "
            f"kept apart, so the union inherits the coarser split. The graph is "
            f"still valid, but is not a superset of every input"
        )
    return merged, note


def _state_count(graph) -> int:
    """Stored states, excluding the seeding z0 of repetition 0."""
    return sum(len(rep) for rep in graph[1:])


class _UnionFind:
    def __init__(self):
        self.parent = []

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def merge_graphs(graphs: list) -> Graph:
    """Union several prepass graphs into one that serves every input.

    States are matched structurally. One ancestor edge has exactly one child,
    so the pair (transition, ancestor) names a state. Two states that share
    such a pair are the same state. Identity is resolved with union-find,
    repetition by repetition, rather than by comparing ``prepass_kt_vec``.

    Each merged state keeps the node with the largest ``latent_signal``.
    Ancestry is rewired in place, so the input graphs are consumed.
    """
    n_rep = len(graphs[0])
    if any(len(g) != n_rep for g in graphs):
        raise ValueError("graphs were built from different sequences")

    merged_of = {}
    reps = [[graphs[0][0][0]]]
    for graph in graphs:
        if len(graph[0]) != 1:
            raise ValueError("repetition 0 must hold a single relaxed z0")
        merged_of[id(graph[0][0])] = 0

    for r in range(1, n_rep):
        states, edges_of = [], []
        groups = _UnionFind()
        first_with_edge = {}

        for graph in graphs:
            for state in graph[r]:
                edges = {
                    (kind, merged_of[id(ancestor)]): coeff
                    for kind, ancestor, coeff in state.ancestors
                }
                index = groups.add()
                states.append(state)
                edges_of.append(edges)
                for edge in edges:
                    if edge in first_with_edge:
                        groups.union(first_with_edge[edge], index)
                    else:
                        first_with_edge[edge] = index

        nodes, node_edges, index_of_root = [], [], {}
        for index, state in enumerate(states):
            root = groups.find(index)
            if root not in index_of_root:
                index_of_root[root] = len(nodes)
                nodes.append(state)
                node_edges.append({})
            merged = index_of_root[root]
            if state.dist_type != nodes[merged].dist_type:
                raise AssertionError(
                    f"merged {state.dist_type} with {nodes[merged].dist_type}"
                )
            if state.latent_signal > nodes[merged].latent_signal:
                nodes[merged] = state
            node_edges[merged].update(edges_of[index])
            merged_of[id(state)] = merged

        for node, edges in zip(nodes, node_edges):
            node.ancestors = [
                (kind, reps[r - 1][ancestor], coeff)
                for (kind, ancestor), coeff in edges.items()
            ]
        reps.append(nodes)

    return Graph(reps)


class Graph(list):
    """:class:`Graph` is a wrapper around the list of states returned by the prepass."""
    def __init__(self, graph: list[list[_prepass.PyDistribution]]) -> None:
        super().__init__(graph)
    
    def plot_data(self,
             transversal_mag: bool = True,
             dephasing: str = "tau",
             color: str = "latent signal"):
        """Retrieve the data used for visualization in `plot()`.

        Parameters
        ----------
        transversal_mag : bool
            If true, return + states, otherwise z(0)
        dephasing : str
            Use one of ``['k_x', 'k_y', 'k_z', 'tau']`` dephasing as the
            y-position of a state in the returned data
        color : str
            Use one of ``['abs(mag)', 'phase(mag)', 'latent signal', 'signal',
            'latent signal unormalized', 'emitted signal']``
            as the color of a state in the returned data
        
        Returns
        -------
        nd.array:
            All selected states as a (state_count, 3) array, where the second
            dimension is `[repetition, dephasing, coloring_value]`.
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
            elif color == "latent signal unormalized":
                value = state.latent_signal_unormalized
            elif color == "emitted signal":
                value = state.emitted_signal
            else:
                raise AttributeError(f"Unknown property color={color}")
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
        
        return data


    def plot(self,
             transversal_mag: bool = True,
             dephasing: str = "tau",
             color: str = "latent signal",
             log_color: bool = True):
        """Visualize the graph.

        When using this function, you need to create and show the matplotlib
        figure yourself. This function uses `plot_data` to generate the data,
        which is then shown as scatter plots.

        Parameters
        ----------
        transversal_mag : bool
            If true, show + states, otherwise z(0)
        dephasing : str
            Use one of ``['k_x', 'k_y', 'k_z', 'tau']`` dephasing as the
            y-position of a state in the scatter plot
        color : str
            Use one of ``['abs(mag)', 'phase(mag)', 'latent signal', 'signal',
            'latent signal unormalized', 'emitted signal']``
            as the color of a state in the scatter plot
        log_color : bool
            If true, use the logarithm of the chosen property for coloring
        """
        data = self.plot_data(transversal_mag, dephasing, color)
        if log_color:
            data[:, 2] = np.log10(np.abs(data[:, 2]) + 1e-7)

        plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=20)
        plt.xlabel("Repetition")
        plt.ylabel(f"${dephasing}$ - Dephasing")
        if log_color:
            plt.colorbar(label="log. " + color)
        else:
            plt.colorbar(label=color)
