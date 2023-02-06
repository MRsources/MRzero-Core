from __future__ import annotations
from MRzeroCore import _prepass
from MRzeroCore._prepass import PyDistribution as PrePassState
from ..sequence import Sequence
from ..phantom.sim_data import SimData
import torch
from numpy import pi


# TODO: Add documentation and functions to analyze the graph


def compute_graph(
    seq: Sequence,
    data: SimData,
    max_state_count: int = 200,
    min_state_mag: float = 1e-4
):
    """Like pre_pass.compute_graph, but computes args from `` data``."""
    return compute_graph_ext(
        seq,
        float(torch.mean(data.T1)),
        float(torch.mean(data.T2)),
        float(torch.mean(data.T2dash)),
        float(torch.mean(data.D)),
        max_state_count,
        min_state_mag,
        data.nyquist,
        data.fov.tolist(),
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
    fov: tuple[float, float, float] = (1.0, 1.0, 1.0),
    avg_b1_trig: torch.Tensor | None = None,
) -> list[list[PrePassState]]:
    if min_state_mag < 0:
        min_state_mag = 0

    if avg_b1_trig is None:
        angle = torch.linspace(0, 2*pi, 361)
        avg_b1_trig = torch.stack([
            torch.sin(angle),
            torch.cos(angle),
            torch.sin(angle/2)**2
        ], dim=1).type(torch.float32)

    return _prepass.compute_graph(
        seq,
        T1, T2, T2dash, D,
        max_state_count, min_state_mag,
        nyquist, fov,
        avg_b1_trig
    )
