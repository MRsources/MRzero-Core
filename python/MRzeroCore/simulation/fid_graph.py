from __future__ import annotations
from typing import Optional

from ..sequence import Sequence, PulseUsage
from .pre_pass import Graph


class FidDistribution:
    """Python stand-in for the distributions returned by the prepass.

    Provides all attributes read by :func:`execute_graph` and
    :meth:`Graph.plot`. There is no prepass simulation, so the prepass metrics
    are placeholders: every state is marked as fully relevant for the signal.
    """
    def __init__(self, dist_type: str,
                 ancestor: Optional[tuple[str, FidDistribution]] = None
                 ) -> None:
        # Same layout as the prepass edges: (relation, ancestor, rot_factor).
        # The main pass recomputes the rotation, so the factor is unused.
        self.ancestors = [(*ancestor, 1.0)] if ancestor else []
        self.dist_type = dist_type
        # Set by the main pass
        self.mag = None
        self.kt_vec = None
        # Never skip a state because of the signal thresholds
        self.emitted_signal = 1.0 if dist_type == '+' else 0.0
        self.latent_signal = 1.0
        self.latent_signal_unormalized = 1.0
        # Fake prepass data, only used for plotting
        self.prepass_mag = 1.0
        self.signal = self.emitted_signal
        self.prepass_kt_vec = [0.0, 0.0, 0.0, 0.0]

    def __repr__(self) -> str:
        return (
            f"FidDist(type: {self.dist_type}, "
            f"#ancestors: {len(self.ancestors)})"
        )


def fid_graph(seq: Sequence) -> Graph:
    """Create a graph that only simulates z0 and one transverse state.

    Alternative to :func:`compute_graph` without running the prepass. The
    returned graph can be executed by :func:`execute_graph`. Echoes from
    stimulated or refocused states (except for the single tracked one) are
    not simulated, so this is only exact for perfectly spoiled sequences.

    The ``+`` state is created by every excitation (or undefined) pulse,
    which discards previous transverse magnetisation. Refocusing pulses
    conjugate it, all other pulses only scale it.

    Parameters
    ----------
    seq : Sequence
        The sequence that produces the returned graph

    Returns
    -------
    Graph
        Graph with one ``z0`` and at most one ``+`` state per repetition
    """
    # The first repetition contains only one element: A fully relaxed z0
    graph = [[FidDistribution('z0')]]

    for rep in seq:
        pulse = rep.pulse
        prev = graph[-1]
        dists = [FidDistribution('z0', ('zz', prev[0]))]

        if pulse.usage in (PulseUsage.EXCIT, PulseUsage.UNDEF):
            dists.append(FidDistribution('+', ('z+', prev[0])))
        elif len(prev) == 2:
            # Can't refocus if nothing is excited
            if pulse.usage == PulseUsage.REFOC:
                dists.append(FidDistribution('+', ('-+', prev[1])))
            else:
                dists.append(FidDistribution('+', ('++', prev[1])))

        # +z and -z are never used because there is no pulse.usage that flips
        # excited magnetisation back to z

        graph.append(dists)

    return Graph(graph)
