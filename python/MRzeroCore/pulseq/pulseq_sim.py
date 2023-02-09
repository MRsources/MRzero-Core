# NOTE: This is just temporary.
# A better approach would be to provide a simple function that can
# convert .seq to MR0 sequences and then do simulation yourself.

# %% Imports
from __future__ import annotations
import torch
import numpy as np
from .pulseq_loader.__init__ import intermediate, PulseqFile, plot_file, Adc, Spoiler
from ..sequence import Sequence, PulseUsage
import matplotlib.pyplot as plt
from ..simulation.pre_pass import compute_graph
from ..simulation.main_pass import execute_graph
from ..reconstruction import reco_adjoint


def sim_external(object_sz=32, reco_sz=0, plot_seq_k=(0, 0), obj=0,
                 dB0=0, M_threshold=1e-3, seqfile='./out/external.seq'):
    #  Load a pulseq file

    # NOTE
    # .seq files exported by the official exporter put the adc sample at the
    # beginning of the event but MRzero has them at the end - we need to shift
    # adc by 1 in our exporters
    #
    # We interpret gradients & pulses with time shapes as being constant until the
    # next time point, but linear interp. might be assumed - the spec says nothing
    #
    # Adc phases in the pulseq files seem to be defined such that e.g. in a GRE,
    # adc phase should equal pulse phase. In MRzeros coordinate space, adc phase
    # should be 90° - pulse phase, which means we should export 90° - adc phase,
    # but currently use adc phase - 45°

    # Import a pulseq file (supported: 1.2.0, 1.2.1, 1.3.0, 1.3.1, 1.4.0)
    pulseq = PulseqFile(seqfile)
    # Can also save it again as 1.4.0 file (for converting older files)
    # pulseq.save("pulseq_loader/tests/out.seq")
    # Plot the full sequence stored in the file
    if plot_seq_k[0]:
        plot_file(pulseq, figsize=(10, 6))
    # Convert seqence into a intermediate form only containing what's simulated
    tmp_seq = intermediate(pulseq)

    # Convert into a MRzero sequence
    seq = Sequence()
    rep = None
    for tmp_rep in tmp_seq:
        rep = seq.new_rep(tmp_rep[0])
        rep.pulse.angle = torch.tensor(tmp_rep[1].angle, dtype=torch.float)
        rep.pulse.phase = torch.tensor(tmp_rep[1].phase, dtype=torch.float)
        is_refoc = abs(tmp_rep[1].angle) > 1.6  # ~91°
        rep.pulse.usage = PulseUsage.REFOC if is_refoc else PulseUsage.EXCIT

        offset = 0
        for block in tmp_rep[2]:
            if isinstance(block, Spoiler):
                rep.event_time[offset] = block.duration
                rep.gradm[offset, :] = torch.tensor(block.gradm)
                offset += 1
            else:
                assert isinstance(block, Adc)
                num = len(block.event_time)
                rep.event_time[offset:offset+num] = torch.tensor(block.event_time)
                rep.gradm[offset:offset+num, :] = torch.tensor(block.gradm)
                rep.adc_phase[offset:offset+num] = np.pi/2 - block.phase
                rep.adc_usage[offset:offset+num] = 1
                offset += num
        assert offset == tmp_rep[0]

    # Trajectory calculation and plotting

    if plot_seq_k[1]:
        seq.plot_kspace_trajectory(plot_timeline=False)

    # Simulate imported sequence
    reco_size = (reco_sz, reco_sz, 1)

    data = obj

    graph = compute_graph(seq, data, 1000, M_threshold)

    signal = execute_graph(graph, seq, data)
    kspace = seq.get_kspace()

    if reco_sz > 0:
        reco = reco_adjoint(
            signal, kspace,
            resolution=reco_size, FOV=(1, 1, 1)
        )

        plt.figure(figsize=(7, 5))
        plt.subplot(211)
        plt.imshow(reco.abs(), vmin=0)
        plt.colorbar()
        plt.subplot(212)
        plt.imshow(reco.angle())
        plt.show()

    return signal, kspace
