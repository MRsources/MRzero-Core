# %% Imports
# TODO: fix them

from __future__ import annotations
import torch
import numpy as np
from pulseq_loader import intermediate, PulseqFile, plot_file, Adc, Spoiler
from new_core.sequence import Sequence, PulseUsage
from new_core.util import plot_kspace_trajectory
from new_core import util
import matplotlib.pyplot as plt
from new_core.sim_data import VoxelGridPhantom, SimData
from new_core.pdg_main_pass import execute_graph
from new_core.reconstruction import reconstruct, nufft_reco_2D
import pre_pass
from time import time
from imageio import mimsave

util.use_gpu = True
SEQUENCE = "PDG paper/SR_30ms_1dummy.seq"
# SEQUENCE = r"\\141.67.249.47\MRTransfer\pulseq_zero\sequences\seq221216\IR_48_48_6_new_1_final_mask01.seq"


# TODO: wrap this all into a function that takes the file path as argument,
# as well as simulation settings, and returns signals or images
# Maybe also just make this an import function and provide the script
# to run it just as an example?


# %% Load a pulseq file

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

# NOTE: The trufi.seq file does not seem to refocus to exactly 0
# -> is this a bug in the sequence design or in the interpreter?

# Import a pulseq file (supported: 1.2.0, 1.2.1, 1.3.0, 1.3.1, 1.4.0)
pulseq = PulseqFile(SEQUENCE)
# Can also save it again as 1.4.0 file (for converting older files)
# pulseq.save("pulseq_loader/tests/out.seq")
# Plot the full sequence stored in the file
plot_file(pulseq, figsize=(10, 6))
# Convert seqence into a intermediate form only containing what's simulated
tmp_seq = intermediate(pulseq)

# Convert into a MRzero sequence
seq = Sequence()
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

plot_kspace_trajectory(seq, plot_timeline=False)


# Convert sequence to gpu if needed
for rep in seq:
    rep.pulse.angle = util.set_device(rep.pulse.angle)
    rep.pulse.phase = util.set_device(rep.pulse.phase)
    rep.adc_phase = util.set_device(rep.adc_phase)
    rep.adc_usage = util.set_device(rep.adc_usage)
    rep.gradm = util.set_device(rep.gradm)
    rep.event_time = util.set_device(rep.event_time)



# %% Simulation

# phantom = VoxelGridPhantom.brainweb("brainweb/output/subject05.npz").interpolate(48, 48, 64)
phantom = VoxelGridPhantom.load_mat("../../data/numerical_brain_cropped.mat").interpolate(128, 128, 1)

# Change slices or comment out line as needed
# phantom = phantom.slices([20, 25, 30, 35, 40, 45])

data = phantom.build()
from time import time
start = time()
graph = util.simple_compute_graph(seq, data)
signal = execute_graph(graph, seq, data)
print(time() - start)

# from new_core.bloch_sim import simulate
# signal = simulate(seq, data, 10000)


# %% Reconstruction


# If the kspace and adc_usage was written into the .seq file, we use it
try:
    kspace, adc_usage = util.extract_data_from_seq_file(SEQUENCE)
    kspace = util.set_device(kspace)
    adc_usage = util.set_device(adc_usage)

except ValueError:
    # We don't know the targeted k-space trajectory and create it based on the
    # pulse angles: < 90° is assumed to be for excitation, > 90° for refocusing
    print("RECO: NUFFT Reco with k-space based on guessed pulse usages")
    kspace = seq.get_kspace()

    # Reconstruct the image - replace if needed
    reco = nufft_reco_2D(signal, kspace / (2*np.pi), (256, 256))
    plt.figure(figsize=(7, 5), dpi=120)
    plt.imshow(reco.abs()[:, :].T.cpu(), origin="lower", vmin=0)
    plt.colorbar()
    plt.show()

else:
    print("RECO: .seq file contained kspace and adc_usage data, using it...")
    contrasts = sorted(set(adc_usage.tolist()))
    print(f"Sequence contains following contrasts: {contrasts}")
    for contrast in contrasts:
        mask = adc_usage == contrast

        # Reconstruct contrast no. 'contrast' - replace if needed
        # reco = nufft_reco_2D(signal[mask], kspace[mask] / (2*np.pi), (256, 256))
        reco = reconstruct(signal[mask], kspace[mask], (48, 48, 6), (1, 1, 1))
        for i in range(6):
            plt.figure(figsize=(7, 5), dpi=80)
            plt.title(f"Contrast {contrast}")
            plt.imshow(reco.abs()[:, :, i].T.cpu(), origin="lower", vmin=0)
            plt.colorbar()
            plt.show()
