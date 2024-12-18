import MRzeroCore as mr0
import matplotlib.pyplot as plt
from numpy import pi
import torch

def build_seq() -> mr0.Sequence:
    seq = mr0.Sequence()

    for i in range(64):
        rep = seq.new_rep(2 + 64 + 1)
        rep.pulse.usage = mr0.PulseUsage.EXCIT
        rep.pulse.angle = 7 * pi/180
        rep.pulse.phase = 0.5 * 137.50776405 * (i**2+i+2) * pi / 180

        rep.event_time[0] = 2e-3  # Pulse
        rep.event_time[1] = 2e-3  # Rewinder
        rep.event_time[2:-1] = 0.08e-3  # Readout
        rep.event_time[-1] = 2e-3  # Spoiler

        rep.gradm[1, 0] = -33
        rep.gradm[2:-1, 0] = 1
        rep.gradm[-1, 0] = 96 - 31

        # Linear reordered phase encoding
        rep.gradm[1, 1] = i - 32
        rep.gradm[-1, 1] = -rep.gradm[1, 1]

        rep.adc_usage[2:-1] = 1
        rep.adc_phase[2:-1] = pi - rep.pulse.phase

    return seq

# Build the default FLASH and show the kspace
seq = build_seq()
seq.plot_kspace_trajectory()

# Until now, the sequence uses normalized grads: The simulation will adapt them
# to the phantom size. If we want to hardcode a fixed FOV instead, we can do so:
seq.normalized_grads = False
for rep in seq:
    rep.gradm[:] /= 200e-3  # 200 mm FOV

# Load a BrainWeb phantom for simulation

# https://github.com/MRsources/MRzero-Core/raw/main/documentation/examples/subject05.npz

data_path =r'data\subject05.npz'

phantom = mr0.VoxelGridPhantom.brainweb(data_path)
phantom = phantom.interpolate(64, 64, 32).slices([16])
phantom.plot()
phantom.fov = torch.tensor([0.15, 0.15, 1])
data = phantom.build()
graph = mr0.compute_graph(seq, data)
signal = mr0.execute_graph(graph, seq, data)
reco = mr0.reco_adjoint(signal, seq.get_kspace())

graph_mi = mr0.compute_graph(seq, data,start_mag=torch.ones_like(data.PD).to(data.PD.device))
signal_mi= mr0.execute_graph(graph_mi, seq, data,start_mag=torch.ones_like(data.PD).to(data.PD.device))
#graph_mi = mr0.compute_graph(seq, data,start_mag=data.PD)
#signal_mi= mr0.execute_graph(graph_mi, seq, data,start_mag=data.PD)
reco_mi = mr0.reco_adjoint(signal_mi, seq.get_kspace())

#graph_mi = mr0.compute_graph(seq, data,start_mag=torch.ones_like(data.PD).to(data.PD.device))
#signal_mi = mr0.execute_graph(graph, seq, data,start_mag=torch.ones_like(data.PD).to(data.PD.device))

# Plot the result

plt.figure()
plt.subplot(131)
plt.title("Reconstruction")
plt.imshow(reco.abs().cpu()[:, :, 0].T, origin='lower', vmin=0)
plt.subplot(132)
plt.title("Reconstruction MI")
plt.imshow(reco_mi.abs().cpu()[:, :, 0].T, origin='lower', vmin=0)
plt.subplot(133)
plt.title("Diff")
plt.imshow(abs(reco.abs().cpu()[:, :, 0].T-reco_mi.abs().cpu()[:, :, 0].T), origin='lower', vmin=0)
plt.show()

print('')