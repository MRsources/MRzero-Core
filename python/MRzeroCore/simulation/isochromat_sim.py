"""
This Bloch simulation is meant as a ground truth to compare PDG against.

Note that this still uses box shaped voxels, which is generally discouraged.

.. code-block:: python

    data = SimData.load(...)
    seq = Sequence()  # Some definition as used by PDG

    spin_count = 123  # R2 - distributed spins, doesn't need to be a square nr.
    signal = simulate(seq, data, spin_count)

    # Signal has same format as returned by execute_graph(...),
    # no further code changes are needed
"""

from __future__ import annotations
import torch
from numpy import pi

from ..sequence import Sequence, PulseUsage
from ..phantom.sim_data import SimData


def isochromat_sim(seq: Sequence, data: SimData, spin_count: int,
                   perfect_spoiling=False,
                   print_progress: bool = True,
                   ) -> torch.Tensor:
    """
    Simulate ``seq`` on ``data`` with ``spin_count`` spins per voxel.

    Parameters
    ----------
    seq: Sequence
        The sequence that will be simulated
    data: SimData
        Simulation data that defines everything else
    spin_count: int
        Number of spins used for simulation
    perfect_spoiling: bool
        If ``True``, the transversal magnetization is set to zero on excitation

    Returns
    -------
    torch.Tensor
        Complex tensor with shape (sample_count, coil_count)
    """
    # RawSimData doesn't store the voxel shape. We could extract it by FFT'ing
    # the dephasing function, if that feature is desired. Until then, we just
    # use the nyquist frequencies, with a special case for ∞ (custom phantoms).
    voxel_size = 0.5 / torch.tensor(data.nyquist, device=data.device)
    if not torch.isfinite(voxel_size).all():
        # Fallback voxel size
        voxel_size = torch.tensor([0.1, 0.1, 0.1], device=data.device)

    # 3 dimensional R2 sequence for intravoxel spin distribution
    g = 1.22074408460575947536  # 3D
    # g = 1.32471795724474602596  # 2D
    a = 1.0 / torch.tensor([g**1, g**2, g**3], device=data.device)
    indices = torch.arange(spin_count, device=data.device)
    spin_pos = torch.stack([
        (0.5 + a[0] * indices) % 1,
        (0.5 + a[1] * indices) % 1,
        (0.5 + a[2] * indices) % 1
    ])
    # spin_pos = torch.rand_like(spin_pos)  # Use white noise
    spin_pos = 2 * pi * (spin_pos - 0.5) * voxel_size.unsqueeze(1)

    # Omega is a cauchy-distributed tensor of spin offset freqencies (for T2')
    off_res = torch.linspace(-0.5, 0.5, spin_count, device=data.device)
    omega = torch.tan(pi * 0.999 * off_res)  # Cut off high frequencies
    omega = omega[torch.randperm(spin_count)]

    # Combine coil sensitivities and proton density to a voxels x coils tensor
    coil_sensitivity = (
        data.coil_sens.t().to(torch.cfloat)
        * data.PD.unsqueeze(1) / spin_count
    )
    coil_count = data.coil_sens.shape[0]

    # Start off with relaxed magnetisation, stored as (voxels x spins)
    spins = torch.zeros((data.PD.numel(), spin_count, 3),
                        device=data.device)
    spins[:, :, 2] = 1

    # Simulation:
    signal = []

    for r, rep in enumerate(seq):
        if print_progress:
            print(f"\r {r+1} / {len(seq)}", end="")

        if perfect_spoiling and rep.pulse.usage == PulseUsage.EXCIT:
            spins[:, :, :2] = 0

        spins = flip(spins, rep.pulse.angle, rep.pulse.phase, data.B1)
        rep_sig = torch.zeros((rep.event_count, coil_count),
                              dtype=torch.cfloat, device=data.device)
        signal.append(rep_sig)

        # When applying relaxation, dephasing etc. event after event,
        # we accumulate a lot of error. Instead, calculate it relative to the
        # state immedeately after the pulse for every event
        spins_start = spins
        time = rep.event_time.cumsum(0)
        gradm = rep.gradm.cumsum(0)

        for e in range(rep.event_count):
            spins = spins_start.clone()
            spins = relax(spins, data.T1, data.T2, time[e])
            spins = dephase(spins, omega, data.T2dash, time[e])
            spins = intravoxel_precess(spins, gradm[e], spin_pos)

            spins = B0_precess(spins, data.B0, time[e])
            spins = grad_precess(spins, gradm[e], data.voxel_pos)

            if rep.adc_usage[e] > 0:
                adc_rot = torch.exp(1j * rep.adc_phase[e])
                rep_sig[e, :] = measure(spins, coil_sensitivity) * adc_rot

    if print_progress:
        print(" - done")
    # Only return measured samples
    return torch.cat([
        sig[rep.adc_usage > 0, :] for sig, rep in zip(signal, seq)
    ])


def measure(spins: torch.Tensor, coil_sensitivity: torch.Tensor
            ) -> torch.Tensor:
    """Calculate the measured signal per coil.

    The returned tensor is 1D with ``coil_count`` elements.
    """
    voxel_mag = spins[:, :, 0].sum(1) + 1j*spins[:, :, 1].sum(1)
    # (voxels), (voxels x coils)
    return voxel_mag @ coil_sensitivity


def relax(spins: torch.Tensor, T1: torch.Tensor, T2: torch.Tensor, dt: float
          ) -> torch.Tensor:
    """Relax xy magnetisation with T1 towards 0 and z with T2 towards 1."""
    relaxed = torch.empty_like(spins)
    r1 = torch.exp(-dt / T1).view(-1, 1)
    relaxed[:, :, 2] = spins[:, :, 2] * r1 + (1 - r1)
    relaxed[:, :, :2] = spins[:, :, :2] * torch.exp(-dt / T2).view(-1, 1, 1)

    return relaxed


def dephase(spins: torch.Tensor, omega: torch.Tensor,
            T2dash: torch.Tensor, dt: float) -> torch.Tensor:
    """T2' - dephase spins, the per-voxel amount is given by T2dash and dt."""
    # shape: voxels x spins
    angle = ((1 / T2dash).unsqueeze(1) @ omega.unsqueeze(0)) * dt
    rot_mat = torch.zeros((*angle.shape, 3, 3), device=spins.device)
    rot_mat[:, :, 0, 0] = torch.cos(angle)
    rot_mat[:, :, 0, 1] = -torch.sin(angle)
    rot_mat[:, :, 1, 0] = torch.sin(angle)
    rot_mat[:, :, 1, 1] = torch.cos(angle)
    rot_mat[:, :, 2, 2] = 1
    # (voxels x spins x 3 x 3), (voxels x spins x 3)
    return torch.einsum("vsij, vsj -> vsi", rot_mat, spins)


def flip(spins: torch.Tensor, angle: torch.Tensor, phase: torch.Tensor,
         B1: torch.Tensor) -> torch.Tensor:
    """Rotate the magnetisation to simulate a RF-pulse."""
    a = angle * B1
    p = torch.as_tensor(phase)
    # Rz(phase) * Rx(angle) * Rz(-phase):
    rot_mat = torch.zeros((B1.numel(), 3, 3), device=spins.device)
    rot_mat[:, 0, 0] = torch.sin(p)**2*torch.cos(a) + torch.cos(p)**2
    rot_mat[:, 0, 1] = (1 - torch.cos(a))*torch.sin(p)*torch.cos(p)
    rot_mat[:, 0, 2] = torch.sin(a)*torch.sin(p)
    rot_mat[:, 1, 0] = (1 - torch.cos(a))*torch.sin(p)*torch.cos(p)
    rot_mat[:, 1, 1] = torch.sin(p)**2 + torch.cos(a)*torch.cos(p)**2
    rot_mat[:, 1, 2] = -torch.sin(a)*torch.cos(p)
    rot_mat[:, 2, 0] = -torch.sin(a)*torch.sin(p)
    rot_mat[:, 2, 1] = torch.sin(a)*torch.cos(p)
    rot_mat[:, 2, 2] = torch.cos(a)
    # (voxels x 3 x 3), (voxels x spins x 3)
    return torch.einsum("vij, vsj -> vsi", rot_mat, spins)


def grad_precess(spins: torch.Tensor, gradm: torch.Tensor,
                 voxel_pos: torch.Tensor) -> torch.Tensor:
    """Rotate individual voxels as given by their position and ```gradm``."""
    angle = 2 * pi * voxel_pos @ gradm  # shape: voxels
    rot_mat = torch.zeros((angle.numel(), 3, 3), device=spins.device)
    rot_mat[:, 0, 0] = torch.cos(angle)
    rot_mat[:, 0, 1] = -torch.sin(angle)
    rot_mat[:, 1, 0] = torch.sin(angle)
    rot_mat[:, 1, 1] = torch.cos(angle)
    rot_mat[:, 2, 2] = 1
    # (voxels x 3 x 3), (voxels x spins x 3)
    return torch.einsum("vij, vsj -> vsi", rot_mat, spins)


def B0_precess(spins: torch.Tensor, B0: torch.Tensor,
               dt: float) -> torch.Tensor:
    """Rotate voxels as given by ``B0`` and the elapsed time ``dt``."""
    angle = 2 * pi * B0 * dt  # shape: voxels
    rot_mat = torch.zeros((angle.numel(), 3, 3), device=spins.device)
    rot_mat[:, 0, 0] = torch.cos(angle)
    rot_mat[:, 0, 1] = -torch.sin(angle)
    rot_mat[:, 1, 0] = torch.sin(angle)
    rot_mat[:, 1, 1] = torch.cos(angle)
    rot_mat[:, 2, 2] = 1
    # (voxels x 3 x 3), (voxels x spins x 3)
    return torch.einsum("vij, vsj -> vsi", rot_mat, spins)


def intravoxel_precess(spins: torch.Tensor, gradm: torch.Tensor,
                       spin_pos: torch.Tensor) -> torch.Tensor:
    """Rotate spins inside of each voxel to simulate the dephasing.

    ``grad_precess`` and ``intravoxel_precess`` are both needed to correctly
    simulate the effect gradients have on the magnetisation.
    """
    angle = spin_pos.T @ gradm  # shape: spins
    rot_mat = torch.zeros((angle.numel(), 3, 3), device=spins.device)
    rot_mat[:, 0, 0] = torch.cos(angle)
    rot_mat[:, 0, 1] = -torch.sin(angle)
    rot_mat[:, 1, 0] = torch.sin(angle)
    rot_mat[:, 1, 1] = torch.cos(angle)
    rot_mat[:, 2, 2] = 1
    # (spins x 3 x 3), (voxels x spins x 3)
    return torch.einsum("sij, vsj -> vsi", rot_mat, spins)
