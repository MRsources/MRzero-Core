from __future__ import annotations
import torch

from ..sequence import Sequence
from ..phantom.sim_data import SimData
from .pre_pass import Graph
import numpy as np


# NOTE: return encoding and magnetization is currently missing. If we want to
# move it into core, it probably should be done by providing a callback that
# has can extract all information it wants during simulation.


def rigid_motion(voxel_pos, motion_func):
    """Shape of returned tensor: events x voxels x 3"""
    def voxel_motion(time):
        # rot: events x 3 x 3, offset: events x 3
        rot, offset = motion_func(time)
        rot = rot.to(device=voxel_pos.device)
        offset = offset.to(device=voxel_pos.device)
        return torch.einsum("vi, eij -> evj", voxel_pos, rot) + offset[:, None, :]
    
    return voxel_motion


def execute_graph(graph: Graph,
                  seq: Sequence,
                  data: SimData,
                  min_emitted_signal: float = 1e-2,
                  min_latent_signal: float = 1e-2,
                  print_progress: bool = True,
                  return_mag_p: int | bool | None = None,
                  return_mag_z: int | bool | None = None,
                  ) -> torch.Tensor | list:
    """Calculate the signal of the sequence by executing the phase graph.

    Parameters
    ----------
    graph : Graph
        Phase Distribution Graph that will be executed.
    seq : Sequence
        Sequence that will be simulated and was used to create :attr:`graph`.
    data : SimData
        Physical properties of phantom and scanner.
    min_emitted_signal : float
        Minimum "emitted_signal" metric of a state for it to be measured.
    min_latent_signal : float
        Minimum "latent_signal" metric of a state for it to be simulated.
        Should be <= than min_emitted_signal.
    print_progress : bool
        If true, the current repetition is printed while simulating.
    return_mag_z : int or bool, optional
        If set, returns the longitudinal magnetisation of either the given
        repetition (int) or all repetitions (``True``).


    Returns
    -------
    signal : torch.Tensor
        The simulated signal of the sequence.
    mag_z : torch.Tensor | list[torch.Tensor]
        The longitudinal magnetisation of the specified or all repetition(s).

    """
    # This is a function that maps time to voxel positions.
    # If it is defined, motion is simulated, otherwise the static data.voxel_pos is used
    t0 = 0
    voxel_pos_func = data.voxel_motion
    if voxel_pos_func is None and data.phantom_motion is not None:
        voxel_pos_func = rigid_motion(data.voxel_pos, data.phantom_motion)


    if seq.normalized_grads:
        grad_scale = 1 / data.size
    else:
        grad_scale = torch.ones_like(data.size)
    signal: list[torch.Tensor] = []

    # Proton density can be baked into coil sensitivity. shape: voxels x coils
    coil_sensitivity = (
        data.coil_sens.t().to(torch.cfloat)
        * torch.abs(data.PD).unsqueeze(1)
    )
    coil_count = int(coil_sensitivity.shape[1])
    voxel_count = data.PD.numel()

    # The first repetition contains only one element: A fully relaxed z0
    graph[0][0].mag = torch.ones(
        voxel_count, dtype=torch.cfloat, device=data.device
    )
    # Calculate kt_vec ourselves for autograd
    graph[0][0].kt_vec = torch.zeros(4, device=data.device)

    mag_p = []
    mag_z = []
    for i, (dists, rep) in enumerate(zip(graph[1:], seq)):
        if print_progress:
            print(f"\rCalculating repetition {i+1} / {len(seq)}", end='')

        angle = torch.as_tensor(rep.pulse.angle)
        phase = torch.as_tensor(rep.pulse.phase)
        shim_array = torch.as_tensor(rep.pulse.shim_array)

        # 1Tx or pTx?
        if shim_array.shape[0] == 1:
            B1 = data.B1.sum(0)
        else:
            assert shim_array.shape[0] == data.B1.shape[0]
            shim = shim_array[:, 0] * torch.exp(-1j * shim_array[:, 1])
            B1 = (data.B1 * shim[:, None]).sum(0)

        angle = angle * B1.abs()
        phase = phase + B1.angle()

        # Unaffected magnetisation
        z_to_z = torch.cos(angle)
        p_to_p = torch.cos(angle/2)**2
        # Excited magnetisation
        z_to_p = -0.70710678118j * torch.sin(angle) * torch.exp(1j*phase)
        p_to_z = -z_to_p.conj()
        m_to_z = -z_to_p
        # Refocussed magnetisation
        m_to_p = (1 - p_to_p) * torch.exp(2j*phase)

        def calc_mag(ancestor: tuple) -> torch.Tensor:
            if ancestor[0] == 'zz':
                return ancestor[1].mag * z_to_z
            elif ancestor[0] == '++':
                return ancestor[1].mag * p_to_p
            elif ancestor[0] == 'z+':
                return ancestor[1].mag * z_to_p
            elif ancestor[0] == '+z':
                return ancestor[1].mag * p_to_z
            elif ancestor[0] == '-z':
                return ancestor[1].mag.conj() * m_to_z
            elif ancestor[0] == '-+':
                return ancestor[1].mag.conj() * m_to_p
            else:
                raise ValueError(f"Unknown transform {ancestor[0]}")

        # shape: events x coils
        adc = rep.adc_usage > 0
        rep_sig = torch.zeros(adc.sum(), coil_count,
                              dtype=torch.cfloat, device=data.device)

        # shape: events x 4
        trajectory = torch.cumsum(torch.cat([
            rep.gradm * grad_scale[None, :], rep.event_time[:, None]
        ], 1), 0)
        dt = rep.event_time

        total_time = rep.event_time.sum()
        r1 = torch.exp(-total_time / torch.abs(data.T1))
        r2 = torch.exp(-total_time / torch.abs(data.T2))

        # Use the same adc phase for all coils
        adc_rot = torch.exp(1j * rep.adc_phase).unsqueeze(1)

        # Calculate the additional phase carried of voxels because of motion
        motion_phase = 0
        if voxel_pos_func is not None:
            time = t0 + torch.cat([torch.zeros(1, device=data.device), trajectory[:, 3]])
            # Shape: events x voxels x 3
            voxel_traj = voxel_pos_func((time[:-1] + time[1:]) / 2) - data.voxel_pos[None, :, :]
            # Shape: events x voxels
            motion_phase = torch.einsum("evi, ei -> ev", voxel_traj, rep.gradm * grad_scale[None, :]).cumsum(0)
        t0 += total_time

        mag_p_rep = []
        mag_p.append(mag_p_rep)
        mag_z_rep = []
        mag_z.append(mag_z_rep)
        for dist in dists:
            # Create a list only containing ancestors that were simulated
            ancestors = list(filter(
                lambda edge: edge[1].mag is not None, dist.ancestors
            ))

            if dist.dist_type != 'z0' and dist.latent_signal < min_latent_signal:
                continue  # skip unimportant distributions
            if dist.dist_type != 'z0' and len(ancestors) == 0:
                continue  # skip dists for which no ancestors were simulated

            dist.mag = sum([calc_mag(ancestor) for ancestor in ancestors])
            if dist.dist_type == '+' and return_mag_p in [i, True]:
                mag_p_rep.append(dist.mag)
            if dist.dist_type in ['z0', 'z'] and return_mag_z in [i, True]:
                mag_z_rep.append(dist.mag)

            # The pre_pass already calculates kt_vec, but that does not
            # work with autograd -> we need to calculate it with torch
            if dist.dist_type == 'z0':
                dist.kt_vec = torch.zeros(4, device=data.device)
            elif ancestors[0][0] in ['-+', '-z']:
                dist.kt_vec = -1.0 * ancestors[0][1].kt_vec
            else:
                dist.kt_vec = ancestors[0][1].kt_vec.clone()

            # shape: events x 4
            dist_traj = dist.kt_vec + trajectory

            # Diffusion
            k2 = dist_traj[:, :3]
            k1 = torch.empty_like(k2)  # Calculate k-space at start of event
            k1[0, :] = dist.kt_vec[:3]
            k1[1:, :] = k2[:-1, :]
            # Integrate over each event to get b factor (lin. interp. grad)
            # Gradients are in rotations / meter, but we need rad / meter,
            # as integrating over exp(-ikr) assumes that kr is a phase in rad
            b = 1/3 * (2 * torch.pi)**2 * dt * (k1**2 + k1*k2 + k2**2).sum(1)
            # shape: events x voxels
            diffusion = torch.exp(-1e-9 * data.D * torch.cumsum(b, 0)[:, None])

            # NOTE: We are calculating the signal for samples that are not
            # measured (adc_usage == 0), which is, depending on the sequence,
            # produces an overhead of ca. 5 %. On the other hand, this makes
            # the code much simpler bc. we only have to apply the adc mask
            # once at the end instead of for every input. Change only if this
            # performance improvement is worth it. Repetitions without any adc
            # are already skipped because the pre-pass returns no signal.

            # NOTE: The bracketing / order of calculations is suprisingly
            # important for the numerical precision. An error of 4% is achieved
            # just by switching 2pi * (pos @ grad) to 2pi * pos @ grad

            if dist.dist_type == '+' and dist.emitted_signal >= min_emitted_signal:
                adc_dist_traj = dist_traj[adc, :]
                if isinstance(motion_phase, torch.Tensor):
                    adc_motion_phase = motion_phase[adc, :]
                else:
                    adc_motion_phase = motion_phase

                T2 = torch.exp(-trajectory[adc, 3:] / torch.abs(data.T2))
                T2dash = torch.exp(-torch.abs(adc_dist_traj[:, 3:]) / torch.abs(data.T2dash))
                rot = torch.exp(2j * np.pi * (
                    (adc_dist_traj[:, 3:] * data.B0) +
                    (adc_dist_traj[:, :3] @ data.voxel_pos.T) +
                    adc_motion_phase
                ))
                dephasing = data.dephasing_func(
                    adc_dist_traj[:, :3], data.nyquist)[:, None]

                # shape: events x voxels
                transverse_mag = (
                    # Add event dimension
                    1.41421356237 * dist.mag.unsqueeze(0)
                    * rot * T2 * T2dash * diffusion[adc, :] * dephasing
                )

                # (events x voxels) @ (voxels x coils) = (events x coils)
                dist_signal = transverse_mag @ coil_sensitivity
                rep_sig += dist_signal

            if dist.dist_type == '+':
                # Diffusion for whole trajectory + T2 relaxation + final phase carried by motion
                dist.mag = dist.mag * r2 * diffusion[-1, :] 
                if isinstance(motion_phase, torch.Tensor):
                    dist.mag = dist.mag * torch.exp(2j * np.pi * motion_phase[-1, :])
                dist.kt_vec = dist_traj[-1]
            else:  # z or z0
                k = torch.linalg.vector_norm(dist.kt_vec[:3])
                diffusion = torch.exp(-1e-9 * data.D * total_time * k**2)
                dist.mag = dist.mag * r1 * diffusion
            if dist.dist_type == 'z0':
                dist.mag = dist.mag + 1 - r1

        rep_sig *= adc_rot[adc]

        # Remove ancestors to save memory as we don't need them anymore.
        # When running with autograd this doesn't change memory consumption
        # bc. the values are still stored in the computation graph.
        for dist in dists:
            for ancestor in dist.ancestors:
                ancestor[1].mag = None

        signal.append(rep_sig)

    if print_progress:
        print(" - done")

    measured = torch.cat(signal)

    if return_mag_p is not None:
        if return_mag_z is not None:
            return measured, mag_p, mag_z
        else:
            return measured, mag_p
    elif return_mag_z is not None:
        return measured, mag_z
    else:
        return measured
