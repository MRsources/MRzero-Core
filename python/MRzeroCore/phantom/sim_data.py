from __future__ import annotations
from typing import Callable, Any
import torch
from numpy import pi


class SimData:
    """This class contains the physical data for simulating a MRI sequence.

    It is not intended to create this class directly, but rather to use one of
    the :class:`SimData` builders / loaders. Those are made fore specific
    tasks and can be converted into :class:`SimData`, but also attach
    metadata to the output so it can be converted back. The attributes of this
    class are nothing but the data needed for simulation, so it can describe
    a single voxel, randomly distributed voxels, a BrainWeb phantom, ...

    Attributes
    ----------
    PD : torch.Tensor
        Per voxel proton density
    T1 : torch.Tensor
        Per voxel T1 relaxation time (seconds)
    T2 : torch.Tensor
        Per voxel T2 relaxation time (seconds)
    T2dash : torch.Tensor
        Per voxel T2' dephasing time (seconds)
    D: torch.Tensor
        Isometric diffusion coefficients [10^-3 mm^2/s]
    B0 : torch.Tensor
        Per voxel B0 inhomogentity (Hertz)
    B1 : torch.Tensor
        (coil_count, voxel_count) Per coil and per voxel B1 inhomogenity
    coil_sens : torch.Tensor
        (coil_count, voxel_count) Per coil sensitivity (arbitrary units)
    size : torch.Tensor
        Physical size of the phantom. If a sequence with normalized gradients
        is simulated, size is used to scale them to match the phantom.
    avg_B1_trig : torch.Tensor
        (361, 3) values containing the PD-weighted avg of sin/cos/sin²(B1*flip)
    voxel_pos : torch.Tensor
        (voxel_count, 3) Voxel positions. These can be anywhere, but for easy
        sequence programming they should be in the range [-0.5, 0.5[
    nyquist : torch.Tensor
        (3, ) tensor: Maximum frequency encoded by the data
    dephasing_func : torch.Tensor -> torch.Tensor
        A function describing the intra-voxel dephasing. Maps a k-space
        trajectory (events, 3) to the measured attenuation (events).
    recover_func : SimData -> Any
        A function that can recover the original data that was used to create
        this instance. Usually a lambda that captures meta data like a mask.
    """

    def __init__(
        self,
        PD: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        T2dash: torch.Tensor,
        D: torch.Tensor,
        B0: torch.Tensor,
        B1: torch.Tensor,
        coil_sens: torch.Tensor,
        size: torch.Tensor,
        voxel_pos: torch.Tensor,
        nyquist: torch.Tensor,
        dephasing_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        recover_func: Callable[[SimData], Any] | None = None,
    ) -> None:
        """Create a :class:`SimData` instance based on the given tensors.

        All parameters must be of shape ``(voxel_count, )``, only B1 and
        coil_sens have an additional first dimension for multiple coils.

        Parameters
        ----------
        normalize : bool
            If true, applies B0 -= B0.mean(), B1 /= B1.mean(), PD /= PD.sum()
        """
        if not (PD.shape == T1.shape == T2.shape == T2dash.shape == B0.shape):
            raise Exception("Mismatch of voxel-data shapes")
        if not PD.ndim == 1:
            raise Exception("Data must be 1D (flattened)")
        if B1.ndim < 2 or B1.shape[1] != PD.numel():
            raise Exception("B1 must have shape [coils, voxel_count]")
        if coil_sens.ndim < 2 or coil_sens.shape[1] != PD.numel():
            raise Exception("coil_sens must have shape [coils, voxel_count]")

        self.PD = PD.clamp(min=0)
        self.T1 = T1.clamp(min=1e-6)
        self.T2 = T2.clamp(min=1e-6)
        self.T2dash = T2dash.clamp(min=1e-6)
        self.D = D.clamp(min=1e-6)
        self.B0 = B0.clone()
        self.B1 = B1.clone()
        self.coil_sens = coil_sens.clone()
        self.size = size.clone()
        self.voxel_pos = voxel_pos.clone()
        self.avg_B1_trig = calc_avg_B1_trig(B1, PD)
        self.nyquist = nyquist.clone()
        self.dephasing_func = dephasing_func
        self.recover_func = recover_func

    def cuda(self) -> SimData:
        """Move the simulation data to the default CUDA device.

        The returned :class:`SimData` is equivalent to :attr:`self` if the data
        already was on the GPU.
        """
        return SimData(
            self.PD.cuda(),
            self.T1.cuda(),
            self.T2.cuda(),
            self.T2dash.cuda(),
            self.D.cuda(),
            self.B0.cuda(),
            self.B1.cuda(),
            self.coil_sens.cuda(),
            self.size.cuda(),
            self.voxel_pos.cuda(),
            self.nyquist.cuda(),
            self.dephasing_func,
            self.recover_func
        )

    def cpu(self) -> SimData:
        """Move the simulation data to the CPU.

        The returned :class:`SimData` is equivalent to :attr:`self` if the data
        already was on the CPU.
        """
        return SimData(
            self.PD.cpu(),
            self.T1.cpu(),
            self.T2.cpu(),
            self.T2dash.cpu(),
            self.D.cpu(),
            self.B0.cpu(),
            self.B1.cpu(),
            self.coil_sens.cpu(),
            self.size.cpu(),
            self.voxel_pos.cpu(),
            self.nyquist.cpu(),
            self.dephasing_func,
            self.recover_func
        )

    @property
    def device(self) -> torch.device:
        """The device (either CPU or a CUDA device) the data is stored on."""
        return self.PD.device

    def recover(self) -> Any:
        """Recover the data that was used to build this instance."""
        if self.recover_func is None:
            raise Exception("No recover function was provided")
        else:
            return self.recover_func(self)


def calc_avg_B1_trig(B1: torch.Tensor, PD: torch.Tensor) -> torch.Tensor:
    """Return a (361, 3) tensor for B1 specific sin, cos and sin² values.

    This function calculates values for sin, cos and sin² for (0, 2pi) * B1 and
    then averages the results, weighted by PD. These 3 functions are the non
    linear parts of a rotation matrix, the resulting look up table can be used
    to calcualte averaged rotations for the whole phantom. This is useful for
    the pre-pass, to get better magnetization estmates even if the pre-pass is
    not spatially resolved.
    """
    # With pTx, there are now potentially multiple B1 maps with phase.
    # NOTE: This is a (probably suboptimal) workaround
    B1 = B1.sum(0).abs()

    B1 = B1.flatten()[:, None]  # voxels, 1
    PD = (PD.flatten() / PD.sum())[:, None]  # voxels, 1
    angle = torch.linspace(0, 2*pi, 361, device=PD.device)[None, :]  # 1, angle
    return torch.stack([
        (torch.sin(B1 * angle) * PD).sum(0),
        (torch.cos(B1 * angle) * PD).sum(0),
        (torch.sin(B1 * angle/2)**2 * PD).sum(0)
    ], dim=1).type(torch.float32)
