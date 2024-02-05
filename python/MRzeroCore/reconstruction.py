from __future__ import annotations
from numpy import pi
import torch


def reco_adjoint(signal: torch.Tensor,
                 kspace: torch.Tensor,
                 resolution: tuple[int, int, int] | float | None = None,
                 FOV: tuple[float, float, float] | float | None = None,
                 return_multicoil: bool = False,
                 ) -> torch.Tensor:
    """Adjoint reconstruction of the signal, based on a provided kspace.

    Parameters
    ----------
    signal : torch.Tensor
        A complex tensor containing the signal,
        shape (sample_count, coil_count)
    kspace : torch.Tensor
        A real tensor of shape (sample_count, 4) for the kspace trajectory
    resolution : (int, int, int) | float | None
        The resolution of the reconstruction. Can be either provided directly
        as tuple or set to None, in which case the resolution will be derived
        from the k-space (currently only for cartesian trajectories). A single
        float value will be used as factor for a derived resolution.
    FOV : (float, float, float) | float | None
        Because the adjoint reconstruction adapts to the k-space used
        for measurement, scaling gradients will not directly change the FOV of
        the reconstruction. All SimData phantoms have a normalized size of
        (1, 1, 1). Similar to the resolution, a value of None will
        automatically derive the FOV of the sequence based on the kspace. A
        float value can be used to scale this derived FOV.
    return_multicoil : bool
        Specifies if coils should be combined or returned separately.

    Returns
    -------
    torch.Tensor
        A complex tensor with the reconstructed image, the shape is given by
        the resolution.
    """
    res_scale = 1.0
    fov_scale = 1.0
    if isinstance(resolution, float):
        res_scale = resolution
        resolution = None
    if isinstance(FOV, float):
        fov_scale = FOV
        FOV = None

    # Atomatic detection of FOV - NOTE: only works for cartesian k-spaces
    # we assume that there is a sample at 0, 0 nad calculate the FOV
    # based on the distance on the nearest samples in x, y and z direction
    if FOV is None:
        def fov(t: torch.Tensor) -> float:
            t = t[t > 1e-3]
            return 1.0 if t.numel() == 0 else float(t.min())
        tmp = kspace[:, :3].abs()
        fov_x = fov_scale / fov(tmp[:, 0])
        fov_y = fov_scale / fov(tmp[:, 1])
        fov_z = fov_scale / fov(tmp[:, 2])
        FOV = (fov_x, fov_y, fov_z)
        print(f"Detected FOV: {FOV}")

    # Atomatic detection of resolution
    if resolution is None:
        def res(scale: float, fov: float, t: torch.Tensor) -> int:
            tmp = (scale * (fov * (t.max() - t.min()) + 1)).round()
            return max(int(tmp), 1)
        res_x = res(res_scale, FOV[0], kspace[:, 0])
        res_y = res(res_scale, FOV[1], kspace[:, 1])
        res_z = res(res_scale, FOV[2], kspace[:, 2])
        resolution = (res_x, res_y, res_z)
        print(f"Detected resolution: {resolution}")

    # Same grid as defined in SimData
    pos_x, pos_y, pos_z = torch.meshgrid(
        FOV[0] * torch.fft.fftshift(torch.fft.fftfreq(resolution[0], device=kspace.device)),
        FOV[1] * torch.fft.fftshift(torch.fft.fftfreq(resolution[1], device=kspace.device)),
        FOV[2] * torch.fft.fftshift(torch.fft.fftfreq(resolution[2], device=kspace.device)),
    )

    voxel_pos = torch.stack([
        pos_x.flatten(),
        pos_y.flatten(),
        pos_z.flatten()
    ], dim=1).t()

    NCoils = signal.shape[1]
    # assert NCoils == 1, "reconstruct currently does not support multicoil"

    # (Samples, 3) x (3, Voxels)
    phase = kspace[:, :3] @ voxel_pos
    # (Samples, Voxels): Rotation of all voxels at every event
    rot = torch.exp(-2j*pi * phase)  # Matches definition of forward DFT

    NCoils = signal.shape[1]

    if return_multicoil:
        return (signal.t() @ rot).view((NCoils, *resolution))
    elif NCoils == 1:
        return (signal.t() @ rot).view(resolution)
    else:
        return torch.sqrt(((torch.abs(signal.t() @ rot))**2).sum(0)).view(resolution)
