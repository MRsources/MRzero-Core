from __future__ import annotations
from typing import Callable
import torch
from numpy import pi
import matplotlib.pyplot as plt
from .sim_data import SimData


def heaviside(t: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Sinc voxel (real space) dephasing function.

    The size describes the distance from the center to the first zero crossing.
    This is not differentiable because of the discontinuity at ±size/2.
    """
    return torch.prod(torch.heaviside(
        0.5/size - t.abs(), torch.tensor(0.5)
    ), dim=1)


def sigmoid(t: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation of the sinc voxel dephasing function.

    The size describes the distance from the center to the first zero crossing.
    A narrow sigmoid is used to avoid the discontinuity of a heaviside.
    """
    return torch.prod(torch.sigmoid((0.5/size - t.abs() + 0.5) * 100), dim=1)


def sinc(t: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Box voxel (real space) dephasing function.

    The size describes the total extends of the box shape.
    """
    return torch.prod(torch.sinc(t * size), dim=1)


def gauss(t: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Normal distribution shaped voxel dephasing function.

    This function is not normalized. The size describes the variance.
    """
    return torch.prod(torch.exp(-0.5 * size * t**2), dim=1)


class CustomVoxelPhantom:
    """Class for manually specifying phantoms from a list of voxels.

    This can be useful to test the simulation or to simulate the point spread
    function. All voxels have the same size and shape but can have different
    physical properties.

    Attributes
    ----------
    voxel_pos : torch.Tensor
        (voxel_count, 3) tensor containing the position of all voxels
    PD : torch.Tensor
        1D tensor containing the Proton Density of all voxels
    T1 : torch.Tensor
        1D tensor containing the T1 relaxation of all voxels
    T2 : torch.Tensor
        1D tensor containing the T2 relaxation of all voxels
    T2dash : torch.Tensor
        1D tensor containing the T2' dephasing of all voxels
    D : torch.Tensor
        1D tensor containing the Diffusion coefficient of all voxels
    B0 : torch.Tensor
        1D tensor containing the dB0 off-resonance
    B1 : torch.Tensor
        1D tensor containing the relative B1 field strength
    voxel_shape : str
        Can be one of ``["sinc", "exact_sinc", "box", "gauss"]``
    voxel_size : torch.Tensor
        3-element tensor containing the size of a voxel
    """

    def __init__(self,
                 pos: list[list[float]] | torch.Tensor,
                 PD: float | list[float] | torch.Tensor = 1.0,
                 T1: float | list[float] | torch.Tensor = 1.5,
                 T2: float | list[float] | torch.Tensor = 0.1,
                 T2dash: float | list[float] | torch.Tensor = 0.05,
                 D: float | list[float] | torch.Tensor = 1,
                 B0: float | list[float] | torch.Tensor = 0,
                 B1: float | list[float] | torch.Tensor = 1,
                 voxel_size=0.1, voxel_shape="sinc"
                 ) -> None:
        """Create a phantom consisting of manually placed voxels.

        See :class:`CustomVoxelPhantom` attributes for explanation of the
        parameters. They can be single floats to set all voxels, or anything
        that can be converted to a 1D torch.Tensor for individual values for
        every voxel.
        """
        pos = torch.tensor(pos)
        assert pos.ndim == 2
        assert pos.shape[1] == 3

        voxel_size = torch.tensor(voxel_size).squeeze()
        if voxel_size.ndim == 0:
            voxel_size = torch.full((3, ), voxel_size)
        assert voxel_size.ndim == 1 and voxel_size.numel() == 3

        assert voxel_shape in ["sinc", "exact_sinc", "box", "gauss"]

        def expand(t) -> torch.Tensor:
            # Input must either be a scalar or a 1D tensor with the correct len
            t = torch.tensor(t).squeeze()
            if t.ndim == 0:
                return torch.full((pos.shape[0], ), t)
            elif t.ndim == 1:
                assert t.numel() == pos.shape[0]
                return t
            else:
                assert False

        self.voxel_pos = pos
        self.PD = expand(PD)
        self.T1 = expand(T1)
        self.T2 = expand(T2)
        self.T2dash = expand(T2dash)
        self.D = expand(D)
        self.B0 = expand(B0)
        self.B1 = expand(B1)
        self.voxel_shape = voxel_shape
        self.voxel_size = voxel_size

    def build(self) -> SimData:
        """Build a :class:`SimData` instance for simulation."""
        # TODO: until the dephasing func fix is here, this only works on the
        # device self.voxel_size happens to be on

        return SimData(
            self.PD,
            self.T1,
            self.T2,
            self.T2dash,
            self.D,
            self.B0,
            self.B1[None, :],
            torch.ones(1, self.PD.numel()),
            torch.tensor([0.2, 0.2, 0.2]),  # FOV for diffusion
            self.voxel_pos,
            torch.tensor([float('inf'), float('inf'), float('inf')]),
            build_dephasing_func(self.voxel_shape, self.voxel_size),
            recover_func=lambda d: recover(self.voxel_size, self.voxel_shape, d)
        )

    def generate_PD_map(self) -> torch.Tensor:
        """Convenience function for MRTwin_pulseq to generate a PD map."""
        kx, ky = torch.meshgrid(
            torch.linspace(-64, 63, 128),
            torch.linspace(-64, 63, 128),
        )
        trajectory = torch.stack([
            kx.flatten(),
            ky.flatten(),
            torch.zeros(kx.numel()),
        ], dim=1)
        PD_kspace = torch.zeros(128, 128, dtype=torch.cfloat)

        # All voxels have the same shape -> same intra-voxel dephasing
        dephasing = build_dephasing_func(
            self.voxel_shape, self.voxel_size
        )(trajectory, None)

        # Iterate over all voxels and render them into the kspaces
        for i in range(self.PD.numel()):
            rot = torch.exp(-2j*pi * (trajectory @ self.voxel_pos[i, :]))
            kspace = (rot * dephasing).view(128, 128)
            PD_kspace += kspace * self.PD[i]

        return torch.fft.fftshift(torch.fft.ifft2(PD_kspace)).abs()[:, :, None]

    def plot(self) -> None:
        """Print and plot all data stored in this phantom."""
        # Best way to accurately plot this is to generate a k-space -> FFT
        # We only render a 2D image with a FOV of 1
        kx, ky = torch.meshgrid(
            torch.linspace(-64, 63, 128),
            torch.linspace(-64, 63, 128),
        )
        trajectory = torch.stack([
            kx.flatten(),
            ky.flatten(),
            torch.zeros(kx.numel()),
        ], dim=1)
        PD_kspace = torch.zeros(128, 128, dtype=torch.cfloat)
        T1_kspace = torch.zeros(128, 128, dtype=torch.cfloat)
        T2_kspace = torch.zeros(128, 128, dtype=torch.cfloat)
        T2dash_kspace = torch.zeros(128, 128, dtype=torch.cfloat)
        D_kspace = torch.zeros(128, 128, dtype=torch.cfloat)

        # All voxels have the same shape -> same intra-voxel dephasing
        dephasing = build_dephasing_func(
            self.voxel_shape, self.voxel_size
        )(trajectory, None)

        # Iterate over all voxels and render them into the kspaces
        for i in range(self.PD.numel()):
            rot = torch.exp(-2j*pi * (trajectory @ self.voxel_pos[i, :]))
            kspace = (rot * dephasing).view(128, 128)
            PD_kspace += kspace * self.PD[i]
            T1_kspace += kspace * self.T1[i]
            T2_kspace += kspace * self.T2[i]
            T2dash_kspace += kspace * self.T2dash[i]
            D_kspace += kspace * self.D[i]

        PD = torch.fft.fftshift(torch.fft.ifft2(PD_kspace))
        T1 = torch.fft.fftshift(torch.fft.ifft2(T1_kspace))
        T2 = torch.fft.fftshift(torch.fft.ifft2(T2_kspace))
        T2dash = torch.fft.fftshift(torch.fft.ifft2(T2dash_kspace))
        D = torch.fft.fftshift(torch.fft.ifft2(D_kspace))

        maps = [PD, T1, T2, T2dash, D]
        titles = ["$PD$", "$T_1$", "$T_2$", "$T_2#$", "$D$"]

        print("CustomVoxelPhantom")
        print(f"Voxel shape: {self.voxel_shape}")
        print(f"Voxel size: {self.voxel_size}")
        plt.figure(figsize=(12, 6))
        for i in range(5):
            plt.subplot(231 + i)
            plt.title(titles[i])
            plt.imshow(maps[i].abs().T, origin="lower", vmin=0)
            plt.xticks([-0.5, 63.5, 127.5], [-1, 0, 1])
            plt.yticks([-0.5, 63.5, 127.5], [-1, 0, 1])
            plt.grid()
            plt.colorbar()
        plt.show()


# TODO: dephasing funcs can store tensors that need to switch device on demand.
# Maybe it is necessary to make a dephasing func class that can do that.
def build_dephasing_func(shape: str, size: torch.Tensor,
                         ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Helper function to get the correct dephasing function."""
    if shape == "sinc":
        return lambda t, _: sigmoid(t, size)
    elif shape == "exact_sinc":
        return lambda t, _: heaviside(t, size)
    elif shape == "box":
        return lambda t, _: sinc(t, size)
    elif shape == "gauss":
        return lambda t, _: gauss(t, size)
    else:
        raise ValueError("shape not implemented:", self.voxel_shape)


def recover(voxel_size: torch.Tensor, voxel_shape: str, sim_data: SimData
            ) -> CustomVoxelPhantom:
    """Provided to :class:`SimData` to reverse the ``build()``"""
    # We don't recover B0, B1 and coil_sens per design - but that could change
    return CustomVoxelPhantom(
        # These can be taken directly from the phantom
        sim_data.voxel_pos.cpu(),
        sim_data.PD.cpu(),
        sim_data.T1.cpu(),
        sim_data.T2.cpu(),
        sim_data.T2dash.cpu(),
        sim_data.D.cpu(),
        0,  # B0
        1,  # B1
        # These must be provided (captured in the recover_func lambda)
        voxel_size,
        voxel_shape
    )
