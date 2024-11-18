from __future__ import annotations
from typing import Literal, Optional, Dict
from warnings import warn
from scipy import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from .sim_data import SimData
from ..util import imshow


def sigmoid(trajectory: torch.Tensor, nyquist: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation of the sinc voxel dephasing function.

    The true dephasing function of a sinc-shaped voxel (in real space) is a
    box - function, with the FFT conform size [-nyquist, nyquist[. This is not
    differentiable, so we approximate the edges with a narrow sigmod at
    ±(nyquist + 0.5). The difference is neglegible at usual nyquist freqs.
    """
    return torch.prod(torch.sigmoid(
        (nyquist - trajectory.abs() + 0.5) * 100
    ), dim=1)


def sinc(trajectory: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    """Box voxel (real space) dephasing function.

    The size describes the total extends of the box shape.
    """
    return torch.prod(torch.sinc(trajectory * size), dim=1)


def identity(trajectory: torch.Tensor) -> torch.Tensor:
    """Point voxel (real space) dephasing function.

    There is no dephasing.
    """
    return torch.ones_like(trajectory[:, 0])


def generate_B0_B1(PD):
    # Generate a somewhat plausible B0 and B1 map.
    # Visually fitted to look similar to the numerical_brain_cropped
    x_pos, y_pos, z_pos = torch.meshgrid(
        torch.linspace(-1, 1, PD.shape[0]),
        torch.linspace(-1, 1, PD.shape[1]),
        torch.linspace(-1, 1, PD.shape[2]),
        indexing="ij"
    )
    B1 = torch.exp(-(0.4*x_pos**2 + 0.2*y_pos**2 + 0.3*z_pos**2))
    dist2 = (0.4*x_pos**2 + 0.2*(y_pos - 0.7)**2 + 0.3*z_pos**2)
    B0 = 7 / (0.05 + dist2) - 45 / (0.3 + dist2)
    # Normalize such that the weighted average is 0 or 1
    weight = PD / PD.sum()
    B0 -= (B0 * weight).sum()
    B1 /= (B1 * weight).sum()
    return B0, B1


class VoxelGridPhantom:
    """Class for using typical phantoms like those provided by BrainWeb.

    The data is assumed to be defined by a uniform cartesian grid of samples.
    As it is bandwidth limited, we assume that there is no signal above the
    Nyquist frequency. This leads to the usage of sinc-shaped voxels.

    Attributes
    ----------
    PD : torch.Tensor
        (sx, sy, sz) tensor containing the Proton Density
    T1 : torch.Tensor
        (sx, sy, sz) tensor containing the T1 relaxation
    T2 : torch.Tensor
        (sx, sy, sz) tensor containing the T2 relaxation
    T2dash : torch.Tensor
        (sx, sy, sz) tensor containing the T2' dephasing
    D : torch.Tensor
        (sx, sy, sz) tensor containing the Diffusion coefficient
    B0 : torch.Tensor
        (sx, sy, sz) tensor containing the B0 inhomogeneities
    B1 : torch.Tensor
        (coil_count, sx, sy, sz) tensor of RF coil profiles
    coil_sens : torch.Tensor
        (coil_count, sx, sy, sz) tensor of coil sensitivities
    size : torch.Tensor
        Size of the data, in meters.
    tissue_masks : Dict[str, torch.Tensor] | None
        Segmentation masks for different tissues. The keys are the tissue names
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
        phantom_motion=None,
        voxel_motion=None,
        tissue_masks: Optional[Dict[str,torch.Tensor]] = None,
    ) -> None:
        """Set the phantom attributes to the provided parameters.

        This function does no cloning nor contain any other funcionality. You
        probably want to use :meth:`brainweb` to load a phantom instead.
        """
        self.PD = torch.as_tensor(PD, dtype=torch.float32)
        self.T1 = torch.as_tensor(T1, dtype=torch.float32)
        self.T2 = torch.as_tensor(T2, dtype=torch.float32)
        self.T2dash = torch.as_tensor(T2dash, dtype=torch.float32)
        self.D = torch.as_tensor(D, dtype=torch.float32)
        self.B0 = torch.as_tensor(B0, dtype=torch.float32)
        self.B1 = torch.as_tensor(B1, dtype=torch.complex64)
        self.tissue_masks = tissue_masks
        if self.tissue_masks is None:
            self.tissue_masks = {}
        self.coil_sens = torch.as_tensor(coil_sens, dtype=torch.complex64)
        self.size = torch.as_tensor(size, dtype=torch.float32)

        self.phantom_motion = phantom_motion
        self.voxel_motion = voxel_motion

    def build(self, PD_threshold: float = 1e-6,
              voxel_shape: Literal["sinc", "box", "point"] = "sinc"
              ) -> SimData:
        """Build a :class:`SimData` instance for simulation.

        Arguments
        ---------
        PD_threshold : float
            All voxels with a proton density below this value are ignored.
        """
        mask = self.PD > PD_threshold

        shape = torch.tensor(mask.shape)
        pos_x, pos_y, pos_z = torch.meshgrid(
            self.size[0] *
            torch.fft.fftshift(torch.fft.fftfreq(
                int(shape[0]), device=self.PD.device)),
            self.size[1] *
            torch.fft.fftshift(torch.fft.fftfreq(
                int(shape[1]), device=self.PD.device)),
            self.size[2] *
            torch.fft.fftshift(torch.fft.fftfreq(
                int(shape[2]), device=self.PD.device)),
            indexing="ij"
        )

        voxel_pos = torch.stack([
            pos_x[mask].flatten(),
            pos_y[mask].flatten(),
            pos_z[mask].flatten()
        ], dim=1)

        if voxel_shape == "box":
            def dephasing_func(t, n): return sinc(t, 0.5 / n)
        elif voxel_shape == "sinc":
            def dephasing_func(t, n): return sigmoid(t, n)
        elif voxel_shape == "point":
            def dephasing_func(t, _): return identity(t)
        else:
            raise ValueError(f"Unsupported voxel shape '{voxel_shape}'")

        return SimData(
            self.PD[mask],
            self.T1[mask],
            self.T2[mask],
            self.T2dash[mask],
            self.D[mask],
            self.B0[mask],
            self.B1[:, mask],
            self.coil_sens[:, mask],
            self.size,
            voxel_pos,
            torch.as_tensor(shape, device=self.PD.device) / 2 / self.size,
            dephasing_func,
            recover_func=lambda data: recover(mask, data),
            phantom_motion=self.phantom_motion,
            voxel_motion=self.voxel_motion,
            tissue_masks=self.tissue_masks
        )
    
    @classmethod
    def brainweb(cls, file_name: str) -> VoxelGridPhantom:
        warn("brainweb() will be removed in a future version, use load() instead", DeprecationWarning)
        return cls.load(file_name)

    @classmethod
    def load(cls, file_name: str) -> VoxelGridPhantom:
        """Load a phantom from data produced by `generate_maps.py`."""
        with np.load(file_name) as data:
            T1 = torch.tensor(data['T1_map'])
            T2 = torch.tensor(data['T2_map'])
            T2dash = torch.tensor(data['T2dash_map'])
            PD = torch.tensor(data['PD_map'])
            D = torch.tensor(data['D_map'])
            try:
                B0 = torch.tensor(data['B0_map'])
                B1 = torch.tensor(data['B1_map'])
            except KeyError:
                B0, B1 = generate_B0_B1(PD)
            try:
                size = torch.tensor(data['FOV'], dtype=torch.float)
            except KeyError:
                size = torch.tensor([0.192, 0.192, 0.192])

            tissue_masks = {
                key: torch.tensor(mask)
                for key, mask in data.items()
                if key.startswith("tissue_")
            }
        if B1.ndim == 3:
            # Add coil-dimension
            B1 = B1[None, ...]

        return cls(
            PD, T1, T2, T2dash, D, B0, B1,
            torch.ones(1, *PD.shape), size,
            tissue_masks=tissue_masks
        )

    @classmethod
    def load_mat(
        cls,
        file_name: str,
        T2dash: float | torch.Tensor = 0.03,
        D: float | torch.Tensor = 1.0,
        size = [0.2, 0.2, 8e-3]
    ) -> VoxelGridPhantom:
        """Load a :class:`VoxelGridPhantom` from a .mat file.

        The file must contain exactly one array, of which the last dimension
        must have size 5. This dimension is assumed to specify (in that order):

        * Proton density
        * T1
        * T2
        * B0
        * B1

        All data is per-voxel, multiple coils are not yet supported.
        Data will be normalized (see constructor).

        Parameters
        ----------
        file_name : str
            Name of the matlab .mat file to be loaded
        T2dash : float, optional
            T2dash value set uniformly for all voxels, by default 0.03
        T2dash : float, optional
            Diffusion value set uniformly for all voxels, by default 1

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing the loaded data.

        Raises
        ------
        Exception
            The loaded file does not contain the expected data.
        """
        data = _load_tensor_from_mat(file_name)

        if data.ndim < 2 or data.shape[-1] != 5:
            raise Exception(
                f"Expected a tensor with shape [..., 5], "
                f"but got {list(data.shape)}"
            )

        if data.ndim == 3:
            # Expand to 3D: [x, y, i] -> [x, y, z, i]
            data = data.unsqueeze(2)

        if isinstance(T2dash, float):
            T2dash = torch.full_like(data[..., 0], T2dash)
        if isinstance(D, float):
            D = torch.full_like(data[..., 0], D)

        return cls(
            data[..., 0],  # PD
            data[..., 1],  # T1
            data[..., 2],  # T2
            T2dash,
            D,
            data[..., 3],  # B0
            data[..., 4][None, ...],  # B1
            coil_sens=torch.ones(1, *data.shape[:-1]),
            size=torch.as_tensor(size),
        )

    def slices(self, slices: list[int]) -> VoxelGridPhantom:
        """Generate a copy that only contains the selected slice(s).

        Parameters
        ----------
        slice: int or tuple
            The selected slice(s)

        Returns
        -------
        SimData
            A new instance containing the selected slice(s).
        """
        assert 0 <= any([slices]) < self.PD.shape[2]

        def select(tensor: torch.Tensor):
            return tensor[..., slices].view(
                *list(self.PD.shape[:2]), len(slices)
            )

        return VoxelGridPhantom(
            select(self.PD),
            select(self.T1),
            select(self.T2),
            select(self.T2dash),
            select(self.D),
            select(self.B0),
            select(self.B1).unsqueeze(0),
            select(self.coil_sens).unsqueeze(0),
            self.size.clone(),
            tissue_masks={
                key: mask[..., slices] for key, mask in self.tissue_masks.items()
            },
        )

    def scale_fft(self, x: int, y: int, z: int) -> VoxelGridPhantom:
        """This is experimental, shows strong ringing and is not recommended"""
        # This function currently only supports downscaling
        assert x <= self.PD.shape[0]
        assert y <= self.PD.shape[1]
        assert z <= self.PD.shape[2]

        # Normalize signal, otherwise magnitude changes with scaling
        norm = (
            (x / self.PD.shape[0]) *
            (y / self.PD.shape[1]) *
            (z / self.PD.shape[2])
        )
        # Center for FT
        cx = self.PD.shape[0] // 2
        cy = self.PD.shape[1] // 2
        cz = self.PD.shape[2] // 2

        def scale(map: torch.Tensor) -> torch.Tensor:
            FT = torch.fft.fftshift(torch.fft.fftn(map))
            FT = FT[
                cx - x // 2:cx + (x+1) // 2,
                cy - y // 2:cy + (y+1) // 2,
                cz - z // 2:cz + (z+1) // 2
            ] * norm
            return torch.fft.ifftn(torch.fft.ifftshift(FT)).abs()

        return VoxelGridPhantom(
            scale(self.PD),
            scale(self.T1),
            scale(self.T2),
            scale(self.T2dash),
            scale(self.D),
            scale(self.B0),
            scale(self.B1.squeeze()).unsqueeze(0),
            scale(self.coil_sens.squeeze()).unsqueeze(0),
            self.size.clone(),
            tissue_masks={
                key: scale(mask) for key, mask in self.tissue_masks.items()
            }
        )

    def interpolate(self, x: int, y: int, z: int) -> VoxelGridPhantom:
        """Return a resized copy of this :class:`SimData` instance.

        This uses torch.nn.functional.interpolate in 'area' mode, which is not
        very good: Assumes pixels are squares -> has strong aliasing.

        Parameters
        ----------
        x : int
            The new resolution along the 1st dimension
        y : int
            The new resolution along the 2nd dimension
        z : int
            The new resolution along the 3rd dimension
        mode : str
            Algorithm used for upsampling (via torch.nn.functional.interpolate)

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing resized tensors.
        """
        def resample(tensor: torch.Tensor) -> torch.Tensor:
            # Introduce additional dimensions: mini-batch and channels
            return torch.nn.functional.interpolate(
                tensor[None, None, ...], size=(x, y, z), mode='trilinear'
            )[0, 0, ...]

        def resample_multicoil(tensor: torch.Tensor) -> torch.Tensor:
            coils = tensor.shape[0]
            output = torch.zeros(coils, x, y, z, dtype=tensor.dtype)
            for i in range(coils):
                re = resample(torch.real(tensor[i, ...]))
                im = resample(torch.imag(tensor[i, ...]))
                output[i, ...] = re + 1j * im

            return output

        def resample_masks(tensors: Dict) -> Optional[Dict]:
            output = {}
            for key, mask in tensors.items():
                # Interpolate the mask
                interpolated_mask = torch.nn.functional.interpolate(
                    mask[None, None, ...].float(), size=(x, y, z), mode='area'
                )[0, 0, ...]
                # Store the result
                output[key] = interpolated_mask

            return output

        return VoxelGridPhantom(
            resample(self.PD),
            resample(self.T1),
            resample(self.T2),
            resample(self.T2dash),
            resample(self.D),
            resample(self.B0),
            resample_multicoil(self.B1),
            resample_multicoil(self.coil_sens),
            self.size.clone(),
            tissue_masks=resample_masks(self.tissue_masks)
        )

    def plot(self, plot_masks=False, plot_slice="center") -> None:
        """
        Print and plot all data stored in this phantom.

        Parameters
        ----------
        plot_masks : bool
            Plot tissue masks stored in this phantom (assumes they exist)
        slice : str | int
            If int, the specified slice is plotted. "center" plots the center
            slice and "all" plots all slices as a grid.
        """
        print("VoxelGridPhantom")
        print(f"size = {self.size}")
        # Center slice
        if plot_slice == "center":
            s = self.PD.shape[2] // 2
        elif plot_slice == "all":
            s = slice(None)
        elif plot_slice is int:
            s = plot_slice
        else:
            raise ValueError("expected plot_slice to be 'all', 'center' or an integer")
        # Warn if we only print a part of all data
        if self.coil_sens.shape[0] > 1:
            print(f"Plotting 1st of {self.coil_sens.shape[0]} coil sens maps")
        if self.B1.shape[0] > 1:
            print(f"Plotting 1st of {self.B1.shape[0]} B1 maps")
        if self.PD.shape[2] > 1:
            print(f"Plotting slice {s} / {self.PD.shape[2]}")

        # Determine the number of subplots needed
        num_plots = 9  # Base number of plots without masks
        if plot_masks:
            num_masks = len(self.tissue_masks)
            num_plots += num_masks

        # Calculate the grid size based on the number of plots
        cols = 3
        rows = int(np.ceil(num_plots / cols))

        plt.figure(figsize=(12, rows * 3))

        # Plot the basic maps
        plt.subplot(rows, cols, 1)
        plt.title("PD")
        imshow(self.PD[:, :, s], vmin=0)
        plt.colorbar()

        plt.subplot(rows, cols, 2)
        plt.title("T1")
        imshow(self.T1[:, :, s], vmin=0)
        plt.colorbar()

        plt.subplot(rows, cols, 3)
        plt.title("T2")
        imshow(self.T2[:, :, s], vmin=0)
        plt.colorbar()

        plt.subplot(rows, cols, 4)
        plt.title("T2'")
        imshow(self.T2dash[:, :, s], vmin=0)
        plt.colorbar()

        plt.subplot(rows, cols, 5)
        plt.title("D")
        imshow(self.D[:, :, s], vmin=0)
        plt.colorbar()

        plt.subplot(rows, cols, 7)
        plt.title("B0")
        imshow(self.B0[:, :, s])
        plt.colorbar()

        plt.subplot(rows, cols, 8)
        plt.title("B1")
        imshow(torch.abs(self.B1[0, :, :, s]))
        plt.colorbar()

        plt.subplot(rows, cols, 9)
        plt.title("coil sens")
        imshow(torch.abs(self.coil_sens[0, :, :, s]), vmin=0)
        plt.colorbar()

        # Conditionally plot masks if plot_masks is True
        if plot_masks:
            for i, (key, mask) in enumerate(self.tissue_masks.items()):
                plt.subplot(rows, cols, 10 + i)
                plt.title(key)
                imshow(mask)
                plt.colorbar()

        plt.tight_layout()
        plt.show()

    def plot3D(self, data2print: int = 0) -> None:
        """Print and plot all slices of one selected data stored in this phantom."""
        print("VoxelGridPhantom")
        print(f"size = {self.size}")
        print()

        label = ['PD', 'T1', 'T2', "T2'", "D", "B0", "B1", "coil sens"]

        tensors = [
            self.PD, self.T1, self.T2, self.T2dash, self.D, self.B0,
            self.B1.squeeze(0), self.coil_sens
        ]

        # Warn if we only print a part of all data
        print(f"Plotting {label[data2print]}")

        tensor = tensors[data2print].squeeze(0)

        util.plot3D(tensor, figsize=(20, 5))
        plt.title(label[data2print])
        plt.show()


def recover(mask, sim_data: SimData) -> VoxelGridPhantom:
    """Provided to :class:`SimData` to reverse the ``build()``"""
    def to_full(sparse):
        assert sparse.ndim < 3
        if sparse.ndim == 2:
            full = torch.zeros(
                [sparse.shape[0], *mask.shape], dtype=sparse.dtype)
            full[:, mask] = sparse.cpu()
        else:
            full = torch.zeros(mask.shape)
            full[mask] = sparse.cpu()
        return full

    return VoxelGridPhantom(
        to_full(sim_data.PD),
        to_full(sim_data.T1),
        to_full(sim_data.T2),
        to_full(sim_data.T2dash),
        to_full(sim_data.D),
        to_full(sim_data.B0),
        to_full(sim_data.B1),
        to_full(sim_data.coil_sens),
        sim_data.size
    )


def _load_tensor_from_mat(file_name: str) -> torch.Tensor:
    mat = io.loadmat(file_name)

    keys = [
        key for key in mat
        if not (key.startswith('__') and key.endswith('__'))
    ]

    arrays = [mat[key] for key in keys if isinstance(mat[key], np.ndarray)]

    if len(keys) == 0:
        raise Exception("The loaded mat file does not contain any variables")

    if len(arrays) != 1:
        raise Exception("The loaded mat file must contain exactly one array")

    return torch.from_numpy(arrays[0]).float()
