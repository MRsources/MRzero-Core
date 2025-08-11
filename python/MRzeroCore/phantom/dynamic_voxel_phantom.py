from __future__ import annotations
from typing import Callable, Any, Literal, Optional, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .sim_data import SimData, calc_avg_B1_trig
from ..util import imshow
from .voxel_grid_phantom import (
    VoxelGridPhantom, 
    generate_B0_B1, 
    identity, 
    recover, 
    sinc, 
    sigmoid
)


class DynamicSimData(SimData):
    """This class contains the dynamic physical data for simulating a MRI sequence.

    It is not intended to create this class directly, but rather to use one of
    the :class:`DynamicSimData` builders / loaders. Those are made fore specific
    tasks and can be converted into :class:`DynamicSimData`, but also attach
    metadata to the output so it can be converted back. The attributes of this
    class are nothing but the data needed for simulation, so it can describe
    a single voxel, randomly distributed voxels, a pyMRXCAT phantom, ...

    Attributes
    ----------
    PD : torch.Tensor
        Per voxel proton density
    T1 : torch.Tensor
        Per voxel T1 relaxation time (seconds) for each repetition time.
    T2 : torch.Tensor
        Per voxel T2 relaxation time (seconds) for each repetition time.
    T2dash : torch.Tensor
        Per voxel T2' dephasing time (seconds) for each repetition time.
    D: torch.Tensor
        Isometric diffusion coefficients [10^-3 mm^2/s] for each repetition time.
    B0 : torch.Tensor
        Per voxel B0 inhomogentity (Hertz) for each repetition time.
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
    recover_func : DynamicSimData -> Any
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
    recover_func: Callable[[DynamicSimData], Any] | None = None,
    phantom_motion=None,
    voxel_motion=None,
    tissue_masks: Optional[Dict[str,torch.Tensor]] = None,
) -> None:
        """Create a :class:`DynamicSimData` instance based on the given tensors.

        All parameters must be of shape ``(voxel_count, )``, only B1 and
        coil_sens have an additional first dimension for multiple coils.

        Parameters
        ----------
        normalize : bool
            If true, applies B0 -= B0.mean(), B1 /= B1.mean(), PD /= PD.sum()
        """
        if not (PD.shape == T1.shape[1:] == T2.shape[1:] == T2dash.shape[1:] == B0.shape[1:]):
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
        self.tissue_masks = tissue_masks
        if self.tissue_masks is None:
            self.tissue_masks = {}
        self.coil_sens = coil_sens.clone()
        self.size = size.clone()
        self.voxel_pos = voxel_pos.clone()
        self.avg_B1_trig = calc_avg_B1_trig(B1, PD)
        self.nyquist = nyquist.clone()
        self.dephasing_func = dephasing_func
        self.recover_func = recover_func

        self.phantom_motion = phantom_motion
        self.voxel_motion = voxel_motion
    
    def cuda(self) -> DynamicSimData:
        """Move the simulation data to the default CUDA device.

        The returned :class:`DynamicSimData` is equivalent to :attr:`self` if the data
        already was on the GPU.
        """
        return DynamicSimData(
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
            self.recover_func,
            self.phantom_motion,
            self.voxel_motion,
            tissue_masks={
                k: v.cuda() for k, v in self.tissue_masks.items()
            },
        )

    def cpu(self) -> DynamicSimData:
        """Move the simulation data to the CPU.

        The returned :class:`DynamicSimData` is equivalent to :attr:`self` if the data
        already was on the CPU.
        """
        return DynamicSimData(
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
            self.recover_func,
            self.phantom_motion,
            self.voxel_motion,
            tissue_masks={
                k: v.cpu() for k, v in self.tissue_masks.items()
            },
        )


class DynamicVoxelPhantom(VoxelGridPhantom):
    """Class for using typical dynamic phantoms like those provided by pyMRXCAT.

    The data is assumed to be defined by a uniform cartesian grid of samples.
    As it is bandwidth limited, we assume that there is no signal above the
    Nyquist frequency. This leads to the usage of sinc-shaped voxels.

    Attributes
    ----------
    PD : torch.Tensor
        (sx, sy, sz) tensor containing the Proton Density [a.u.].
    T1 : torch.Tensor
        (time_steps, sx, sy, sz) tensor containing the T1 relaxation values per voxel over time.
        Each time step represents a snapshot of the 3D T1 map [s].
    T2 : torch.Tensor
        (time_steps, sx, sy, sz) tensor containing the T2 relaxation values per voxel over time.
        Each time step represents a snapshot of the 3D T2 map [s].
    T2dash : torch.Tensor
        (time_steps, sx, sy, sz) tensor containing the T2' dephasing per voxel over time.
        Each time step represents a snapshot of the 3D T2' map [s].
    D : torch.Tensor
        (time_steps, sx, sy, sz) tensor containing the Diffusion coefficient per voxel over time.
        Each time step represents a snapshot of the 3D Diffusion map [10^-3 mm² / s].
    B0 : torch.Tensor
        (time_steps, sx, sy, sz) tensor containing the B0 inhomogeneities [Hz].
    B1 : torch.Tensor
        (coil_count, sx, sy, sz) tensor of RF coil profiles.
    coil_sens : torch.Tensor
        (coil_count, sx, sy, sz) tensor of coil sensitivities.
    size : torch.Tensor
        Size of the data, in meters.
    tissue_masks : Dict[str, torch.Tensor] | None
        Segmentation masks for different tissues. The keys are the tissue names.
    time_points : torch.Tensor
        (time_steps,) tensor containing the time after the beginning of the acquisition of T1/T2 snapshot [s].
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
        time_points: torch.Tensor = torch.tensor(0.),
    ) -> None:
        """Set the phantom attributes to the provided parameters.

        This function does no cloning nor contain any other functionality. You
        probably want to use :meth:`load` to load a phantom instead.
        """
        self.PD = torch.as_tensor(PD, dtype=torch.float32)
        self.T1 = torch.as_tensor(T1, dtype=torch.float32)
        if self.T1.ndim==3:
            self.T1 = self.T1.expand(len(time_points), *self.T1.shape)
        self.T2 = torch.as_tensor(T2, dtype=torch.float32)
        if self.T2.ndim==3:
            self.T2 = self.T2.expand(len(time_points), *self.T2.shape)
        self.T2dash = torch.as_tensor(T2dash, dtype=torch.float32)
        if self.T2dash.ndim==3:
            self.T2dash = self.T2dash.expand(len(time_points), *self.T2dash.shape)
        self.D = torch.as_tensor(D, dtype=torch.float32)
        if self.D.ndim==3:
            self.D = self.D.expand(len(time_points), *self.D.shape)
        self.B0 = torch.as_tensor(B0, dtype=torch.float32)
        if self.B0.ndim==3:
            self.B0 = self.B0.expand(len(time_points), *self.B0.shape)
        self.B1 = torch.as_tensor(B1, dtype=torch.complex64)
        self.tissue_masks = tissue_masks
        if self.tissue_masks is None:
            self.tissue_masks = {}
        self.coil_sens = torch.as_tensor(coil_sens, dtype=torch.complex64)
        self.size = torch.as_tensor(size, dtype=torch.float32)
        self.time_points = time_points

        self.phantom_motion = phantom_motion
        self.voxel_motion = voxel_motion
        
    @classmethod
    def load(cls, file_name: str) -> VoxelGridPhantom:
        """Load a phantom from data produced by `generate_maps.py`."""
        with np.load(file_name) as data:
            T1 = torch.tensor(data['T1_map'])
            T2 = torch.tensor(data['T2_map'])
            PD = torch.tensor(data['PD_map'])
            try:
                T2dash = torch.tensor(data['T2dash_map'])
            except:
                T2dash = torch.full_like(PD, 3*1e-2)
            try:
                D = torch.tensor(data['D_map'])
            except:
                D = torch.zeros_like(PD)
            try:
                B0 = torch.tensor(data['B0_map'])
                B1 = torch.tensor(data['B1_map'])
            except KeyError:
                B0, B1 = generate_B0_B1(PD)
            try:
                size = torch.tensor(data['FOV'], dtype=torch.float)
            except KeyError:
                size = torch.tensor([0.192, 0.192, 0.192])
            try:
                time_points = torch.tensor(data['time_points'])
            except KeyError:
                time_points = torch.arange(0, 600, dtype=torch.float) # 10 min by default
            try:
                coil_sens = torch.tensor(data['coil_sens'])
            except KeyError:
                coil_sens = torch.ones(1, *PD.shape)
                
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
            coil_sens, size,
            tissue_masks=tissue_masks, 
            time_points=time_points,
        )
    
    def build(self, repetition_times, PD_threshold: float = 1e-6,
              voxel_shape: Literal["sinc", "box", "point"] = "sinc"
              ) -> DynamicSimData:
        """Build a :class:`DynamicSimData` instance for simulation.

        Arguments
        ---------
        repetition_times: torch.Tensor
            1D tensor containing the times for each repetition in the sequence.
        PD_threshold : float
            All voxels with a proton density below this value are ignored.
        voxel_shape: str
            shape of the voxel used for simulation. Default to sinc shape.
        """
        T1_rep, T2_rep, T2dash_rep, D_rep, B0_rep = self.compute_param_at_repetition(repetition_times)
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

        return DynamicSimData(
            self.PD[mask],
            T1_rep[:,mask],
            T2_rep[:,mask],
            T2dash_rep[:,mask],
            D_rep[:,mask],
            B0_rep[:,mask],
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
    
    def compute_param_at_repetition(self, repetition_times: torch.Tensor):
        """Computes the T1, T2, T2', D and B0 based on the provided repetition times.

        Arguments
        ---------
        repetition_times: torch.Tensor
            1D tensor containing the times for each repetition in the sequence.
        """
        # Find the indices where repetition_times would fit in time_points (sorted).
        indices = torch.searchsorted(self.time_points, repetition_times, side='left')

        # Calculate differences to the left and right to determine the closest point.
        left_diff = repetition_times - self.time_points[indices - 1]
        right_diff = self.time_points[indices] - repetition_times
        
        # Choose the closest index by comparing left and right differences.
        closest_indices = torch.where(left_diff <= right_diff, indices - 1, indices)
        T1_rep = self.T1[closest_indices]
        T2_rep = self.T2[closest_indices]
        T2dash_rep = self.T2dash[closest_indices]
        D_rep = self.D[closest_indices]
        B0_rep = self.B0[closest_indices]
        return T1_rep, T2_rep, T2dash_rep, D_rep, B0_rep
    
    def slices(self, slices: list[int]) -> DynamicVoxelPhantom:
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

        return DynamicVoxelPhantom(
            select(self.PD),
            torch.stack([select(T1_rep) for T1_rep in self.T1]),
            torch.stack([select(T2_rep) for T2_rep in self.T2]),
            torch.stack([select(T2dash_rep) for T2dash_rep in self.T2dash]),
            torch.stack([select(D_rep) for D_rep in self.D]),
            torch.stack([select(B0_rep) for B0_rep in self.B0]),
            torch.stack([select(b1) for b1 in self.B1]),
            torch.stack([select(c) for c in self.coil_sens]),
            self.size.clone(),
            tissue_masks={
                key: mask[..., slices] for key, mask in self.tissue_masks.items()
            },
            time_points=self.time_points,
        )
    def plot(self, plot_masks=False, plot_slice="center", time_unit="s", display_units=False, t=0) -> None:
        """
        Print and plot all data stored in this phantom.

        Parameters
        ----------
        plot_masks : bool
            Plot tissue masks stored in this phantom (assumes they exist)
        plot_slice : str | int
            If int, the specified slice is plotted. "center" plots the center
            slice and "all" plots all slices as a grid.
        time_unit : str
            Unit used to display T1, T2 and T2dash. Either "s" or "ms". Default to "s"
        display_units : bool
            If True, display parameter units. Default to False
        t : int
            Time frame to display. Default to 0.
        """
        assert time_unit in ["ms", "s"], "time_unit should be either 's' or 'ms'"
        time_factor = 1e3 if time_unit=="ms" else 1
        print("VoxelGridPhantom")
        print(f"size = {self.size}")
        # Center slice
        if plot_slice == "center":
            s = self.PD.shape[2] // 2
        elif plot_slice == "all":
            s = slice(None)
        elif isinstance(plot_slice, int):
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
        if self.T1.shape[0] > 1:
            print(f"Plotting 1st of {self.T1.shape[0]} time frames")

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
        ax = plt.subplot(rows, cols, 1)
        plt.title("PD (a.u.)") if display_units else plt.title("PD")
        imshow(self.PD[:, :, s], vmin=0)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 2)
        plt.title("T1 (%s)" % time_unit) if display_units else plt.title("T1")
        imshow(self.T1[t,:, :, s]*time_factor, vmin=0)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 3)
        plt.title("T2 (%s)" % time_unit) if display_units else plt.title("T2")
        imshow(self.T2[t,:, :, s]*time_factor, vmin=0)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 4)
        plt.title("T2' (%s)" % time_unit) if display_units else plt.title("T2'")
        imshow(self.T2dash[t,:, :, s]*time_factor, vmin=0)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 5)
        plt.title("D (x$10^{-3}$ mm$^2$/s)") if display_units else plt.title("D")
        imshow(self.D[t,:, :, s], vmin=0)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 7)
        plt.title("B0 (Hz)") if display_units else plt.title("B0")
        imshow(self.B0[t,:, :, s])
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 8)
        plt.title("B1 (a.u.)") if display_units else plt.title("B1")
        imshow(torch.abs(self.B1[0, :, :, s]))
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 9)
        plt.title("coil sens (a.u.)") if display_units else plt.title("coil sens")
        imshow(torch.abs(self.coil_sens[0, :, :, s]), vmin=0)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
        
        # Conditionally plot masks if plot_masks is True
        if plot_masks:
            for i, (key, mask) in enumerate(self.tissue_masks.items()):
                plt.subplot(rows, cols, 10 + i)
                plt.title(key)
                imshow(mask)
                plt.colorbar()
                plt.axis('off')

        plt.tight_layout()
        plt.show()
    
    def plot_dynamic(self, plot_masks=False, plot_slice="center", time_unit="s", display_units=False,
                     delay_frame=0.1, repeat=True, save_gif=False, gif_filename='dynamic_Phantom.gif') -> None:
        """
        Print and plot all data stored in this phantom.

        Parameters
        ----------
        plot_masks : bool
            Plot tissue masks stored in this phantom (assumes they exist)
        plot_slice : str | int
            If int, the specified slice is plotted. "center" plots the center
            slice and "all" plots all slices as a grid.
        time_unit : str
            Unit used to display T1, T2 and T2dash. Either "s" or "ms". Default to "s"
        display_units : bool
            If True, display parameter units. Default to False
        delay_frame : float
            Delay between time frames in seconds. Default to 0.1 seconds.
        repeat : bool
            Whether to loop the animation once it ends. Default is True
        save_gif : bool
            If True, the animation is saved as a GIF file instead of being displayed. Default is False
        gif_filename : str
            Filename (including extension) used when saving the animation as a GIF. Default is "dynamic_Phantom.gif"
        """
        assert time_unit in ["ms", "s"], "time_unit should be either 's' or 'ms'"
        time_factor = 1e3 if time_unit=="ms" else 1
        print("VoxelGridPhantom")
        print(f"size = {self.size}")
        # Center slice
        if plot_slice == "center":
            s = self.PD.shape[2] // 2
        elif plot_slice == "all":
            s = slice(None)
        elif isinstance(plot_slice, int):
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

        fig = plt.figure(figsize=(12, rows * 3))
        suptitle = fig.suptitle(f"Time: {self.time_points[0]:.2f} s", fontsize=16)

        # Plot the basic maps
        ax = plt.subplot(rows, cols, 1)
        plt.title("PD (a.u.)") if display_units else plt.title("PD")
        imshow(self.PD[:, :, s], vmin=0)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 2)
        plt.title("T1 (%s)" % time_unit) if display_units else plt.title("T1")
        img_T1 = imshow(self.T1[0,:, :, s]*time_factor, vmin=0, animated=True)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 3)
        plt.title("T2 (%s)" % time_unit) if display_units else plt.title("T2")
        img_T2 = imshow(self.T2[0,:, :, s]*time_factor, vmin=0, animated=True)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 4)
        plt.title("T2' (%s)" % time_unit) if display_units else plt.title("T2'")
        img_T2dash = imshow(self.T2dash[0,:, :, s]*time_factor, vmin=0, animated=True)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 5)
        plt.title("D (x$10^{-3}$ mm$^2$/s)") if display_units else plt.title("D")
        img_D = imshow(self.D[0,:, :, s], vmin=0, animated=True)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 7)
        plt.title("B0 (Hz)") if display_units else plt.title("B0")
        img_B0 = imshow(self.B0[0,:, :, s], animated=True)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 8)
        plt.title("B1 (a.u.)") if display_units else plt.title("B1")
        imshow(torch.abs(self.B1[0, :, :, s]))
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)

        ax = plt.subplot(rows, cols, 9)
        plt.title("coil sens (a.u.)") if display_units else plt.title("coil sens")
        imshow(torch.abs(self.coil_sens[0, :, :, s]), vmin=0)
        plt.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax=cax)
        
        # Conditionally plot masks if plot_masks is True
        if plot_masks:
            for i, (key, mask) in enumerate(self.tissue_masks.items()):
                plt.subplot(rows, cols, 10 + i)
                plt.title(key)
                imshow(mask)
                plt.colorbar()
                plt.axis('off')

        plt.tight_layout()
        
        def data_gen():
            T1_maps = self.T1[:, :, :, s]*time_factor
            T2_maps = self.T2[:, :, :, s]*time_factor
            T2dash_maps = self.T2dash[:, :, :, s]*time_factor
            D_maps = self.D[:, :, :, s]
            B0_maps = self.B0[:, :, :, s]
            for t, time in enumerate(self.time_points):
                yield T1_maps[t], T2_maps[t], T2dash_maps[t], D_maps[t], B0_maps[t], time
                
        def run(data):
            T1_map, T2_map, T2dash_map, D_map, B0_map, time = data
            img_T1.set_data(T1_map.T)
            img_T2.set_data(T2_map.T)
            img_T2dash.set_data(T2dash_map.T)
            img_D.set_data(D_map.T)
            img_B0.set_data(B0_map.T)
            # Set timnig in a nice format
            if time >= 60:
                minutes, seconds = divmod(time.item(), 60)
                suptitle.set_text(f"Time: {int(minutes):02d}:{int(seconds):02d}")
            else:
                suptitle.set_text(f"Time: {time:.1f} s")
            return [img_T1, img_T2, img_T2dash, img_D, img_B0]
        ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=delay_frame*1e3, repeat=repeat)
        
        if save_gif:
            print(f"Saving animation to {gif_filename}...")
            ani.save(gif_filename, writer=PillowWriter(fps=5))
            print("Saved.")
        
        plt.show()
    
    def interpolate(self, x: int, y: int, z: int) -> DynamicVoxelPhantom:
        """Return a resized copy of this :class:`DynamicVoxelPhantom` instance.

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
        DynamicVoxelPhantom
            A new :class:`DynamicVoxelPhantom` instance containing resized tensors.
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

        return DynamicVoxelPhantom(
            resample(self.PD),
            torch.stack([resample(T1_rep) for T1_rep in self.T1]),
            torch.stack([resample(T2_rep) for T2_rep in self.T2]),
            torch.stack([resample(T2dash_rep) for T2dash_rep in self.T2dash]),
            torch.stack([resample(D_rep) for D_rep in self.D]),
            torch.stack([resample(B0_rep) for B0_rep in self.B0]),
            resample_multicoil(self.B1),
            resample_multicoil(self.coil_sens),
            self.size.clone(),
            tissue_masks=resample_masks(self.tissue_masks),
            time_points=self.time_points,
        )
        
    def save(self, file_name: str) -> None:
        """Save the phantom to a npz file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the phantom to.
        """
        if not file_name.endswith(".npz"):
            file_name += ".npz"
        np.savez(
            file_name,
            PD_map=self.PD.cpu().numpy(),
            T1_map=self.T1.cpu().numpy(),
            T2_map=self.T2.cpu().numpy(),
            T2dash_map=self.T2dash.cpu().numpy(),
            D_map=self.D.cpu().numpy(),
            B0_map=self.B0.cpu().numpy(),
            B1_map=self.B1.cpu().numpy(),
            coil_sens=self.coil_sens.cpu().numpy(),
            FOV=self.size.cpu().numpy(),
            time_points=self.time_points.cpu().numpy(),
            **{f"tissue_{key}": mask.cpu().numpy() for key, mask in self.tissue_masks.items()}
        )
