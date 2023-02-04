from __future__ import annotations
import torch
import matplotlib.pyplot as plt
from . import sequence

# NOTE: better approach would be that functions detect the device from their
# arguments or provide an use_gpu arg, so we dont need this global setting

# NOTE: maybe make util a submodule too to hide imports

use_gpu = False
gpu_dev = 0


def get_device() -> torch.device:
    """Return the device as given by ``util.use_gpu`` and ``util.gpu_dev``."""
    if use_gpu:
        return torch.device(f"cuda:{gpu_dev}")
    else:
        return torch.device("cpu")


def set_device(x: torch.Tensor) -> torch.Tensor:
    """Set the device of the passed tensor as given by :func:`get_deivce`."""
    if use_gpu:
        return x.cuda(gpu_dev)
    else:
        return x.cpu()


# TODO: Remove this funciton and all references to it
import numpy as np
def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()



def plot_kspace_trajectory(seq: sequence.Sequence,
                           figsize: tuple[float, float] = (5, 5),
                           plotting_dims: str = 'xy',
                           plot_timeline: bool = True) -> None:
    """Plot the kspace trajectory produced by self.

    Parameters
    ----------
    kspace : list[Tensor]
        The kspace as produced by ``Sequence.get_full_kspace()``
    figsize : (float, float), optional
        The size of the plotted matplotlib figure.
    plotting_dims : string, optional
        String defining what is plotted on the x and y axis ('xy' 'zy' ...)
    plot_timeline : bool, optional
        Plot a second subfigure with the gradient components per-event.
    """
    assert len(plotting_dims) == 2
    assert plotting_dims[0] in ['x', 'y', 'z']
    assert plotting_dims[1] in ['x', 'y', 'z']
    dim_map = {'x': 0, 'y': 1, 'z': 2}

    # TODO: We could (optionally) plot which contrast a sample belongs to,
    # currently we only plot if it is measured or not

    kspace = seq.get_full_kspace()
    adc_mask = [rep.adc_usage > 0 for rep in seq]

    cmap = plt.get_cmap('rainbow')
    plt.figure(figsize=figsize)
    if plot_timeline:
        plt.subplot(211)
    for i, (rep_traj, mask) in enumerate(zip(kspace, adc_mask)):
        kx = to_numpy(rep_traj[:, dim_map[plotting_dims[0]]])
        ky = to_numpy(rep_traj[:, dim_map[plotting_dims[1]]])
        measured = to_numpy(mask)

        plt.plot(kx, ky, c=cmap(i / len(kspace)))
        plt.plot(kx[measured], ky[measured], 'r.')
        plt.plot(kx[~measured], ky[~measured], 'k.')
    plt.xlabel(f"$k_{plotting_dims[0]}$")
    plt.ylabel(f"$k_{plotting_dims[1]}$")
    plt.grid()

    if plot_timeline:
        plt.subplot(212)
        event = 0
        for i, rep_traj in enumerate(kspace):
            x = np.arange(event, event + rep_traj.shape[0], 1)
            event += rep_traj.shape[0]
            rep_traj = to_numpy(rep_traj)

            if i == 0:
                plt.plot(x, rep_traj[:, 0], c='r', label="$k_x$")
                plt.plot(x, rep_traj[:, 1], c='g', label="$k_y$")
                plt.plot(x, rep_traj[:, 2], c='b', label="$k_z$")
            else:
                plt.plot(x, rep_traj[:, 0], c='r', label="_")
                plt.plot(x, rep_traj[:, 1], c='g', label="_")
                plt.plot(x, rep_traj[:, 2], c='b', label="_")
        plt.xlabel("Event")
        plt.ylabel("Gradient Moment")
        plt.legend()
        plt.grid()

    plt.show()
