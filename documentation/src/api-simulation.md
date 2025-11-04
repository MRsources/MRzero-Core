# Simulation

MR-zero provides two simulations:
- a purposefully simple isochromat based simulation to check for correctness
- a state-of-the-art PDG simulation, built for speed and accuracy.

Both simulations are differentiable and capable to run on any device supported by PyTorch (all CPUs and NVIDIA GPUs).
While the isochromat simulation is rather slow and only there for confirmation purposes, PDG is one of the fastest currently available simulations.
You can find more information about it in the [Paper](https://doi.org/10.1002/mrm.30055).

## Phase Distribution Graph simulation



### `compute_graph`
### `compute_graph_ext`
### `execute_graph`
### `Graph`

## `isochromat_sim`

Isochromat based Bloch simulation.
The code is implemented for readability, not speed.
This allows to verify its correctness, bugs are not expected.

> [!WARNING]
> This simulation
> - is slow
> - does not support diffusion
> - always uses `"box"` - voxels, which is usually a bad choice and will introduce slight blurring. See [Voxel Shape](api-phantom.md#voxel-shape) for more information. This choice was made for simplicity.

| Parameter | Description |
| --------- | ----------- |
| `seq` | the [`Sequence`](api-sequence.md#sequence) to simulate |
| `data` | the [`SimData`](api-phantom.md#simdata) to simulate |
| `spin_count` | number of spins per voxel |
| `perfect_spoiling=False` | set transversal magnetization to zero before excitation pulses |
| `print_progress=True` | print the currently simulated repetition |
| `spin_dist="rand"` | `"rand"`: random spin distribution; `"r2"`: blue noise quasirandom sequence - [link](https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/) |
| `r2_seed=None` | seed for `"r2"` distribution (for reproducibility); random if `None` |
