# Simulation Phantoms

Phantoms based on BrainWeb data can be downloaded and generated easily, as explained in [Generating Phantoms](phantoms.md).
The data required by the simulation defined by [`SimData`](#simdata).
It should not be created directly, but rather by defining or loading a [`VoxelGridPhantom`](#voxelgridphantom) or [`CustomVoxelPhantom`](#customvoxelphantom).

Example for loading a BrainWeb phantom, scaling it and using it as [`SimData`](#simdata) on the GPU:

```python
phantom = mr0.VoxelGridPhantom.load("subject05.npz")
phantom = phantom.interpolate(128, 128, 32).slices([16])
data = phantom.build().cuda()
```

> [!TIP]
> A new type of phantoms is coming!
> These are based on NIfTIs and allow partial volume effects, segmented phantoms and easy phantom reconfiguration without code changes.
> They should be documented here as soon as they are merged with MR-zero Core.

## Voxel Shape

Simulation data (and therefore all Phantoms) define a shape for their voxels.
The following options are available:

| Shape          | Description |
| -------------- |-------------|
| `"exact_sinc"` | Sinc shape = rect function as k-space drop off |
| `"sinc"`       | Sigmoid (rounded rect) in k-space for differentiability |
| `"box"`        | A box shaped voxel, with a sinc shaped k-space responce |
| `"gauss"`      | Normal distribution voxel shape, available for [`CustomVoxelPhantom`](#customvoxelphantom) |

*Most cases should use `"sinc"`:*\
Input maps, signals, and reconstructed images are all sampled as *points*.
Extending this bandwidth-limited data to continuous functions means that *sinc*-shaped voxels are the correct form in basically all cases.
Using box-shaped voxels means that a round trip of putting maps into the simulation and then quantifying them from the reconstructed images will inevitably blur them.
Box shaped voxels do *not* have flat signal response like sinc shaped voxels have (up to the nyquist frequency).
Point shaped voxels would be a good fit too, except that they would prohibit gradient spoiling (and are a bad description of actual tissue).

## `SimData`

Contains the data used for simulation (together with the [`Sequence`](api-sequence.md#sequence)). The values are flattened tensors, which only contain voxels with a non-zero proton density.

```python
# Usually not constructed directly but via the `.build()` function on a phantom
SimData(PD, T1, T2, T2dash, D, B0, B1, coil_sens, size, voxel_pos, nyquist, dephasing_func, recover_func=None, phantom_motion=None, voxel_motion=None, tissue_masks=None)
```

| Parameter | Description |
| --------- | ----------- |
| `PD` | proton density |
| `T1` | \\(T_1\\) relaxation time (seconds) |
| `T2` | \\(T_2\\) relaxation time (seconds) |
| `T2dash` | \\(T_2'\\) dephasing time (seconds) |
| `D` | isometric diffusion coefficient (10^-3 mm^2/s) |
| `B0` | \\(B_0\\) off-resonance (Hz) |
| `B1` | per-coil \\(B_1\\) inhomogeneity (normalized) |
| `coil_sens` | per-coil sensitivity (normalized) |
| `size` | physical phantom size (meters) |
| `voxel_pos` | voxel position tensor |
| `nyquist` | Nyquist frequency, given by phantom size and resolution |
| `dephasing_func` | k-space response of voxel shape |
| `recover_func` | optional - provided by phantoms to revert `.build()` function |
| `phantom_motion` | optional rigid phantom motion trajectory |
| `voxel_motion` | optional per-voxel motion trajectory |
| `tissue_masks` | optional dictionary of masks, useful for learning tasks |

- `SimData.cpu()`: move data to the CPU
- `SimData.cuda()`: move data to the GPU
- `SimData.device` (property): the device of the repetition data
- `SimData.recover()`: reconstruct the [`VoxelGridPhantom`](#voxelgridphantom) or [`CustomVoxelPhantom`](#customvoxelphantom) that was used to build this `SimData` 


## `VoxelGridPhantom`

Phantom format where tissue parameters are cartesian grids of voxels. The data is stored in 3D or 4D tensors with indices `[(channel), x, y, z]`.

```python
VoxelGridPhantom(PD, T1, T2, T2dash, D, B0, B1, coil_sens, size, phantom_motion=None, voxel_motion=None, tissue_masks=None)
```

Parameters of this function are identical to those of [`SimData`](#simdata), except that they are stored in 3D or 4D tensors.
They are indexed `[(channel), x, y, z]`.

- `VoxelGridPhantom.load(file_name)`: Loading phantom from a .npz file
- `VoxelGridPhantom.slices(slices)`: Removing all but the specified list of slices (in z direction)
- `VoxelGridPhantom.scale_fft()`: Resizing the phantom by doing FFT -> truncate -> FFt
- `VoxelGridPhantom.interpolate()`: Resizing the phantom by using `torch.nn.functional.interpolate(..., mode='trilinear')`
- `VoxelGridPhantom.plot()`: Plotting all maps of the phantom (center slice)
- `VoxelGridPhantom.build(PD_threshold)`: Convert this phantom into [`SimData`](#simdata), sparsifying the arrays (flattening and masking where `pd < PD_threshold`)

## `CustomVoxelPhantom`

Phantom format where tissue parameters are 1D lists of voxels with arbitrary positions.

```python
CustomVoxelPhantom(pos, PD=1.0, T1=1.5, T2=0.1, T2dash=0.05, D=1.0, B0=0.0, B1=1.0, voxel_size=0.1, voxel_shape="sinc")
```

> The `size` of this phantom is computed from the extends of the voxel positions.\
>  Useful for testing reconstruction, sub-voxel positioning, teaching and more.

Parameters of this function are identical to those of [`SimData`](#simdata), except that they can also be python lists (and are converted to tensors internally).
In addition, it contains a convenience method for plotting the phantom by rendering the voxels (and their selected shape) at any resolution.

- `CustomVoxelPhantom.generate_PD_map()`: Voxels can be sinc- box- or gauss-shaped and at arbitrary positions. This function renders them into a fixed-resolution proton density map.
- `CustomVoxelPhantom.plot()`: Render and plot all maps of this phantom
- `CustomVoxelPhantom.build()`: Convert into [`SimData`](#simdata) instance for simulation