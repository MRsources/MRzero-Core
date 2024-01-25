(phantom)=
```{eval-rst}
.. currentmodule:: MRzeroCore
```

# Phantom

Phantoms based on BrainWeb data can be downloaded and generated easily, as explained [here](generating_phantoms).
The data required by the simulation is stored in the {class}`SimData` class. It holds all the necessary maps as sparse tensors of voxels, the voxel position is stored in one of those tensors. {class}`SimData` should not be created directly, but rather by using one of the phantoms. Currently, there are two phantom classes available:

::::{grid}
:gutter: 3

:::{grid-item-card} [VoxelGridPhantom](voxel_grid_phantom)
A phantom described by a uniform cartesian grid of voxels.
:::

:::{grid-item-card} [CustomVoxelPhantom](custom_voxel_phantom)
A user specified list of voxel with custom, but uniform size and shape.
:::

:::{grid-item-card} [SimData](sim_data)
Simulation data used by the simulation. 
:::
::::

(load_brainweb)=
Example for loading a BrainWeb phantom, scaling it and using it as {class}`SimData` on the GPU:

```python
phantom = mr0.VoxelGridPhantom.brainweb("subject05.npz")
phantom = phantom.interpolate(128, 128, 32).slices([16])
data = phantom.build().cuda()
```

When building {class}`SimData`, a voxel shape can be selected. Following options are available:

| Shape          | Description |
| -------------- |-------------|
| `"exact_sinc"` | An sinc shaped voxel with a rect function as k-space drop off. |
| `"sinc"`       | The hard edge of the `"exact_sinc"` in k-space can be problematic for optimization. The default `"sinc"` shape has uses a sigmoid to smoothly drop to zero between the Nyquist frequency and the next k-space sample |
| `"box"`        | A box shaped voxel, with a sinc shaped k-space responce. When used for simulation, this responce will blur the image and at the same time introduce higher frequencies.
| `"gauss"`      | Normal distribution shaped voxels. Voxel size describes the variance. Only available for {class}`CustomVoxelPhantom`. |


(voxel_grid_phantom)=
## Voxel Grid Phantom

When converting to {class}`SimData`, voxels are sinc-shaped by default in order to correctly describe a bandwidth-limited signal: This phantom will only emit a signal up to the Nyquist frequency. Above of the Nyquist frequency, no signal will be emitted as these frequencies are not contained in the original input data as well.


```{eval-rst}
.. autoclass:: VoxelGridPhantom
    :members:
```

(custom_voxel_phantom)=
## Custom Voxel Phantom

Analytical phantom for experimentation. There is no resolution or grid, voxels can be placed anywhere and have any size or shape. Useful for testing reconstruction, sub-voxel positioning etc.

:::{note}
Because of how the voxel k-space responce is implemented currently, all voxels are limited to have the same shape and size. This limitation could be lifted in the future, if required.
:::

```{eval-rst}
.. autoclass:: CustomVoxelPhantom
    :members:
```

(sim_data)=
## Simulation Data

Simulation data by default is stored on the **CPU**. If simulation should be done using the **GPU**, transfer the data by using something like:

```python
data = data.cuda()
```

```{eval-rst}
.. autoclass:: SimData
    :members:
```
