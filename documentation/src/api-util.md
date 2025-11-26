# util module

Additional helper functions.
```python
import MRzeroCore as mr0
mr0.util.<function>()
```

## `get_signal_from_real_system`

Wait for scanner TWIX files on the given path, and return if ready.
Useful if a shared network folder between scanner and research PC is available.

> [!NOTE]
> This is developed for internal use and might not be useful for others.

| Parameter | Description |
| --------- | ----------- |
| `path` | path to TWIX file |
| `NRep` | number of repetitions |
| `NRead` | number of ADC samples per repetition |
| `ncoils` | number of recieve channels |
| `heuristic_shift` | needed to remove garbage data from the file |


## `insert_signal_plot`

Insert a measured signal into a currently open pypulseq plot.
This is only supported by newer pulseq versions, which do not show (and close) the figure immediately if `plot_now=False`.

```python
seq.plot(plot_now=False)  # option doesn't exist for older pulseq versions
mr0.util.insert_signal_plot(seq, signal)
plt.show()
```

| Parameter | Description |
| --------- | ----------- |
| `seq` | pulseq sequence, needed to align the signal with the ADC samples |
| `signal` | signal to insert into the ADC plot |


## `pulseq_plot`

Wrapper around [`insert_signal_plot`](#insert_signal_plot), [`pulseq_plot_142`](#pulseq_plot_142) and [`pulseq_plot_pre14`](#pulseq_plot_pre14) which selects the appropriate plotting function based on the installed pulseq version.

| Parameter | Description |
| --------- | ----------- |
| `seq` | sequence object |
| `type` | _ignored_ |
| `time_range` | _same as for pulseq `plot()`_ |
| `time_disp` | _same as for pulseq `plot()`_ |
| `show_blocks` | _same as for pulseq `plot()`_ |
| `clear` | _ignored_ |
| `signal` | insert simulated signal into the ADC plot |
| `figid` | _ignored_ |


## `pulseq_plot_142`

Use only with **pypulseq versions 1.4.x** / tested with 1.4.2

Identical to pulseq `plot()`, *but returns the generated axes!*
This is needed to insert the signal into the ADC plot.
_Modified from pypulseq 1.4.2 [sequence.py - plot()](https://github.com/imr-framework/pypulseq/blob/ff4f01fe6c072a5dbe22af0556efb25e128da13b/pypulseq/Sequence/sequence.py#L921-L1145)_ - see pulseq documentation for details.

| Parameter | Description |
| --------- | ----------- |
| `seq` | sequence object |
| `label` | _same as for pulseq `plot()`_ |
| `show_blocks` | _same as for pulseq `plot()`_ |
| `save` | _same as for pulseq `plot()`_ |
| `time_range` | _same as for pulseq `plot()`_ |
| `time_disp` | _same as for pulseq `plot()`_ |
| `grad_disp` | _same as for pulseq `plot()`_ |
| `plot_now` | _same as for pulseq `plot()`_ |


## `pulseq_plot_pre14`

Use only with **pypulseq versions before 1.4**

Pulseq sequence plot with some aesthetical changes.
Most important, it can insert the simulated signal into the ADC plot.
_Modified from pypulseq 1.2.0post1 [sequence.py - plot()](https://github.com/imr-framework/pypulseq/blob/595351bc38ed0b57a359c4aa12d6b38c5b88f6d0/pypulseq/Sequence/sequence.py#L389-L465)_ - see pulseq documentation for details.

| Parameter | Description |
| --------- | ----------- |
| `seq` | sequence object |
| `type` | _same as for pulseq `plot()`_ |
| `time_range` | _same as for pulseq `plot()`_ |
| `time_disp` | _same as for pulseq `plot()`_ |
| `clear` | clear matplotlib figures before plotting |
| `signal` | insert simulated signal into the ADC plot |
| `figid` | use specific figure IDs (to reuse previously created figures) |


## `imshow`

Modified version of matplotlib `imshow()`, tailored for showing MR reconstructions.
`reco` is expected to be an array with dimensions `[x, y]`, `[x, y, z]` or `[coil, x, y, z]`
- plots images in RAS+
    - x is left-to-right
    - y is bottom-to-top
- plots 3d data (z) as a side-by-side grid of its slices
- applies quadratic coil-combination for multi-coil data


```python
# Replace
import matplotlib.pyplot as plt
plt.imshow(reco, *args, **kwargs)

# with
import MRzeroCore as mr0
mr0.util.imshow(reco, *args, **kwargs)
```


## `load_phantom`

Download a phantom from the MR-zero repository (or a custom URL) for getting started quickly.

```python
DEFAULT_PHANTOM_URL = "https://github.com/MRsources/MRzero-Core/raw/main/documentation/playground_mr0/numerical_brain_cropped.mat"
```

| Parameter | Description |
| --------- | ----------- |
| `size` | scale phantom if desired |
| `url` | where to load the phantom from, defaults to `DEFAULT_PHANTOM_URL` |
| `dB0_fB0` | `(offset, scaling)` floats to modify the \\(B_0\\) map |
| `dB1_fB1` | `(offset, scaling)` floats to modify the \\(B_1\\) map |
| `B0_polynomial` | optional 2d, 2nd degree polinomial to modify the \\(B_0\\) map |


## `simulate`

Helper function to simulate using [PDG](api-simulation.md#phase-distribution-graph-pdg-simulation).

| Parameter | Description |
| --------- | ----------- |
| `seq` | MR-zero sequence, pulseq sequence or file name |
| `phantom` | [`VoxelGridPhantom`](api-phantom.md#voxelgridphantom), [`CustomVoxelPhantom`](api-phantom.md#customvoxelphantom) or .npz / .mat file name |
| `sim_size` | scales phantom if set |
| `accuracy` | sets both `min_emitted_signal` and `min_latent_signal` for the PDG sim |
| `noise_level` | add normal distributed noise with the given std to the signal |


## `simulate_2d`

> [!CAUTION]
> Deprecated - use [`load_phantom`](#load_phantom) + [`simulate`](#simulate) instead.
> For parameter descriptions, look at these functions.
