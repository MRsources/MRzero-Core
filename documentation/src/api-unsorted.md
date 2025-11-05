# Other methods

Miscellaneous methods provided by MR-zero.
These are currently exported directly, but might be sorted into sub-modules if the list grows.


## `generate_brainweb_phantoms`

Builds MR-zero phantoms with the data provided by [BrainWeb](http://www.bic.mni.mcgill.ca/brainweb/).
These phantoms are not stored in the repository; call this function once to generate them locally.

The configuration files with the used tissue parameters can be found here: [brainweb_data.json](https://github.com/MRsources/MRzero-Core/blob/main/python/MRzeroCore/phantom/brainweb/brainweb_data.json)

| Parameter | Description |
| --------- | ----------- |
| `outpub_dir` | path to the folder where the generated phantoms (and a cache folder for downloaded data) are written to |
| `config` | one of `"3T"`, `"7T-noise"`, or `"3T-highres-fat"` |


## `sig_to_mrd`

Write the simulation output to an [ISMRMD](https://github.com/ismrmrd/ismrmrd) file; supported by many reconstruction tools.

| Parameter | Description |
| --------- | ----------- |
| `mrd_path` | path where the file will be written to |
| `mr0_signal` | signal to write, shape `[samples, channels]` |
| `seq` | pulseq sequence object (for kspace trajectory, labels, definitions) |
| `verbose` | logging verbosity (0 to 5) |


## `pulseq_write_cartesian`

> [!WARNING]
> Outdated - we suggest writing sequences with Pulseq(-zero) instead.
> Exporters might be removed in the future.

Export an MR-zero sequence to Pulseq.
Since MR-zero sequences have no limitations in timing, slew rates, amplitudes and more, this exporter has to try to convert this into a sequence that can actually be measured.
The exporter only supports cartesian trajectories due to its use of trapezoidal gradients.
There is no guarantee that it produces the expected results.


| Parameter | Description |
| --------- | ----------- |
| `seq_param` | the `mr0` sequence object to export |
| `path` | path to write the .seq file to |
| `FOV` | field of view, used for slice selection to scale gradients and |
| `plot_seq` | wether to plot the generated pulseq sequence |
| `num_slices` | used in combination with `FOV` to determine slice thickness |
| ~`write_data`~ | *ignored* |