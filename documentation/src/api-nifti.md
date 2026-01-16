# NIfTI Phantoms

The classes shown here reflect the content of the `.json` config file structure.
These files define how various NIfTIs are combined into a phantom.
You can load a complete phantom directly with [`mr0.TissueDict.load`](api-phantom.md#tissuedict).
*But you can also* load only the configuration with [`mr0.NiftiPhantom.load`](#niftiphantom).
This allows you to modify it as needed before loading the phantom or saving the changes.

All classes are [dataclasses](https://docs.python.org/3/library/dataclasses.html), augmented with additional functions.

## `PhantomUnits`

Stores the physical units used by the phantom.
These are currently fixed to the values used by MR-zero (default values) and are stored for documentation (no conversion is implemented).
However, different units and automatic converstion might be implemented in the future.

```python
PhantomUnits(gyro, B0, T1, T2, T2dash, ADC, dB0, B1_tx, B1_rx)
```

| Parameter | Description | Default unit |
| --------- | ----------- | ------------ |
| `gyro` | Gyromagnetic ratio | `MHz / T` |
| `B0` | Strength of the main magnetic field | `T` |
| `T1` | Exponential T1 relaxation | `s` |
| `T2` | Exponential T2 relaxation | `s` |
| `T2dash` | Exponential T2 dephasing | `s` |
| `ADC` | Apparent Diffusion Coefficient | `10^-3 mm^2/s` |
| `dB0` | Offresonance through `B0` fluctuations | `Hz` |
| `B1_tx` | Fluctuations in B1+ (transmit) field | `rel` |
| `B1_rx` | Fluctuations in B1- (recieve) field | `rel` |

- `PhantomUnits.default()`: return `PhantomUnits` with the default units given above
- `PhantomUnits.from_dict(config)`: load from a dictionary and check if units are valid
- `PhantomUnits.to_dict()`: convert to a dictionary

## `PhantomSystem`

Describes the physical properties of the MRI experiment.

```python
PhantomSystem(gyro, B0)
```

| Parameter | Description |
| --------- | ----------- |
| `gyro` | Gyromagnetic ratio, given in units of `PhantomUnits.gyro` |
| `B0` | Strength of the main magnetic field, given in units of `PhantomUnits.B0` |

- `PhantomSystem.from_dict(config)`: load from a dictionary
- `PhantomSystem.to_dict()`: convert to a dictionary

## `NiftiRef`

Reference to a NIfTI file, including the index to the tissue.
NIfTIs can contain multiple 3D volumes, stacked in the 4th dimension.
For a description look at the [specification](nifti-spec.md#tissue-properties).

```python
NiftiRef(file_name, tissue_index)
```

| Parameter | Description |
| --------- | ----------- |
| `file_name` | [Path](https://docs.python.org/3/library/pathlib.html#concrete-paths) to the `.nii(.gz)` file |
| `tissue_index` | Index along the 4th dimension of the specified NIfTI file |

As given by the spec, `NiftiRef`s are stored as `"path/to/nifti.nii.gz[<index>]"` strings:
- `NiftiRef.parse(config)`: parse from a string
- `NiftiRef.to_str()`: convert to a string

## `NiftiMapping`

Combination of `NiftiRef` and a mapping function for simple modifications.
For a description look at the [specification](nifti-spec.md#tissue-properties).

```python
NiftiMapping(file, func)
```

| Parameter | Description |
| --------- | ----------- |
| `file` | A [`NiftiRef`](#niftiref) |
| `func` | A mapping function as `str` - see [spec](nifti-spec.md#mapping_func) |

- `NiftiRef.parse(config)`: parse from a dictionary (calling `NiftiRef.parse`)
- `NiftiRef.to_str()`: convert to a dictionary

## `NiftiTissue`

Definition of a single tissue, given by a `float`, [`NiftiRef`](#niftiref) or [`NiftiMapping`](#niftimapping) for every property.
Units are given by [`PhantomUnits`](#phantomunits).

```python
NiftiTissue(density, T1, T2, T2dash, ADC, dB0, B1_tx, B1_rx)
```

| Parameter | Description | Default value |
| --------- | ----------- | ------------- |
| `density` | Proton density - arbitrary units (convention: 0-1 range) | *must be specified* |
| `T1` | Exponential T1 relaxation | `inf` |
| `T2` | Exponential T2 relaxation | `inf` |
| `T2dash` | Exponential T2 dephasing | `inf` |
| `ADC` | Apparent Diffusion Coefficient | `0.0` |
| `dB0` | Offresonance through `B0` fluctuations | `1.0` |
| `B1_tx` | Fluctuations in B1+ (transmit) field | `1.0` |
| `B1_rx` | Fluctuations in B1- (recieve) field | `1.0` |

- `NiftiTissue.default()`: return a `NiftiTissue` with the default units given above - density must be passed
- `NiftiTissue.from_dict(config)`: load from a dictionary
- `NiftiTissue.to_dict()`: convert to a dictionary

## `NiftiPhantom`

Represents the configuration given by a phantom `.json` file.

```python
NiftiPhantom(file_type, units, system, tissues)
```

| Parameter | Description |
| --------- | ----------- |
| `file_type` | Specifies the version of the specification used by the config file. Currently *must* be `"nifti_phantom_v1"` |
| `units` | [`PhantomUnits`](#phantomunits) object |
| `system` | [`PhantomSystem`](#phantomsystem) object |
| `tissues` | Dictionary of tissues; key specifies name, values are [`NiftiTissue`](#niftitissue) objects |

- `NiftiPhantom.default(gyro=42.5764, B0=3)`: return a `NiftiPhantom` without tissues
- `NiftiPhantom.load(path)`: load from `.json` file, given by the path
- `NiftiPhantom.save(path)`: save to a `.json` file, given by the path
- `NiftiPhantom.from_dict(config)`: load from a dictionary
- `NiftiPhantom.to_dict()`: convert to a dictionary