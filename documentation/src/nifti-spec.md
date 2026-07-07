# NIfTI Phantom Specification

The "NIfTI phantom specification" describes the storage of data suitable for MR *imaging* simulations.
A phantom consists of one or more NIfTI files for per-voxel data and a single JSON file that defines the phantom and references the NIfTI files.

## 1. Storage

The naming of files is a *convention*: implementations can but are not required to reject non-conforming names. The supported convention is:

```
📂 subj42
├ 📄 subj42.nii.gz
├ 📄 subj42_dB0.nii.gz
├ 📄 subj42_B1+.nii.gz
├ ...
├ 📄 subj42-3T.json
└ 📄 subj42-7T.json
```

The phantom name is `subj42`, used as directory name and file prefix. It is available in two variants: `subj42-3T.json` and `subj42-7T.json`. Contents of the JSON file are described in [Section 4](#4-json-phantom-file).

Per-voxel data is stored in `<name>_<property>.nii(.gz)` files; the `density` map omits the property postfix. See [Section 2](#2-tissue-properties) for properties and [Section 3](#3-nifti-data) for the NIfTI format requirements. The `density` property is required for every tissue loaded from NIfTI (as opposed to a constant value).


## 2. Phantom Definition

### Units

The JSON file contains a `"units"` field that explicitly specifies the unit of each property. Currently, only the default units listed below are supported; automatic conversion may be added in the future.

| Key     | Default unit       | Description                               |
| ------- | ------------------ | ----------------------------------------- |
| gyro    | `MHz/T`            | gyromagnetic ratio                        |
| B0      | `T`                | main magnetic field strength              |
| T1      | `s`                | seconds                                   |
| T2      | `s`                | seconds                                   |
| T2'     | `s`                | seconds                                   |
| ADC     | `10^-3 mm^2/s`    | 10^-3 mm^2/s                              |
| dB0     | `Hz`               | frequency offset                          |
| B1+     | `rel`              | relative factor (dimensionless)           |
| B1-     | `rel`              | relative factor (dimensionless)           |

### System

The JSON file contains a `"system"` field that describes the physical system of the MRI experiment.

| Key  | Description                  | Default | Unit    |
| ---- | ---------------------------- | ------- | ------- |
| gyro | gyromagnetic ratio           | 42.5764 | MHz/T   |
| B0   | main magnetic field strength | 3.0     | T       |

The `gyro` value implicitly defines the nucleus: 42.5764 MHz/T corresponds to ^1H. Tissue properties (particularly relaxation times and dB0) are specific to the system they were measured at. A phantom defined at 3T should not be used in a 7T simulation without adjusting tissue values accordingly.

### Tissue Properties

The following properties are defined by the NIfTI phantom specification. If stored as per-voxel NIfTI data, the file postfix should match the key (e.g.: `subj42_dB0.nii.gz`).

| Key     | Property                       | Default |
| ------- | ------------------------------ | ------- |
| density | Proton density                 | *required* |
| T1      | T1 relaxation                  | inf     |
| T2      | T2 relaxation                  | inf     |
| T2'     | T2' dephasing                  | inf     |
| ADC     | Apparent Diffusion Coefficient | 0       |
| dB0     | B0 frequency offset            | 0       |
| B1+     | B1 transmit field              | [1]     |
| B1-     | B1 receive / coil sensitivity  | [1]     |


## 3. NIfTI Data

Per-voxel tissue properties are stored in `.nii` files following the [NIfTI v1.1](https://nifti.nimh.nih.gov/nifti-1/) specification, optionally gzip-compressed (`.nii.gz`).

- Each file contains a single property for all tissues
- Data must be 4-dimensional (use singleton dimensions for non-3D data)
  - Dimensions 1-3: spatial (size 1 if unused)
  - Dimension 4: tissue index
- All NIfTI files must share the same resolution and orientation
- Spatial data should follow the RAS+ convention (index 0: R, 1: A, 2: S, growing towards positive) to ensure correct orientation for tools ignoring the affine matrix
- The affine matrix must transform data into RAS+ using mm as units (as per NIfTI spec)


## 4. JSON Phantom File

The phantom is defined in a [`.json` file](https://www.json.org/json-en.html) with the following structure:

```json
{
  "file_type": "nifti_phantom_v1",
  "units": {
    "gyro": "MHz/T",
    "B0": "T",
    "T1": "s",
    "T2": "s",
    "T2'": "s",
    "ADC": "10^-3 mm^2/s",
    "dB0": "Hz",
    "B1+": "rel",
    "B1-": "rel"
  },
  "system": {
    "gyro": 42.5764,
    "B0": 3.0
  },
  "tissues": {
    "gm": {
      "density": "subj42.nii.gz[0]",
      "T1": 1.56,
      "T2": 0.083,
      "T2'": 0.32,
      "ADC": 0.83,
      "dB0": "subj42_dB0.nii.gz[0]",
      "B1+": [
        "subj42_B1+.nii.gz[0]",
        "subj42_B1+.nii.gz[1]",
        "subj42_B1+.nii.gz[2]",
        "subj42_B1+.nii.gz[3]",
        "subj42_B1+.nii.gz[4]",
        "subj42_B1+.nii.gz[5]",
        "subj42_B1+.nii.gz[6]",
        "subj42_B1+.nii.gz[7]"
      ]
    },
    "wm": {
      "density": "subj42.nii.gz[1]",
      "T1": 0.83,
      "dB0": "subj42_dB0.nii.gz[0]"
    },
    "fat": {
      "density": "subj42.nii.gz[4]",
      "dB0": {
        "file": "subj42_dB0.nii.gz[0]",
        "func": "x - 420"
      }
    }
  }
}
```

- `file_type`: must be `"nifti_phantom_v1"`.
- `units`: units for all properties (see [Units](#units)).
- `system`: physical system parameters (see [System](#system)).
- `tissues`: keys are arbitrary tissue names; values define [tissue properties](#tissue-properties).

Each tissue *must* have a `density` of type `file_ref` (it defines the tissue's shape). All other properties are optional and fall back to their [default values](#tissue-properties). `B1+` and `B1-` are *lists* of maps to support multi-channel data (a single-element list is valid).

### Tissue Property Values

Each property in a tissue can be set via one of three methods:

| Method     | Type   | Example                                             |
| ---------- | ------ | --------------------------------------------------- |
| `constant` | number | `1.56`, `0.4e-3`                                    |
| `file_ref` | string | `"subj42_B0.nii.gz[0]"`, `"felix-bloch_PD.nii[46]"` |
| `mapping`  | object | `{"file": <file_ref>, "func": <mapping_func>}`      |

#### `file_ref`

A file reference has the form `<file_name>[<index>]`. The file must be a `.nii(.gz)` in the same directory as the JSON file. The index selects a volume along the 4th dimension of the NIfTI file, allowing multiple tissues' data to share a single file.

#### `mapping_func`

A mapping function is an equation evaluated per voxel. It supports basic arithmetic (`+ - * / ( )`) and the following variables:

| Variable | Description                              |
| -------- | ---------------------------------------- |
| `x`      | value of the current voxel               |
| `x_min`  | minimum value of the loaded map          |
| `x_max`  | maximum value of the loaded map          |
| `x_mean` | mean value of the loaded map             |
| `x_std`  | standard deviation of the loaded map     |

*Examples:*

| func                                 | use case                       |
| ------------------------------------ | ------------------------------ |
| `x - 420`                            | apply chemical shift to dB0    |
| `(x - x_min) / (x_max - x_min)`     | normalize proton density       |
| `(x - x_mean) * 3 / x_std + x_mean` | set B1 standard deviation to 3 |
