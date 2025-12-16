# NIfTI Phantom Specification

The "NIfTI phantom specification" describes the storage of data suitable for MR *imaging* simulations.
These phantoms consits of one or more NIfTI files for the data, and a single JSON file for the definition of the phantom, which references the NIfTI files.

## 1. Storage

The exact naming of files is a *convention*: Implementations can but are not required to reject phantoms which do not follow the naming described here. The only officially supported naming and storing convention is the following:

```
📂 subj42
├ 📄 subj42.nii.gz
├ 📄 subj42_dB0.nii.gz
├ 📄 subj42_B1+.nii.gz
├ ...
├ 📄 subj42.json
└ 📄 subj42-7T.json
```

In the example above, the name of the phantom is `subj42`. There is one base definition, stored in `subj42.json`, as well as a modified 7T version. The contents are described in [Section 4](#4-json-phantom-file).

> [!note]
> Every phantom should have a base definition. Its name is the same as the stem of the other files and the same as the folder name. Some tools might auto-load the base .json file or compare modified versions to it. They could error if it is missing.

Per-voxel data is stored in files named `<base>_<property>.nii(.gz)` files. A list of supported properties is described in [Section 2](#2-tissue-properties). The content of the NIfTI files is described in [Section 3](#3-nifti-data). The `density` property is the most important map, required for every tissue to be loaded from a NIfTI (opposed to being set as constant value). Its file should have no postfix (no `_density` or `_PD` at the end of the file name).

> [!note]
> In theory, the phantom definition in a .json file can reference any file name, making the property postfix of the naming optional. It is still adviced to follow this convention, as some tools might rely on correct labelling of the data.


## 2. Tissue Properties

The following properties are currently defined by the NIfTI phantom specification. If stored in a NIfTI file, the file postfix should match the key of the property.

| Key     | Property                       | Unit         | Default value |
| ------- | ------------------------------ | ------------ | ------------- |
| density | proton density                 | a.u.         | 1             |
| T1      | T1 relaxation                  | s            | inf.          |
| T2      | T2 relaxation                  | s            | inf.          |
| T2'     | T2' dephasing                  | s            | inf.          |
| ADC     | apparent diffusion coefficient | 10⁻³ mm² / s | 0             |
| dB0     | B0 frequency offset            | Hz           | 0             |
| B1+     | B1 transmit field              | rel. factor  | 1             |
| B1-     | B1 recieve / coil sensitivity  | rel. factor  | 1             |

## 3. NIfTI data

Per-voxel tissue properties are stored in `.nii` files following the [NIfTI v1.1](https://nifti.nimh.nih.gov/nifti-1/) specification. These files can be optionally compressed with gzip (extension `.nii.gz`).
- Each file contains a list of all tissues but only a single property
- The files contain 4-dimensional data (they must be 4D, with singelton dimensions if neccesary)
  - The first 3 dimension are spatial (must be size 1 for non-3D data)
  - Tissues are stored along the 4th dimension
- All NIfTI files must have the same resolution and orientation
  - Some tools might be less restrictive, but you cannot rely on this. This requirement might be lifted if the official loader gains support for reslicing. With this, all tools using it would gain support for phantoms with non-aligned tissues.
- If possible, the first 3 dimensions of the *data* stored in the NIfTI files should follow the RAS+ convention
  - index 0: R, 1: A, 2: S, indices grow towards positive
  - This ensures correct orientation for tools ignoring the affine matrix
- Affine matrix must transform data into RAS+, using mm as units (as per NIfTI spec)

## 4. JSON phantom file

The phantom is defined in a [`.json` file](https://www.json.org/json-en.html), which has to be referenced while loading it. This file can in turn reference the `.nii(.gz)` files, which were described above. The contents of a phantom `.json` file look like the following:

```json
{
  "file_type": "nifti_phantom_v1",
  "tissues": {
    "gm": {
      "density": "subj42.nii.gz:0",
      "T1": 1.56,
      "T2": 0.083,
      "T2'": 0.32,
      "ADC": 0.83,
      "dB0": "subj42_dB0.nii.gz:0",
      "B1+": [
        "subj42_B1+.nii.gz:0",
        "subj42_B1+.nii.gz:1",
        "subj42_B1+.nii.gz:2",
        "subj42_B1+.nii.gz:3",
        "subj42_B1+.nii.gz:4",
        "subj42_B1+.nii.gz:5",
        "subj42_B1+.nii.gz:6",
        "subj42_B1+.nii.gz:7"
      ]
    },
    "wm": {
      "density": "subj42.nii.gz:1",
      "T1": 0.83,
      "dB0": "subj42_dB0.nii.gz:0",
    },
    "fat": {
      "density": "subj42.nii.gz:4",
      "dB0": {
        "file": "subj42_dB0.nii.gz:0",
        "func": "x - 420"
      },
    }
  }
}
```

- `file_type`: for future revisions. Must be `"nifti_phantom_v1"`.
- `tissues`: keys are arbitrary strings; can be used to identify tissues.
- `T1`, `T2`, `T2dash`, `ADC`, `density`, `dB0`, `B1+`, `B1-`: [tissue properties](#tissue-properties)

> [!important]
> Each tissue *must* have a shape, which means that the `density` property is *required* and must by of type `file_ref`. All other properties are optional.

> [!tip]
> `B1+` and `B1-` are *lists* of maps to allow multi-channel data. Are allowed to contain only one map.

### tissue properties

The list of available properties is given in [Section 2](#2-tissue-properties). If a property is omitted, its default value is used. To overwrite its value, the following options are available:

| method     | type   | example                                           |
| ---------- | ------ | ------------------------------------------------- |
| `constant` | number | `1.56`,  `0.4e-3`                                 |
| `file_ref` | string | `"subj42_B0.nii.gz:0"`, `"felix-bloch_PD.nii:46"` |
| `mapping`  | object | `{"file": <file_ref>, "func": <mapping_func> }`   |

The `mapping` method exists to allow simple modifications on load like increasing inhomogeneity, applying a chemical shift or to map from 3T to 7T data. It consists of a `file_ref` and a `mapping_func`.

#### `file_ref`

A _file reference_ consists of `<file_name>[<index>]`. The `file_name` must be the name of a `.nii(.gz)` file that is in the same directory as the `.json` phantom file. The index specifies the index in the 4th dimension of the NIfTI file. It must be valid for the chosen file, otherwise the import can fail. This allows to store e.g. the T1 maps of multiple tissues in a single NIfTI file and to select the right one by its index.

#### `mapping_func`

A mapping function is an equation, evaluated once per voxel of the loaded data. It currently supports basic arithmetic (`+ - * / ( )`) and can use the following variables:

| variable | description                                    |
| -------- | ---------------------------------------------- |
| `x`      | value of the current voxel (loaded from NIfTI) |
| `x_min`  | minimum value of the loaded map                |
| `x_max`  | maximum value of the loaded map                |
| `x_mean` | mean value of the loaded map                   |
| `x_std`  | standard deviation of the loaded map           |

*Examples:*
| func                                | map | use case                       |
| ----------------------------------- | --- | ------------------------------ |
| `x - 420`                           | B0  | apply chemical shift to the B0 |
| `(x - x_min) / (x_max - x_min)`     | PD  | normalize the proton density   |
| `(x - x_mean) * 3 / x_std + x_mean` | B1  | set B1 standard deviation to 3 |
