# BrainWeb NIfTI phantoms

Phantoms are built from [BrainWeb](https://brainweb.bic.mni.mcgill.ca/) data and stored in the NIfTI format.
This data is not included directly.
Instead, a BrainWeb downloader is provided by `mr0`, which automatically downloads the necessary data and combines it with literature values for the physical properties at different field strenghts.
It is provided by [`mr0.generate_brainweb_phantoms](api-unsorted.md#generate_brainweb_phantoms):


```python
import MRzeroCore as mr0

# Stores the downloaded data in a `cache` folder in the given directory.
# All generated phantoms are emitted as a series of folders.
mr0.generate_brainweb_phantoms("output/brainweb")
```

These phantoms can be loaded as tissue dictionaries, where individual tissues can be modified before converting it into simulation data.
For more information look at the [NIfTI API documentation](api-nifti.md).

```python
tissues = mr0.TissueDict.load("output/brainweb/subj05/subj05-3T.json")
# you can modify individual tissues - its a dictionary of `VoxelGridPhantom`s:
tissues.pop("fat")
tissues["gm"].T1 *= 1.2

# either merge all tissues into one VoxelGridPhantom with `tissues.combine()`,
# which can be useful for plotting the maps or slightly faster simulation
tissues.combine().plot()
# ...or build simulation data directly to get partial volume effects:
sim_data = tissues.build()
```

# Custom phantoms for experimentation

Sometimes it is useful to generate phantoms consisting of individual voxels.
This can help with educational tasks, analysing effective resolution and PSFs and more.

For a single voxel:
```python
obj_p = mr0.CustomVoxelPhantom(
    pos=[[-0.25, -0.25, 0]],
    PD=[1.0],
    T1=[3.0],
    T2=[0.5],
    T2dash=[30e-3],
    D=[0.0],
    B0=0,
    voxel_size=0.1,
    voxel_shape="box"  # Can also be "sinc" or "gauss"
)
```

For a more complex structure try an L-shape:
```python
obj_p = mr0.CustomVoxelPhantom(
    pos=[[-0.4, -0.4, 0], [-0.4, -0.2, 0],
         [-0.3, -0.2, 0], [-0.2, -0.2, 0], [-0.1, -0.2, 0]],
    PD=[1.0, 1.0, 0.5, 0.5, 0.5],
    T1=[1.0, 0.5, 0.5, 0.5, 2],
    T2=0.1,
    T2dash=0.1,
    D=0.0,
    B0=0,
    voxel_size=0.1,
    voxel_shape="box"
)
```

# Predefined phantoms

Older versions of MR-zero Core stored phantoms as MATLAB .mat files or as NumPy .npz.
These are deprecated in favour of the new NIfTI phantom standard, but still available online.

## A simple 2D brain phantom as MATLAB .mat file

This phantom comes from a quantification measurement.
It is 2D, 1Tx (single channel B1+) and contains the following data:
- [x] PD (proton density)
- [x] T1 relaxation
- [x] T2 relaxation
- [ ] T2' dephasing
- [ ] ADC (apparent diffusion coefficient)
- [x] dB0 inhomogeneities
- [x] B1+ transmit field
- [ ] B1- recieve field

The phantom is stored on GitHub: [numerical_brain_cropped.mat](https://github.com/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/numerical_brain_cropped.mat).
Load it with MR-zero:

```python
# Load file with a shell command - works in interactive notebooks (e.g.: on Google Colab):
!wget https://github.com/MRsources/MRzero-Core/raw/main/documentation/playground_mr0/numerical_brain_cropped.mat &> /dev/null

# After downloading, load with MR-zero
import MRzeroCore as mr0
obj_p = mr0.VoxelGridPhantom.load_mat('numerical_brain_cropped.mat')
# You can modify the phantom first:
# change resolution, remove dB0 inhomogeneities, increase diffusion...
brain_phantom_res = 64
obj_p = obj_p.interpolate(brain_phantom_res, brain_phantom_res, 1)
obj_p.B0[:] = 0
obj_p.D *= 1.5
# Convert the phantom into sparse SimData
obj_p = obj_p.build()
```


## Predefined 3T BrainWeb phantom

The old, pre-generated phantoms from BrainWeb data are replaced by the new [NIfTI phantoms](#brainweb-nifti-phantoms).
There is still one available for backwards compatibility, stored on GitHub: [subject05.npz](https://github.com/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/subject05.npz)
It contains all maps except for recieve fields:
- [x] PD (proton density)
- [x] T1 relaxation
- [x] T2 relaxation
- [x] T2' dephasing
- [x] ADC (apparent diffusion coefficient)
- [x] dB0 inhomogeneities (*generated empirically*)
- [x] B1+ transmit field (*generated empirically*)
- [ ] B1- recieve field

```python
# Load file with a shell command - works in interactive notebooks (e.g.: on Google Colab):
!wget https://github.com/MRsources/MRzero-Core/raw/main/documentation/playground_mr0/subject05.npz &> /dev/null

sz = [64, 64]
# (i) load a phantom object from file
obj_p = mr0.VoxelGridPhantom.brainweb('subject05.npz')
obj_p = obj_p.interpolate(sz[0], sz[1], 32).slices([15])
obj_p.size[2]=0.08

# ========================================================
# EXAMPLE: insert rectangular "Tumor" in the diffusion map
# ========================================================

# typical brain tumor ADC values are around ~1.5 * 10^-3 mm^2/s,
# which lies between GM/WM and CSF (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3000221)
# mr0 uses D in units of 10^-3 * mm ^2/s  this is the same as µm^2/ms

# construct tumor border region
for ii in range(15, 25):
    for jj in range(15, 25):
        obj_p.D[ii, jj] = torch.tensor(0.75)

# construct tumor filling
for ii in range(16, 24):
    for jj in range(16, 24):
        obj_p.D[ii, jj] = torch.tensor(1.5)

# Convert the phantom into sparse SimData
obj_p = obj_p.build()
```