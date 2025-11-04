> [!CAUTION]
> This document might be outdated. We definitely should include **NIfTI Phantoms** here

# Generating Phantoms

Phantoms are built from [BrainWeb](https://brainweb.bic.mni.mcgill.ca/) data. This data is not included directly.
Instead, a BrainWeb downloader is part of `mr0`, that can be run once to download all segmentation data provided by BrainWeb, which is then filled to produce files that can be loaded as mentioned [here](load_brainweb).

Phantoms can be generated with different configurations, depending on the use-case, like 3T or 7T data, high-res, or whether to include fat are options. A fixed set of configuartions facilitates reproducibility. To execute generation, just run the following code:

```python
import MRzeroCore as mr0

mr0.generate_brainweb_phantoms("output/brainweb", "3T")
```

# Loading predefined 2D or 3D phantoms
We have some phantoms predefined.

## A simple 2D brain phantom as .mat file, 3T (PD,T1,T2, dB0, rB1, no T2', no Diffusion)
```python
!wget https://github.com/MRsources/MRzero-Core/raw/main/documentation/playground_mr0/numerical_brain_cropped.mat &> /dev/null
```

This can be loaded and modified via mr0.VoxelGridPhantom.load_mat():
```python
obj_p = mr0.VoxelGridPhantom.load_mat('numerical_brain_cropped.mat')
brain_phantom_res = 64 #@param {type:"slider", min:16, max:128, step:16}
obj_p = obj_p.interpolate(brain_phantom_res, brain_phantom_res, 1)
obj_p.B0[:] = 0
# obj_p.D[:] = 0
obj_p = obj_p.build()
```

## A filled brainweb phantom 3T (PD,T1,T2, dB0, rB1, T2', Diffusion)

```python
!wget https://github.com/MRsources/MRzero-Core/raw/main/documentation/playground_mr0/subject05.npz &> /dev/null
```

This can be loaded and modified via mr0.VoxelGridPhantom.brainweb():
```python
sz = [64, 64]
# (i) load a phantom object from file
obj_p = mr0.VoxelGridPhantom.brainweb('subject05.npz')
obj_p = obj_p.interpolate(sz[0], sz[1], 32).slices([15])
obj_p.size[2]=0.08

if 1: # insert rectangular "Tumor" in the diffusion map
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

# Store PD and B0 for comparison
D = obj_p.D
B0 = obj_p.B0
D=obj_p.D
obj_p.plot()
# Convert Phantom into simulation data, this is sparse, and cannot be plotted directly, see obj_p.recover()
obj_p = obj_p.build()
```

## A simple voxel phantom via mr0.CustomVoxelPhantom():

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
    voxel_shape="box"
)
```
For a more complex structure, her an L-shape:
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


```{eval-rst}
.. autofunction:: MRzeroCore.phantom.brainweb.generate_brainweb_phantoms
```
