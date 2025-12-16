# NIfTI phantoms

In previous versions, MR-zero used ad-hoc formats for its simulation phantoms.
It stored the right combination of maps in numpys compressed file format.
Starting 2026, we fully switch to a new file format with a precise specification.
It is a combination of the widely used NIfTI file format an a JSON configuration file.
Exact definitions of the contents of this configuration file makes writing loaders for different applications / programming languages easy.
At the same time, care was taken to allow great flexibility and human readability of the files.

You can see the specification [here](nifti-spec.md).
A closer look at the coordinate system used by MR-zero is shown [here](nifti-coords.md).


## How to load NIfTI Phantoms

> [!caution]
> This is taken from the addon readme and needs adjustment to the MR-zero version.
> Especially the separate config class, which can be used to modify the phantom before loading (or to alter and then write it) is completely missing here.

The loading functionality is currently part of the MR-zero `addon`.
NIfTI phantoms consist of one or more tissues.
These are loaded as a special python dictionary of MR-zero `VoxelGridPhantom`s.
This allows to modify the individual tissues easily.
The `PhantomDict` provides methods to interpolate the resolution of all tissues or to convert the phantom to simulation data.
If you think more methods should be added, please open an issue or a pull request.

```python
from addon.data.phantoms_v2 import PhantomDict
phantom = PhantomDict.load("brainweb-subj42/brainweb-subj42.json")
phantom = phantom.interpolate(181, 217, 64).slices([30])

# Modify the phantom
if NO_FAT:
    phantom.pop("fat")
if NO_FAT_SHIFT and not NO_FAT:
    phantom.B0 -= phantom.B0.mean()

# generate simulation data with partial volume effects
data = phantom.build()
# generate without, differences in overlapping tissues only
data = phantom.combine().build()
# combining returns a VoxelGridPhantom, useful for e.g.: plotting
phantom.combine().plot()
```

A couple of points can be seen in the example above:
- Loading the data is done by pointing the `PhantomDict` to the correct `.json` file
- The `PhantomDict` has methods to interpolate the whole phantom, which acts on all tissues
- The phantom is still a simple python dictionary, which allows to easily remove tissues
- There are two methods of generating `SimData`:
  1. Calling `build()` directly will generate simulation data where some voxels can appear multiple times if tissues are overlapping.
  2. The `.combine()` method merges all tissues into a single `VoxelGridPhantom` by weighting the tissue properties by their proton densities. This is identical to the old phantoms without partial volume effects and can result in smaller simulation data but will not produce the more realistic partial volume effects. The combined `VoxelGridPhantom` can be used for plotting and other methods which are not available on the `PhantomDict`.
