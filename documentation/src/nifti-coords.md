# Coordinate System

- We always use RAS+ in a subject-aligned coordinate system
- NIfTI's can store two orientations at once and do not specify which one to use
- MITK uses a LPS+ coordinate system and negates the xy affine entries on loading
- The scanner says data is in the `SCANNER` coordinate system, but this changes with sequence settings.
- Phantom z direction should always point in $B_0$ direction

> [!note]
> We assume that in measurement and FOV, MRI sequences are aligned to the subject.
>
> When storing phantoms, we should always orient them to the subject-aligned RAS+ system (origin best at center of FOV but can be arbitrary).
> Correctly stored with `sform_code == 2` and `qform` unused (`qform_code == 0`), which is the default for `nibabel` but check for `simpleITK`!


## MR-zero

The MR-zero coordinate system is equivalent to NIfTI's RAS+ with voxel coordinates at their centers.
MR-zero (currently) ignores the affine matrix, except to extract the physical size of the NIfTI data.
It generates the voxel coordinates with the following code:
```python
pos_x, pos_y, pos_z = torch.meshgrid(
    size[0] * torch.fft.fftshift(torch.fft.fftfreq(shape[0])),
    size[1] * torch.fft.fftshift(torch.fft.fftfreq(shape[1])),
    size[2] * torch.fft.fftshift(torch.fft.fftfreq(shape[2])),
    indexing="ij"
)
```

There is no corner vs center here - voxels are _points_. The grid follows the typical FFT definition [`torch.fft.fftshift`](https://docs.pytorch.org/docs/stable/generated/torch.fft.fftfreq.html), which means they are placed as follows:
```python
# Even number of samples, e.g.: N = 32
pos = [-16, -15, ..., -1, 0, 1, ..., 15] / 32 * size
# Odd number of samples, e.g.: N = 21
pos = [-10, ..., -1, 0, 1, ..., 10] / 21 * size
```
This is matching typical FFT impls and k-space encodings: a plain FFT reco will return a pixel-perfect image without half-voxel shift.
(This also means that for even sample counts, the phantom extends are not exactly centered. On the other hand, there is always a voxel exactly at (0, 0, 0)).

---

# NIfTI coordinate system

The "true" specification of NIfTI seems to be the C header: [nifti1_h.pdf](https://afni.nimh.nih.gov/pub/dist/doc/nifti/nifti1_h.pdf) - search for section "3D IMAGE (VOLUME) ORIENTATION AND LOCATION IN SPACE".

The specification says that NIfTI's are in **RAS+**:
> the continuous coordinates are referred to as (x,y,z). The voxel index coordinates [...] are referred to as (i,j,k) \
> [...] \
> The (x,y,z) coordinates refer to the CENTER of a voxel. **In methods 2 and 3**, the (x,y,z) axes refer to a subject-based coordinate system, with _+x = Right +y = Anterior +z = Superior_. This is a right-handed coordinate system. \
> [...] \
> The i index varies most rapidly, j index next, k index slowest.

The 3 Methods this refers to are:

1. `qform_code == 0 && sform_code == 0`: compat for ANALYZE files, only supports scaling
2. `qform_code > 0`: use a [quaternion](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) for additional rotation
3. `sform_code > 0`: full affine matrix specifies rotation, scale, offset (and even skewing)

**Problem** \
NIfTI's allow `qform` and `sform` to co-exist and doesn't specify which one to use. It states:
> In this scheme, a dataset would originally be set up so that the Method 2 coordinates represent what the scanner reported. Later, a registration to some standard space can be computed and inserted in the header. Image display software can use either transform, depending on its purposes and needs.

`nibabel` defaults to use `sform` and in normal use never sets different transforms for `sform` and `qform`, expect when set manually.

## Mapping

The affine matrices map $[i,j,k] \mapsto (x, y, z)$. Even though the standard says that $(x, y, z)$ always refer to the subject-based RAS+ system, the origin of the coordinate system is specified twice, by the `qcode_form` and the `scode_form`:

| code | name | description |
| --- | --- | --- |
| 0 | `UNKNOWN` | no affine provided |
| 1 | `SCANNER` | scanner coordinates |
| 2 | `ALIGNED` | arbitrary coordinate center |
| 3 | `TALAIRACH` | [Talairach coordinates - Wikipedia](https://en.wikipedia.org/wiki/Talairach_coordinates) |
| 4 | `MNI_152` | from a database which [coregistered 152 brains](https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009)|

## Conclusion

NIfTI's written with `nibabel` always use (unless forced) `sform_code == 2`: affine matrix with arbitrary coordinates, since the scanner system is typically not known when creating phantoms from code. Similarly, when reading files, we usually have no way of transforming between scanner and subject coordinate systems, since this information is not stored in the NIfTI...

> [!warning]
> ...except if both `sform` and `qform` are used and one of them selects scanner coordinates and the other one subject coordinates. We should not use this in our generated phantoms, but have to take care when loading NIfTI files that use this property:
> 
> Different viewers could decide differently which of the both systems to use - double check when debugging a wrong orientation!
>
> This information is also found in the [NIfTI FAQ](https://nifti.nimh.nih.gov/nifti-1/documentation/faq.html) - Q17 and Q19
