# NIfTI phantoms

In previous versions, MR-zero used ad-hoc formats for its simulation phantoms.
It stored the right combination of maps in numpys compressed file format.
Starting 2026, we fully switch to a new file format with a precise specification.
It is a combination of the widely used NIfTI file format an a JSON configuration file.
Exact definitions of the contents of this configuration file makes writing loaders for different applications / programming languages easy.
At the same time, care was taken to allow great flexibility and human readability of the files.

You can see the specification [here](nifti-spec.md).
A closer look at the coordinate system is shown [here](nifti-coords.md).

To see how to use phantoms, you can either [look at examples](nifti-generate.md) or [view the API documentation](api-nifti.md).
