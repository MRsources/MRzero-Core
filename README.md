# MR Zero Core
```python
import MRZeroCore as mr0
```

Documentation: https://mrzero-core.readthedocs.io/

MRzero Core contains all essential parts of MRzero that are (close to) finalized.
Over time, more and more functionality should be pushed from MRzero to the Core,
while experimental scripts and WIP functionality stays in the MRzero git.
Everything contained in the Core should be documented, have a stable API and,
in the near future, be tested.

# Building and uploading
This is for Windows as host operating system. Requires docker to be installed
for manylinux compilation. Crosscompilation from Linux for Windows is currently
not setup, but theoretically possible. https://www.maturin.rs/distribution.html

## ...for Windows
```
maturin build --interpreter python
docker run --rm -v D:/repos/MRzero-Core:/io ghcr.io/pyo3/maturin build

maturin upload target/wheels/MRzeroCore-{ version }-cp37-abi3-win_amd64.whl target/wheels/MRzeroCore-{ version }-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -u <pypi-user> -p <pypi-pwd>
```

To build the documentation, run
```
jupyter-book build documentation/
```
in the root folder of this project. This requires jupyter-book, as well as MRzeroCore itself to be installed.


# CHANGELOG

We really should start tracking all the changes...

# CONTENTS

The MRzero Core contains the following, which can be imported with e.g.:
```python
from MRzeroCore.phantom import VoxelGridPhantom
```

# TODO:
Mention pTx support for pulseq / simulation

> NOTE:
>
> This list is currently WIP and does not reflect the actual state of MRzero Core.
> Before publishing v1.0 of MRzero Core, it should be changed to reflect what is
> currently contained and what is still TODO.

### MRzeroCore
- phantom
    - CustomVoxelPhantom
    - VoxelGridPhantom
    - SimData
- pulseq
    - Pulseq Interpreter
    - Pulseq Sequence Exporter
- reconstruction
    - Adjoint
    - FFT
    - NUFFT (requires torchkbnufft dependency)
    - Grappa
- sequence
    - Tools for sequence design and visualisation
    - Templates for simple GRE, TSE, ... sequences
- simulation
    - Prepass
    - PDG
    - Isochromats
