[![Documentation Status](https://readthedocs.org/projects/mrzero-core/badge/?version=latest)](https://mrzero-core.readthedocs.io/en/latest/?badge=latest)

# MRzero Core

The MRzero Core contains the core functionality of [MRzero](https://arxiv.org/abs/2002.04265) like MRI sequence building, simulation and reconstruction. MRzero Core does not force you to take any particular approach to e.g., reconstruction, as it targets easy integration in existing projects. Nevertheless, more tools can be added in the future if they helpful for the general application space.

## Usage

MRzero Core is written in [Python](https://www.python.org/), heavily relying on [PyTorch](https://pytorch.org/) for fast (GPU-) Tensor calculations.
To improve performance, parts of the simulation are written in [Rust](https://www.rust-lang.org/) and compiled for x86 Windows and Linux, other platforms are currently not supported.

Install with pip:
```
pip install MRzeroCore
```

The typical way of using it is like the following:
```python
import MRzeroCore as mr0
```

Examples on how to use can be found in the [Playground](https://mrzero-core.readthedocs.io/en/latest/playground_mr0/overview.html).

## Links

- Documentation: https://mrzero-core.readthedocs.io/
- Examples: [Playground](https://mrzero-core.readthedocs.io/en/latest/playground_mr0/overview.html)
- PyPI: https://pypi.org/project/mrzerocore/
- Original MRzero Paper: https://arxiv.org/abs/2002.04265

## Building from source

This assumes windows as host operating system. For building the python wheel, you need:
- the Rust toolchain: [rustup](https://rustup.rs/)
- the rust-python build tool tool: [pip install maturin](https://github.com/PyO3/maturin)
- for Linux crosscompilation: [docker](https://www.docker.com/)
- to build the documentation: [pip install jupyter-book](https://jupyterbook.org/en/stable/intro.html)

**Building for Windows**
```
maturin build --interpreter python
```
**Building for Linux**
```
docker run --rm -v <path-to-repo>/MRzero-Core:/io ghcr.io/pyo3/maturin build
```

To **build the documentation**, run
```
jupyter-book build documentation/
```
in the root folder of this project. This requires jupyter-book as well as MRzeroCore itself to be installed.


## Official builds

The [python wheels](https://pypi.org/project/mrzerocore/) hosted by [PyPI](https://pypi.org/) is built as described above, and uploaded as following:

```
maturin upload target/wheels/MRzeroCore-{ version }-cp37-abi3-win_amd64.whl target/wheels/MRzeroCore-{ version }-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -u <pypi-user> -p <pypi-pwd>
```

The [documentation](https://mrzero-core.readthedocs.io/en/latest/intro.html) is built using [readthedocs](https://readthedocs.org/), which works the same as described above.
