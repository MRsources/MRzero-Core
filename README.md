![PyPI - Version](https://img.shields.io/pypi/v/MRzeroCore)

# MRzero Core

The MRzero Core contains the core functionality of [MRzero](https://arxiv.org/abs/2002.04265) like MRI sequence building, simulation and reconstruction. MRzero Core does not force you to take any particular approach to e.g., reconstruction, as it targets easy integration in existing projects. Nevertheless, more tools can be added in the future if they helpful for the general application space.

> ### 🎉 1.0 release
> Version 1.0 adds Pulseq 1.5 support and flips the sign convention of the simulated signal, so that it is now the complex conjugate of what previous versions returned. Reconstruction accordingly switches from `fft` to `ifft`.
> 
> See the [FFT sign conventions](https://mrsources.github.io/MRzero-Core/fft_sign_conventions.html) page for details, and the [CHANGELOG](CHANGELOG.md) for the full list of changes. If anything else broke, please [open an issue](https://github.com/MRsources/MRzero-Core/issues).

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

Examples on how to use MR-zero can be found in the [Playground](https://mrzero-core.readthedocs.io/en/latest/playground_mr0/overview.html).

## Pulseq Integration

**MRzero Core makes Pulseq simulation incredibly easy** - simulate any .seq file in just one line:

```python
import MRzeroCore as mr0

# Simulate any Pulseq file
seq = mr0.Sequence.import_file("your_sequence.seq")
signal = mr0.util.simulate(seq)  # That's it!
```

### Key Features:
- **One-line simulation** of any Pulseq .seq file
- **PyPulseq integration** - write sequences in Python, simulate with MR-zero
- **Google Colab ready** - [13+ ready-to-run examples](https://mrzero-core.readthedocs.io/en/latest/playground_mr0/overview.html#code-and-simulate-pypulseq)
- **MATLAB ↔ Python workflow** - create in MATLAB Pulseq, simulate in Python
- **No dependencies** - fully self-contained Pulseq parser included
- **Scanner-ready export** - export MR-zero sequences as .seq files

Try it now: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_upload_seq.ipynb)

## Links

- Documentation: https://mrsources.github.io/MRzero-Core
- Examples: [Playground](https://mrsources.github.io/MRzero-Core/playground.html)
- PyPI: https://pypi.org/project/mrzerocore/
- Original MRzero Paper: https://arxiv.org/abs/2002.04265

## Building from source

This assumes windows as host operating system. For building the python wheel, you need:
- the Rust toolchain: [rustup](https://rustup.rs/)
- the rust-python build tool tool: [pip install maturin](https://github.com/PyO3/maturin)
- for Linux crosscompilation: [docker](https://www.docker.com/)
- to build the documentation: [mdBook](https://rust-lang.github.io/mdBook/guide/installation.html)

**Building for Windows**
```
maturin build --interpreter python
```
**Building for Linux**
```
docker run --rm -v <path-to-repo>/MRzero-Core:/io ghcr.io/pyo3/maturin build
```

**Building the documentation**

Install newest (beta) version of mdBook:
```bash
cargo install --git https://github.com/rust-lang/mdBook.git mdbook
```

Run a live-preview server:
```bash
# in the root directory:
mdbook serve documentation
```


## Official builds

The [python wheels](https://pypi.org/project/mrzerocore/) hosted by [PyPI](https://pypi.org/) is built as described above. It is deployed as GitHub action (see [pypi_publish.yml](/.github/workflows/pypi_publish.yml))
