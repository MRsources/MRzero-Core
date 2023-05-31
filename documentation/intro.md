# Introduction

## ToDo

- [ ] Rename and document "cartesian pulseq exporter"
- [x] Describe how to NUFFT reco in the reco page
- [x] Integrate and document BrainWeb
- [x] Rename matrics to "emitted signal" and "latent signal"
- [ ] Add more graph plotting functions
- [x] Rename spin_sim to isochromat_sim
- [ ] Document the pulseq importer and mention gradient / diffusion limitation

## MRzero Overview

MRzero {cite}`loktyushin_mrzero_2021` is a framework that replicates the whole MRI pipeline consisting of sequence and phantom definition, signal simulation, and image reconstruction. It uses a state-of-the-art PDG Bloch simulation, capable of calculating an accurate ADC signal comparable to that returned by a in vivo measurement of the signal in less time and while exhibiting no noise compared to isochromat based Monte-Carlo simulations.

The MRzero Framework is built using [PyTorch](https://pytorch.org/), enabling it to run on CUDA capable GPUs and providing automatic differentiation via backpropagation of the whole pipeline. This means that sequence parameters or phantom values can be optimized based on loss functions that consider the reconstructed image of the simulated signal.

## Getting Started

To see a simple sequence in action, have a look at the [FLASH](flash) example!

All examples are provided as [Jupyter Notebooks](https://jupyter.org/) and can be explored on sites like [Binder](https://mybinder.org/) or [Google Colab](https://colab.research.google.com/). Options are listed in the header of the according documentation pages.

To run the scripts locally on your computer, you need to install

- PyTorch: https://pytorch.org/get-started/locally/
- MRzeroCore: ```pip install MRzeroCore```

MRzeroCore also contains a pulseq .seq file parser and sequence exporter. It is fully self contained, so [pypulseq](https://github.com/imr-framework/pypulseq) or alternatives are only needed if you want to program sequences in them directly.

```{note}
This documentation builds on Jupyter Notebooks to represent text, code and outputs in an easy and reproducible way.
For the best user experience, it is recommended to install MRzeroCore locally and to use Python scripts for development. Editors like [PyCharm](https://www.jetbrains.com/de-de/pycharm/), [Spyder](https://www.spyder-ide.org/) or [VSCode](https://code.visualstudio.com/) provide autocompletion, an interactive console and direct access to the extensive documentation of MRzero.
```

## Build Process

The following Jupyter notebooks are executed when building this documentation, in order to keep all outputs up to date:

```{nb-exec-table}
```

## Literature

```{bibliography}
```
