# Introduction


## MRzero Overview

[MRzero](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28727) is a framework that replicates the whole MRI pipeline consisting of sequence and phantom definition, signal simulation, and image reconstruction. It uses a state-of-the-art PDG Bloch simulation, capable of calculating an accurate ADC signal comparable to that returned by a in vivo measurement of the signal in less time and while exhibiting no noise compared to isochromat based Monte-Carlo simulations.

The MRzero Framework is built using [PyTorch](https://pytorch.org/), enabling it to run on CUDA capable GPUs and providing automatic differentiation via backpropagation of the whole pipeline. This means that sequence parameters or phantom values can be optimized based on loss functions that consider the reconstructed image of the simulated signal.
## Playground MR0

MRzero Core can be used in Jupyter Notebooks and can be used in online services like Google Colab.
A constantly increasing selection of example script can be found in the [Playground MR0](playground_mr0)

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

## Literature

_Endres, J., Weinmüller, S., Dang, HN., Zaiss, M._ **Phase distribution graphs for fast, differentiable, and spatially encoded Bloch simulations of arbitrary MRI sequences.** Magn Reson Med. 2024; 92(3): 1189-1204. https://doi.org/10.1002/mrm.30055

_Weinmüller, S., Endres, J., Dang, HN., Stollberger, R., Zaiss, M._ **MR-zero meets FLASH – Controlling the transient signal decay in gradient- and rf-spoiled gradient echo sequences** Magn Reson Med. 2024; https://doi.org/10.1002/mrm.30318

_Loktyushin, A., Herz, K., Dang, H._ **MRzero - Automated discovery of MRI sequences using supervised learning.** Magn Reson Med. 2021; 86: 709–724. https://doi.org/10.1002/mrm.28727

_Dang, H., Endres, J., Weinmüller, S., et al._ **MR-zero meets RARE MRI: Joint optimization of refocusing flip angles and neural networks to minimize T2-induced blurring in spin echo sequences.** Magn Reson Med. 2023; 90(4): 1345-1362. https://doi.org/10.1002/mrm.29710


**Abstarcts**

_Glang, Felix, Alexander Loktyushin, Kai Herz, Hoai Nam Dang, Anagha Deshmane, Simon Weinmüller, Arnd Doerfler, .., Moritz Zaiss._ **Advances in MRzero: supervised learning of parallel imaging sequences including joint non-Cartesian trajectory and flip angle optimization**. In Proc. Intl. Soc. Mag. Reson. Med. 29, 29:4200, 2021. https://archive.ismrm.org/2021/4200.html.

Weinmüller, Simon, Tim Baum, Hoai Nam Dang, Jonathan Endres, und Moritz Zaiss._ **DREAM-zero – Optimized variable flip angles for decreased image blurring in magnetization-prepared DREAM sequences**. 2023, Magnetic Resonance Materials in Physics, Biology and Medicine, Book of Abstracts ESMRMB 2023 Online 39th Annual Scientific Meeting 4–7 October 2023 36 (1): 240–41. https://doi.org/10.1007/s10334-023-01108-9.
