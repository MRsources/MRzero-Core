![logo](logo.png)

```bash
# Requires Python 3.9 or higher
pip install MRzeroCore
```

MR-zero is a framework for easy MRI sequence optimization and development of self-learning sequence development strategies.
The vision is documented in [this paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28727).
These goals are backed by a modern Bloch simulation (see [this paper](https://doi.org/10.1002/mrm.30055)).
More material can be found in the [literature](literature.md).

# Quick start

For a quick introduction, look at the [**MR-zero Playground**](playground.md)!

An ever growing collection of Jupyter Notebook demonstrate core capabilities of MR-zero.
These notebooks run on Google Colab - directly in your browser, no installation needed!

Alternatively, install MR-zero Core locally:
```bash
# Requires Python 3.9 or higher
pip install MRzeroCore
```

# Simulate any Pulseq sequence in one line

MR-zero makes MRI sequence simulation *easy* - just one line for simulating .seq files:

```python
import MRzeroCore as mr0

# That's it - automatic phantom download and simulation!
signal, ktraj_adc = mr0.util.simulate('your_sequence.seq')
```

Even simpler with **PyPulseq** - no need to worry about writing .seq files:

```python
import pypulseq as pp
# Create sequence with PyPulseq
seq = build_your_sequence()
# ... build sequence ...
signal, ktraj_adc = mr0.util.simulate(seq)
# ... reconstruct, e.g.: with a NUFFT of signal and ktraj_adc
```

# Further documentation

MR-zero uses [PyTorch](https://pytorch.org/) for computation, enabling GPU compute automatic differentiation with backpropagation.
This means you can easily develop your own loss functions, sequence building code and more - and then optimize any input parameters efficiently with gradient descent.
In the example above, this could mean to simply extend `build_your_sequence()` by sequence parameters like flip angles and writing an image based loss function for the reconstruction.
Then you can lie back and let the computer find the best pulse train.

How this can be done will be explained in the following pages, where this documentation lists the most important ideas and applications of MR-zero.
If you think that something is missing or misleading, pleas open an issue [on GitHub](https://github.com/MRsources/MRzero-Core) or directly via the button on the top right on every documentation page.