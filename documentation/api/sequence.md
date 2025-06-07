(sequence)=
```{eval-rst}
.. currentmodule:: MRzeroCore
```

# Sequence definition in MRzeroCore

The MRzeroCore {class}`Sequence` is a Python list of {class}`Repetition` s, each of which is starting with an instantaneous {class}`Pulse` .
Thus, MRzeroCore treats all sequences as list of 'repetitions', that have the same building blocks of pulse, gradient moments, adcs and event times. However each 'repetition' can be different, thus some events can be just zero. See **repetition** for more details.
The {class}`PulseUsage` of these pulses is only used for automatically building the k-space trajectory for easier reconstruction.
Typical use is similar to the following:

```python
import MRzeroCore as mr0
from numpy import pi

seq = mr0.Sequence()

# Iterate over the repetitions
for i in range(64):
    rep = seq.new_rep(2 + 64 + 2)
    
    # Set the pulse
    rep.pulse.usage = mr0.PulseUsage.EXCIT
    rep.pulse.angle = 5 * pi/180
    rep.pulse.phase = i**2 * 117*pi/180 % (2*pi)

    # Set encoding
    rep.gradm[:, :] = ...

    # Set timing
    rep.event_time[:] = ...

    # Set ADC
    rep.adc_usage[2:-2] = 1
    rep.adc_phase[2:-2] = pi - rep.pulse.phase
```

Sometimes it might be desirable to measure multiple contrasts in a single sequence. This can be realized by combining sequences with {func}`chain`, followed by masking the simulated signal using {meth}`Sequence.get_contrasts()`. Alternatively, {attr}`Repetition.adc_usage` allows to manually assign ADC samples to different contrasts.


:::{note}
The importer tries to minimize the amount of events for imported sequences. This can be undesirable for diffusion-weighted sequences that rely on spoiler gradients, which might be removed in this process. For that reason, this behaviour might be removed in the future. Furthermore, MRzero currently uses instantaneous pulses and will ignore slice selection, off-resonance etc.
:::



```{eval-rst}
.. autofunction:: pulseq_write_cartesian
```

## Sequence

```{eval-rst}
.. autoclass:: Sequence
    :members:

.. autofunction:: chain
```

## Repetition

In MRzeroCore, a repetition describes a section of the sequence starting with an RF pulse and ending just before the next. This is intuitive for sequences that consists of many similar sections (usually identical apart from phase encoding), but is used more loosely here as a general term even for sequences, where those "repetitions" are completely different.

```{eval-rst}
.. autoclass:: Repetition
    :members:
```

## Pulse

```{eval-rst}
.. autoclass:: Pulse
    :members:
```

## PulseUsage

```{eval-rst}
.. autoclass:: PulseUsage
    :members:
```

# **Import of Pulseq sequences into the MRzeroCore sequence format**

You can import Pulseq files into the internal MRzeroCore sequence format.

```python
import pypulseq as pp
import MRzeroCore as mr0

seq0 = mr0.Sequence.import_file('temp.seq')

# Now simulate with MR-zero
signal = mr0.util.simulate(seq0)
```

See more Details under the item [Pulseq Integration →](pulseq_integration.html)
Examples in Colab:
- [![Upload and simulate a .seq file](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_upload_seq.ipynb)
- [Simulate PyPulseq example sequences](mr0_pypulseq_example)

