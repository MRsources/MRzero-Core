(sequence)=
```{eval-rst}
.. currentmodule:: MRzeroCore
```

# Sequence

A MRI {class}`Sequence` is a Python list of {class}`Repetition` s, each of which is starting with an instantaneous {class}`Pulse` .
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

## Pulseq

Sequences can also be imported from [Pulseq](https://pulseq.github.io/) `.seq` files using the {meth}`Sequence.from_seq_file` method.

:::{note}
The importer tries to minimize the amount of events for imported sequences. This can be undesirable for diffusion-weighted sequences that rely on spoiler gradients, which might be removed in this process. For that reason, this behaviour might be removed in the future. Furthermore, MRzero currently uses instantaneous pulses and will ignore slice selection, off-resonance etc.
:::

Sometimes it might be desirable to measure multiple contrasts in a single sequence. This can be realized by combining sequences with {func}`chain`, followed by masking the simulated signal using {meth}`Sequence.get_contrasts()`. Alternatively, {attr}`Repetition.adc_usage` allows to manually assign ADC samples to different contrasts.

To export MRzero sequences as Pulseq .seq files, {func}`pulseq_write_cartesian` can be used. This exporter does only support sequences with k-space trajectories that are on a cartesian grid. Exporters that are more flexible will be added in the future, as well as better documentation of these exporters.

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

In MRzero, a repetition describes a section of the sequence starting with an RF pulse and ending just before the next. This is intuitive for sequences that consists of many similar sections (usually identical apart from phase encoding), but is used more loosely here as a general term even for sequences, where those "repetitions" are completely different.

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
