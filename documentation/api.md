# API Reference

All functionality provided by MRzeroCore is re-exported at the top level. It is recommended to import MRzeroCore as follows:

```
import MRzeroCore as mr0

# Example: build a sequence
seq = mr0.Sequence()
rep = seq.new_rep(65)
rep.pulse.usage = mr0.PulseUsage.EXCIT
```

To run simulations on the GPU, the approach is similar to when using pyTorch:

```
graph = mr0.compute_graph(seq, obj)
# Calculate signal on the GPU, move returned tensor back to the CPU:
signal = mr0.execute_graph(seq.cuda(), obj.cuda()).cpu()
```

The following pages list all functionality provided by `MRzeroCore`

::::{grid}
:gutter: 3

:::{grid-item-card} [Sequence](sequence)
Create MRI sequences
:::

:::{grid-item-card} [Reconstruction](reco)
Reconstruct images
:::

:::{grid-item-card} [Phantom](phantom)
Various virtual subjects
:::

:::{grid-item-card} [Simulation](simulation)
Calculate ADC signals
:::
::::

