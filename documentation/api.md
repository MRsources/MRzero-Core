# API Reference

```{warning}
NOTE: Don't use autosummary as it is impossible to configure to produce the desired output. Instead, write all documentation pages yourself and use autodoc to include the dcstrings.
```

All functionality provided by MRzeroCore is re-exported at the top level. It is recommended to import MRzeroCore as follows:

```
import MRzeroCore as mr0

# Example: build a sequence
seq = mr0.Sequence()
rep = seq.new_rep(65)
rep.pulse.usage = mr0.PulseUsage.EXCIT
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

