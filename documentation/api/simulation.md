(simulation)=
```{eval-rst}
.. currentmodule:: MRzeroCore
```

# Simulation

The main simulation of MRzero is based on Phase Distribution Graphs ([PDG simulation](pdg_sim)). It is split in two parts: The [pre-pass](pre_pass) and [main-pass](main_pass). The pre-pass is an approximate calculation of the signal to determine which parts of the full signal equations (which states in PDG terms) are important to the signal. This info is stored in a graph, that is then "executed" by the main-pass at full resolution.

There is also an [isochromat simulation](isochromat_sim_doc) available. It is meant as an intentionally simple ground truth and not written for speed (it generally performs well but has a large overhead, noticable at low resolutions).

::::{grid}
:gutter: 3

:::{grid-item-card} [PDG simulation](pdg_sim)
Two-pass simulation based on Phase Distribution Graphs. Fast, analytical (noise-free), precise and differentiable signal calculation.
:::

:::{grid-item-card} [Isochromat simulation](isochromat_sim_doc)
Very basic isochromat-based Bloch simulation. Meant as ground-truth and intentionally simple implementation. At least 10x slower than PDG.
:::
::::
