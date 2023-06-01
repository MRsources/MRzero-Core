(pdg_sim)=
```{eval-rst}
.. currentmodule:: MRzeroCore
```

# PDG Simulation

Phase Distribution Graphs split the magnetization into multiple states, forming a Graph over the duration of the sequence. More details can be found in the PDG Paper, once published.

(pre_pass)=
## Pre-Pass

Two functions are provided: {func}`compute_graph` and {func}`compute_graph_ext`. They are both wrappers around the actual prepass, which is written in rust. {func}`compute_graph` computes average values for $T_1$, $T_2$, $T_2'$ and $D$ and then calls {func}`compute_graph_ext`.

```{eval-rst}
.. autofunction:: compute_graph

.. autofunction:: compute_graph_ext
```

## Phase Distribution Graph

```{eval-rst}
.. autoclass:: MRzeroCore.simulation.pre_pass.Graph
    :members:
```

(main_pass)=
## Main-Pass

Takes the {class}`Sequence`, the {class}`SimData` and the {class} `Graph` produced by both in the [pre-pass](pre_pass) in order to calculate the measured ADC signal. Because of the work done by the [pre-pass](pre_pass), only the minimal work needed in order to achieve the desired precision is executed. This precision can be tuned by the {attr}`min_emitted_signal` and {attr}`min_latent_signal` thresholds. Higher values lead to less states being simulated, which improves speed and reduces accuracy. A value of 1 will mean that 2 states will be simulated (z0 and one +), resulting in the FID signal. A value of 0 means that everything will be simulated that somehow contributes to the signal.

```{eval-rst}
.. autofunction:: execute_graph
```
