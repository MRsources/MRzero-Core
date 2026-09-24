# State selection

### Documentation on PDG State Parameters and Configuration

#### Overview
The Phase Distribution Graph (PDG) states are frequently initialized with default parameters. This document outlines the essential considerations and configurations required for optimal performance, especially when dealing with exotic sequences.

#### Standard PDG Configuration
Typically, PDG states are invoked using the following standard parameters:
```python
graph = mr0.compute_graph(seq0, obj_ph)
signal = mr0.execute_graph(graph, seq0, obj_ph)
```
The standard values used then are here written explicitly:
```python
graph = mr0.compute_graph(seq0, obj_ph, max_state_count=200, min_state_mag=0.0001)
signal = mr0.execute_graph(graph, seq0, obj_ph, min_emitted_signal=0.01, min_latent_signal=0.01)
```
#### Importance of State Selection Thresholds
For arbitrary sequences, careful examination of state selection is crucial. This process remains important for all users employing PDG for scientific purposes.

As outlined in the main paper, there are four main parameters to adjust:

1. **Threshold Number of States (`max_state_count`)**
2. **Threshold Magnetization (`min_state_mag`)**
3. **Threshold Emitted Signal (`min_emitted_signal`)**
4. **Threshold Latent Signal (`min_latent_signal`)** 

To resolve issues of inaccurate simulations one can adjust the parameters as follows to get a more accurate simulation:
```python
graph = mr0.compute_graph(seq0, obj_ph, max_state_count=5000, min_state_mag=1e-12)
signal = mr0.execute_graph(graph, seq0, obj_ph, min_emitted_signal=0.001, min_latent_signal=0.001)
```
This adjustment will result in a longer simulation duration, but more accurate results

#### One graph for every tissue

`compute_graph` builds the graph from a single T1, T2, T2' and D. By default those are the means of the phantom, so the graph is the one a phantom made entirely of average tissue would need. On a brain that drops states CSF or fat still carry, and the main pass cannot simulate a state the graph does not hold. Lowering `min_state_mag` does not bring those states back: they were never built.

From version 1.1.0, pass a tissue table with more than one entry. The prepass then runs once on the phantom means and once per tissue, and the graphs are unioned. The main-pass thresholds are unchanged. Omit `tissues`, or pass only one tissue, and you get the mean prepass above.

```python
tissues = {
    "gm":  {"T1": 1.56, "T2": 0.083, "T2dash": 0.32,  "D": 0.83},
    "wm":  {"T1": 0.83, "T2": 0.075, "T2dash": 0.18,  "D": 0.65},
    "csf": {"T1": 4.16, "T2": 1.65,  "T2dash": 0.059, "D": 3.19},
    "fat": {"T1": 0.37, "T2": 0.125, "T2dash": 0.012, "D": 0.1},
}
graph = mr0.compute_graph(
    seq0, obj_ph,
    max_state_count=5000,
    min_state_mag=1e-5,
    tissues=tissues,
)
signal = mr0.execute_graph(
    graph, seq0, obj_ph,
    min_emitted_signal=0.01,
    min_latent_signal=0.01,
)
```

`tissues` may also be the path of a BIfTI-phantom JSON file. The relaxation values are read from each tissue (`T2'` and `ADC` in that file). There is no built-in table: the caller supplies the tissues that bracket the object.

The extra prepasses run concurrently. The merged graph is larger than the mean graph, because it keeps states only some tissues need.

We recommend periodically validating results with more precise simulations.

Each sequence may require unique adjustments to balance simulation speed and accuracy. It's often beneficial to start with a higher state count for a more accurate baseline.

For sequences with poor timing or spoiling, state explosion can occur (Hennig 1991). The state selection thresholds will filter out these states, potentially leading to inaccurate simulation results.