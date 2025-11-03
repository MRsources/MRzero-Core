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

We recommend periodically validating results with more precise simulations.

Each sequence may require unique adjustments to balance simulation speed and accuracy. It's often beneficial to start with a higher state count for a more accurate baseline.

For sequences with poor timing or spoiling, state explosion can occur (Hennig 1991). The state selection thresholds will filter out these states, potentially leading to inaccurate simulation results.