# API overview

All available functions (`snake_case` naming) / classes (`CamelCase`) are listed here.
In the following pages, more information is listed.
For detailed descriptions of each items, look at their Python docstrings (as written in the source code or shown by your Python IDE) as well.

---

```python
import MRzeroCore as mr0
```
- [Sequence building blocks](sequence.md)
    - [`mr0.Sequence`](sequence.md#sequence)
    - [`mr0.chain`](sequence.md#chain)
    - [`mr0.Repetition`](sequence.md#repetition)
    - [`mr0.Pulse`](sequence.md#pulse)
    - [`mr0.PulseUsage`](sequence.md#pulseusage)
- Simulation data
    - `mr0.VoxelGridPhantom`
    - `mr0.CustomVoxelPhantom`
    - `mr0.SimData`
    - `mr0.generate_brainweb_phantoms`
- Simulation
    - `mr0.isochromat_sim`
    - `mr0.compute_graph`
    - `mr0.compute_graph_ext`
    - `mr0.Graph`
    - `mr0.execute_graph`
- `mr0.sig_to_mrd`
- `mr0.reco_adjoint`
- `mr0.pulseq_write_cartesian`
- `util` module
    - `mr0.util.get_signal_from_real_system`
    - `mr0.util.insert_signal_plot`
    - `mr0.util.pulseq_plot`
    - `mr0.util.pulseq_plot_142`
    - `mr0.util.pulseq_plot_pre14`
    - `mr0.util.imshow`
    - `mr0.util.load_phantom`
    - `mr0.util.simulate`
    - `mr0.util.simulate_2d`
