# API overview

All available functions (`snake_case` naming) / classes (`CamelCase`) are listed here.
In the following pages, more information is listed.
For detailed descriptions of each items, look at their Python docstrings (as written in the source code or shown by your Python IDE) as well.

---

```python
import MRzeroCore as mr0
```
- [Sequence building blocks](api-sequence.md)
    - [`mr0.Sequence`](api-sequence.md#sequence)
    - [`mr0.chain`](api-sequence.md#chain)
    - [`mr0.Repetition`](api-sequence.md#repetition)
    - [`mr0.Pulse`](api-sequence.md#pulse)
    - [`mr0.PulseUsage`](api-sequence.md#pulseusage)
- [Simulation data](api-phantom.md)
    - [`mr0.SimData`](api-phantom.md#simdata)
    - [`mr0.VoxelGridPhantom`](api-phantom.md#voxelgridphantom)
    - [`mr0.TissueDict`](api-phantom.md#tissuedict)
    - [`mr0.CustomVoxelPhantom`](api-phantom.md#customvoxelphantom)
- [NIfTI phantoms](api-nifti.md)
    - [`mr0.PhantomUnits`](api-nifti.md#phantomunits)
    - [`mr0.PhantomSystem`](api-nifti.md#phantomsystem)
    - [`mr0.NiftiRef`](api-nifti.md#niftiref)
    - [`mr0.NiftiMapping`](api-nifti.md#niftimapping)
    - [`mr0.NiftiTissue`](api-nifti.md#niftitissue)
    - [`mr0.NiftiPhantom`](api-nifti.md#niftiphantom)
- [Simulation](api-simulation.md)
    - [`mr0.isochromat_sim`](api-simulation.md#isochromat_sim)
    - [`mr0.compute_graph`](api-simulation.md#compute_graph)
    - [`mr0.compute_graph_ext`](api-simulation.md#compute_graph_ext)
    - [`mr0.Graph`](api-simulation.md#graph)
    - [`mr0.execute_graph`](api-simulation.md#execute_graph)
- [Reconstruction](api-reconstruction.md)
    - [`mr0.reco_adjoint`](api-reconstruction.md#reco_adjoint)
- [`util` module](api-util.md)
    - [`mr0.util.get_signal_from_real_system`](api-util.md#get_signal_from_real_system)
    - [`mr0.util.insert_signal_plot`](api-util.md#insert_signal_plot)
    - [`mr0.util.pulseq_plot`](api-util.md#pulseq_plot)
    - [`mr0.util.pulseq_plot_142`](api-util.md#pulseq_plot_142)
    - [`mr0.util.pulseq_plot_pre14`](api-util.md#pulseq_plot_pre14)
    - [`mr0.util.imshow`](api-util.md#imshow)
    - [`mr0.util.load_phantom`](api-util.md#load_phantom)
    - [`mr0.util.simulate`](api-util.md#simulate)
    - [`mr0.util.simulate_2d`](api-util.md#simulate_2d)
- [`mr0.sig_to_mrd`](api-unsorted.md#sig_to_mrd)
- [`mr0.pulseq_write_cartesian`](api-unsorted.md#pulseq_write_cartesian)
- [`mr0.generate_brainweb_phantoms`](api-unsorted.md#generate_brainweb_phantoms)
