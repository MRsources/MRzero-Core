(isochromat_sim_doc)=
```{eval-rst}
.. currentmodule:: MRzeroCore
```

# Isochromat Bloch Simulation

This simulation is meant as ground-truth and purposely written as simple as possible. Spins (used interchangably with isochromats) are distributed in a voxel via the [R2 Sequence](http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/).

:::{note}
The isochromat simulation still uses cubic voxels, ignoring the voxel shape specified in {class}`SimData`. **Cubic voxels are nearly always the wrong choice**. The `"box"` voxel shape is only supported by PDG so it can be compared to the isochromat simulation.

Additionally, the isochromat simulation does not support diffusion.
:::

```{eval-rst}
.. autofunction:: isochromat_sim
```

## Helper functions

The simulation is split into multiple functions, each of them only executing a single, easily understandable step of the overall simulation. These are not exposed by `MRzeroCore` as they are not meant to be used directly.

```{eval-rst}
.. autofunction:: MRzeroCore.simulation.isochromat_sim.measure

.. autofunction:: MRzeroCore.simulation.isochromat_sim.relax

.. autofunction:: MRzeroCore.simulation.isochromat_sim.dephase

.. autofunction:: MRzeroCore.simulation.isochromat_sim.flip

.. autofunction:: MRzeroCore.simulation.isochromat_sim.grad_precess

.. autofunction:: MRzeroCore.simulation.isochromat_sim.B0_precess

.. autofunction:: MRzeroCore.simulation.isochromat_sim.intravoxel_precess
```
