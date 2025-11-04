(reco)=
```{eval-rst}
.. currentmodule:: MRzeroCore
```

# Reconstruction


## NUFFT Reconstruction

Nufft reconstruction is not integrated into `mr0` directly, but can be realized with `torchkbnufft` and code similar to the following:

```python
import torchkbnufft as tkbn

# Construct a sequence "seq" and simulate it, resulting in "signal"

kdata = signal.view(1, 1, -1)
ktraj = seq.get_kspace()[:, :2].T / (2 * np.pi)
im_size = (64, 64)

adjnufft_ob = tkbn.KbNufftAdjoint(im_size)
dcomp = tkbn.calc_density_compensation_function(ktraj, im_size=im_size)
reco = adjnufft_ob(kdata * dcomp, ktraj).view(im_size)
```


## Adjoint Reconstruction

Adjoint reconstruction builds a very simple backwards version of the encoding / measurement operation. Essentially, it resembles a DFT and is thus slower and consumes more memory than an FFT, but can handle any readout trajectory.

```{eval-rst}
.. autofunction:: reco_adjoint
```
