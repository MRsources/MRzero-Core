# Frequently asked questions

Common questions and solutions for MRzero Core and Pulseq integration.

## What's the easiest way to simulate a sequence?

**A:** Code and export your sequence as Pulseq file. One line:
```python
import MRzeroCore as mr0

# Automatic phantom download and simulation
signal, ktraj_adc = mr0.util.simulate('your_sequence.seq')
```

For complete Pulseq integration features, see: [Pulseq Integration Guide →](pulseq_integration.html)


## Can I simulate PyPulseq sequences directly without writing .seq files?

**A:** Yes! This is new functionality:
```python
import pypulseq as pp
import MRzeroCore as mr0

# Create sequence with PyPulseq
seq = pp.Sequence()
# ... build sequence ...

# Simulate directly - no .seq file needed!
signal, ktraj_adc = mr0.util.simulate(seq)  # seq is PyPulseq object
```
Note, this is will still generate a temporary seq file.
For avoiding seq files completely, check out [Pulseq-zero](https://github.com/pulseq-frame/pulseq-zero) 


## How do I use custom phantoms with B0 field modifications?

**A:** 
```python
# Load phantom with polynomial B0 augmentation
obj = mr0.util.load_phantom(
    [128, 128], 
    B0_polynomial=(0, 0, -150, 1, 1, 0),  # (const, lin_x, lin_y, quad_x, quad_y, quad_xy)
    url='numerical_brain_cropped.mat'     # Custom phantom URL or local file
)

# Further customize
obj.B1 = obj.B1**3     # Inhomogeneous B1+ field
obj.T2 = 80e-3         # Custom relaxation times

# High-accuracy simulation
signal, ktraj_adc = mr0.util.simulate(seq, obj, accuracy=1e-5)
```


## Does MRzero support ISMRMRD raw data format?

**A:** Current status: ISMRMRD raw data support is **not yet available**, but development is underway as a first pull request exists. So it can be expected in upcoming releases.
Stay updated:
- Watch the [GitHub repository](https://github.com/MRsources/MRzero-Core) for ISMRMRD support announcements
- Check [release notes](https://github.com/MRsources/MRzero-Core/releases) for new features


## My .seq file won't load / import fails

**A:** Pulseq 1.4.2 is used and tested by us the most. Pulseq 1.3 and 1.5 should work too, but might need some different load and plot functions.

See also in the comments in [mr0_upload_seq.ipynb {{#include snippets.txt:colab_button}}]({{#include snippets.txt:colab_url}}/mr0_upload_seq.ipynb)


## Getting version compatibility errors

**A:** Most issues are resolved by:
- Using matching Pulseq versions between creation and simulation
- Updating to latest MRzeroCore: `pip install --upgrade MRzeroCore`
- Checking [GitHub Issues](https://github.com/MRsources/MRzero-Core/issues) for known compatibility problems


## Simulation is too slow / runs out of memory

**A:** Reduce sequence resolution, go to single slice, reduce phantom resolution, reduce accuracy.
```python
# Reduce phantom size for testing
phantom = mr0.util.load_phantom(size=(32, 32))  # Instead of (128, 128)

# Reduce simulation accuracy for speed
signal = mr0.util.simulate(seq, accuracy=1e-2)  # Default: 1e-3

# Use smaller sequence for testing
# Remove unnecessary repetitions during development
```


## GPU not being used / CUDA errors

**A:** 
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# MR-zero uses GPU when seq0 and obj is using cuda
```python
graph = mr0.compute_graph(seq0, obj, 200, 1e-4)
# Minimum emitted and latent signal of 0.01 (compared per repetition to strongest state)
signal = mr0.execute_graph(graph, seq0.cuda(), obj.cuda(), 0.01, 0.01)
```


## How does MRzero handle slice-selective pulses?

**A:** Current limitation: MRzero uses instantaneous pulses and currently **ignores slice selection gradients**. This means:

```python
# In PyPulseq - slice selective pulse
rf, gz, gzr = pp.make_sinc_pulse(
    flip_angle=90*np.pi/180, 
    slice_thickness=5e-3,  # This is ignored in MRzero
    duration=1e-3
)
```
To have correct gradient behavior (spoiling, diffusion), our interpreter puts the instantanoues pulse in the center of the pulse, and correctly plays out the gradients.
Thus, if you have assymetric pulses, you have to be careful, and better replace them by block pulses.

Workarounds:
1. **For 2D simulations**: Use thin phantom slices to approximate slice selection
2. **For comparison with Scans**: Consider this limitation when comparing to real measurements, apparent flip angles might be smaller due to slice profile effects. 


## How do I choose the right accuracy parameter?

**A:** The `accuracy` parameter controls simulation precision vs speed. When in doubt rerun with higher accuracy.

```python
# High accuracy (slow, precise)
signal = mr0.util.simulate(seq, accuracy=1e-8)  # Research-quality

# Fast accuracy (quick testing)
signal = mr0.util.simulate(seq, accuracy=1e-2)  # Development/testing

# Very fast (rough estimates)
signal = mr0.util.simulate(seq, accuracy=1e-1)  # Initial debugging only
```


## What does the accuracy parameter actually control?

**A:** It sets the `min_emitted_signal` and `min_latent_signal` thresholds in the PDG simulation:
- Lower values = more computation but higher precision
- Higher values = faster simulation but potential artifacts
- For sequences with small signals (like multi-echo), use lower values


## Does MRzero support magnetization transfer effects?

**A:** Current status: MRzero currently does **not model MT effects**. The simulation assumes:
- Single pool (free water) only
- No bound pool interactions
- No MT saturation effects


## How accurate is diffusion simulation in MRzero?

**A:** MRzero handles currently only isotropic diffusion, using the EPG equations from [Weigel et al.](https://doi.org/10.1016/j.jmr.2010.05.011):


## How are relaxation times handled?

**A:** 
```python
# Set realistic values in phantom
phantom.T1 = 1.0    # seconds
phantom.T2 = 80e-3  # seconds  
phantom.T2dash = 50e-3  # T2' dephasing effects with 1/T2* = 1/T2 + 1/T2'

# All relaxation effects are included in simulation
# including T2* decay during readout
```
Tips:
- Use literature values for tissue types
- T2' does NOT include intravoxel B0 inhomogeneity effects, this can be achieved by high enough B0 map resolution.


## Sequence created in MATLAB doesn't work in Python

**A:** Check Pulseq version compatibility
```python
# Try different import approaches
seq = mr0.Sequence.import_file('file.seq')  # Standard since Pulseq 1.4, shoudl work for 1.5
seq = mr0.Sequence.from_seq_file('file.seq')  # Alternative for Pulseq 1.3 
```

---

**Resources:**
- **GitHub Issues**: [Report bugs](https://github.com/MRsources/MRzero-Core/issues)
- **Examples**: [Interactive examples](playground_mr0/overview.html) 
- **API Docs**: [Complete reference](api/sequence.html)
- **Pulseq Integration**: [Integration guide](pulseq_integration.html)

**Still stuck?** Try the interactive examples first - they solve most common issues! 