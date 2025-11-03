# Pulseq Integration

MRzero Core provides seamless integration with the Pulseq standard, making it easy to simulate any Pulseq sequence file in just one line of code.

## Get Started Immediately

### Simplest possible - one line simulation:
```python
import MRzeroCore as mr0

# That's it - simulate any .seq file with automatic phantom download!
signal, ktraj_adc = mr0.util.simulate(seq)  # Downloads standard phantom if not found

seq.plot(plot_now=False)
mr0.util.insert_signal_plot(seq=seq, signal =signal.numpy())  # show sequence plots with simulated signals inserted in ADC plot.
plt.show()
```

### Full control with prepass-mainpass:
```python
import MRzeroCore as mr0
obj_p = mr0.VoxelGridPhantom.brainweb('subject05.npz') #load a phantom object from file
obj_p = obj_p.interpolate(64, 64, 32).slices([15])
obj_p = obj_p.build()

seq0 = mr0.Sequence.import_file('your_sequence.se')  # Read in the Pulseq seq file

graph = mr0.compute_graph(seq0, obj_p, 2000, 1e-4)  # pre-pass
signal = mr0.execute_graph(graph, seq0, obj_p, print_progress=False) # main-pass

seq.plot(plot_now=False)
mr0.util.insert_signal_plot(seq=seq, signal =signal.numpy())
plt.show()
```

**Try now**: [mr0_upload_seq.ipynb {{#include snippets.txt:colab_button}}]({{#include snippets.txt:colab_url}}/mr0_upload_seq.ipynb)

## Simplified Workflows

### Utily functions

MRzero now includes **streamlined utility functions** for common simulation tasks:

1. **One line** - automatic everything
    ```python
    # Quickest 2D brain phantom simulation 
    signal, ktraj_adc = mr0.util.simulate(seq)  # Downloads standard phantom automatically
    ```
2. **Two lines** - custom phantom parameters
    ```python
    # Quick 2D brain phantom sim with modifications
    obj = mr0.util.load_phantom([128,128], B0_polynomial=(0,0,-150,1,1,0))  # Custom B0 map
    signal, ktraj_adc = mr0.util.simulate(seq, obj)
    ```
3. **Three lines** - full control
    ```python
    # Custom phantom from URL/file with arbitrary modifications and high accuracy
    obj = mr0.util.load_phantom([128,128], B0_polynomial=(0,0,-150,1,1,0), url='numerical_brain_cropped.mat')  
    obj.B1 = obj.B1**3     # Very inhomogeneous B1+ field
    signal, ktraj_adc = mr0.util.simulate(seq, obj, accuracy=1e-5)
    ```


### Direct PyPulseq Integration
```python
import pypulseq as pp

# Create sequence with PyPulseq
seq = pp.Sequence()
# ... build sequence ...

# Simulate directly - no need to write/read .seq file!
signal, ktraj_adc = mr0.util.simulate(seq)  # seq can be PyPulseq object
```


### For MATLAB Pulseq Users
See [MATLAB integration](matlab.md)


## Examples - Ready-to-run in Google Colab

| Notebook | Complexity | {{#include snippets.txt:colab_button}} |
| -------- | ---------- | -------------------------------------- |
| Simulate own uploaded seq files | Beginner | [mr0_upload_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_upload_seq.ipynb) |
| FLASH 2D sequence | Beginner | [mr0_FLASH_2D_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_FLASH_2D_seq.ipynb) |
| GRE EPI 2D sequence | Intermediate | [mr0_EPI_2D_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_EPI_2D_seq.ipynb) |
| DWI SE EPI 2D sequence | Advanced | [mr0_DWI_SE_EPI.ipynb]({{#include snippets.txt:colab_url}}/mr0_DWI_SE_EPI.ipynb) |
| TSE 2D sequence | Advanced | [mr0_TSE_2D_multi_shot_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_TSE_2D_multi_shot_seq.ipynb) |


[**Browse All 20+ Examples →**](playground.md)


## Further Features

### Visualization
```python

# Plot PyPulseq sequence plot with MR-zero signals
seq.plot(plot_now=False)
mr0.util.insert_signal_plot(seq=seq, signal =signal.numpy())
plt.show()

# Plot the detected k-space used by MR-zero
seq.plot_kspace_trajectory()

#Plot the phase distribution graph
graph = mr0.compute_graph(seq0, obj_p, 2000, 1e-4)  # pre-pass
graph.plot()

```

### Simulation
```python
# Simple simulation
signal = mr0.util.simulate(seq)

# With custom phantom
phantom = mr0.util.load_phantom(size=(64, 64))
signal, kspace = mr0.util.simulate(seq, phantom)
```

## Getting Help

**Quick help:**
- **Common issues**: [FAQ & Troubleshooting](faq.html)
- **GitHub Issues**: [Report bugs or ask for features](https://github.com/MRsources/MRzero-Core/issues)
- **Github Discussions**: [Ask Questions](https://github.com/MRsources/MRzero-Core/discussions)
