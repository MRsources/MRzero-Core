# Pulseq Integration

**MRzero Core provides seamless integration with the Pulseq standard**, making it easy to simulate any Pulseq sequence file in just one line of code.

## **Get Started Immediately**

### **Simplest possible - one line simulation:**
```python
import MRzeroCore as mr0

# That's it - simulate any .seq file with automatic phantom download!
signal, ktraj_adc = mr0.util.simulate(seq)  # Downloads standard phantom if not found

seq.plot(plot_now=False)
mr0.util.insert_signal_plot(seq=seq, signal =signal.numpy())  # show sequence plots with simulated signals inserted in ADC plot.
plt.show()
```

### **Full control with prepass-mainpass:**
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

**Try now**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_upload_seq.ipynb)

## **Simplified Workflows**

**MRzero now includes streamlined utility functions** for common simulation tasks:

### **One Line - Automatic Everything**
```python
# Quickest 2D brain phantom simulation 
signal, ktraj_adc = mr0.util.simulate(seq)  # Downloads standard phantom automatically
```

### **Two Lines - Custom Phantom Parameters**
```python
# Quick 2D brain phantom sim with modifications
obj = mr0.util.load_phantom([128,128], B0_polynomial=(0,0,-150,1,1,0))  # Custom B0 map
signal, ktraj_adc = mr0.util.simulate(seq, obj)
```

### **Three Lines - Full Control**
```python
# Custom phantom from URL/file with arbitrary modifications and high accuracy
obj = mr0.util.load_phantom([128,128], B0_polynomial=(0,0,-150,1,1,0), url='numerical_brain_cropped.mat')  
obj.B1 = obj.B1**3     # Very inhomogeneous B1+ field
signal, ktraj_adc = mr0.util.simulate(seq, obj, accuracy=1e-5)
```

### **Direct PyPulseq Integration**
```python
import pypulseq as pp

# Create sequence with PyPulseq
seq = pp.Sequence()
# ... build sequence ...

# Simulate directly - no need to write/read .seq file!
signal, ktraj_adc = mr0.util.simulate(seq)  # seq can be PyPulseq object
```



### **For MATLAB Pulseq Users**
See [MATLAB integration](matlab_integration.html)


## **Examples - Ready-to-run examples in Google Colab:**

| **Sequence** | **Description** | **Complexity** | **Try Now** |
|-------------|-----------------|----------------|-------------|
| **Upload Any .seq** | Universal .seq file simulator | Beginner | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_upload_seq.ipynb) |
| **FLASH 2D** | Basic gradient echo imaging | Beginner | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_FLASH_2D_seq.ipynb) |
| **EPI** | Echo planar imaging | Intermediate | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_EPI_2D_seq.ipynb) |
| **Diffusion** | Diffusion-weighted imaging | Advanced | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_DWI_SE_EPI.ipynb) |
| **TSE** | Turbo spin echo | Advanced | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_TSE_2D_multi_shot_seq.ipynb) |

[**Browse All 13+ Examples →**](playground_mr0/overview.html)

## **Further Features**


### **Visualization**
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

### **Simulation**
```python
# Simple simulation
signal = mr0.util.simulate(seq)

# With custom phantom
phantom = mr0.util.load_phantom(size=(64, 64))
signal, kspace = mr0.util.simulate(seq, phantom)
```

## **Getting Help**

**Quick help:**
- **Common issues**: [FAQ & Troubleshooting](faq.html)
- **GitHub Issues**: [Report bugs or ask for features](https://github.com/MRsources/MRzero-Core/issues)
- **Github Discussions**: [Ask Questions](https://github.com/MRsources/MRzero-Core/discussions)
