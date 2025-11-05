# MATLAB Pulseq ↔ MR-zero Integration Guide

This guide shows how to seamlessly integrate MATLAB Pulseq with Python MR-zero for powerful sequence development and simulation workflows.

## Quick Start for MATLAB Users

*Already have MATLAB Pulseq sequences?* Get started in 3 steps:

1. **Export your sequence**: `seq.write('my_sequence.seq')` in MATLAB
2. **Upload to Colab**: Use our ([mr0_upload_seq.ipynb ![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_upload_seq.ipynb))
3. **Simulate instantly**: No Python installation required!

## Integration Workflows

### Workflow 1: MATLAB seq creation → Python Simulation

**Use Case**: Leverage MR-zero's fast simulation while keeping your MATLAB development workflow

```matlab
% In MATLAB: Create your sequence as usual
seq = mr.Sequence();
% ... build your sequence ...
seq.write('my_flash.seq');
```
Then switch to python and load the seq file.
    
```python
# In Python/Colab: Simulate with one line
import MRzeroCore as mr0
seq = mr0.Sequence.import_file('my_flash.seq')
signal = mr0.util.simulate(seq)
```

You can just upload your seq file in [mr0_upload_seq.ipynb ![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_upload_seq.ipynb)

### Workflow 2: MATLAB seq creation → Call Python from Matlab using sys command

see Pulseq demo [simMR0](https://github.com/pulseq/pulseq/blob/master/matlab/demoUnsorted/simMR0.m)



---

**Need Help?** Try our [interactive examples](playground_mr0) first, or check the [GitHub issues](https://github.com/MRsources/MRzero-Core/issues) for community support. 