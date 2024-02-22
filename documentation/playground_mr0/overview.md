(playground_mr0)=
# Playground MR0

Welcome to Playground MR0, a playground to share, vary and simulate MR sequences.
MR sequences are written in the Pulseq standard using the pypulseq library.
Pulseq files are simulated with the efficient Phase Distribution Graph Bloch simulation.
Here we share links to example colabs that contain various MR sequences or let you upload your own seq file for simulation.

Many of the examples are build using [PyPulseq](https://github.com/imr-framework/pypulseq) and simulate the resulting .seq files with `MR0`.
These .seq files could also be measured on any MRI scanner using a Pulseq interpreter.


## Code and simulate PyPulseq

| Sequence |   |
| -------- | - |
| [Free Induction Decay](FID_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_FID_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Spin Echo CPMG](SE_CPMG_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_SE_CPMG_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Stimulated Echo 3 pulses - 5 echoes](STE_3pulses_5echoes_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_STE_3pulses_5echoes_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [FLASH 2D sequence](FLASH_2D_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_FLASH_2D_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [EPI 2D sequence](EPI_2D_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_EPI_2D_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [RARE 2D sequence](RARE_2D_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_RARE_2D_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [bSSFP 2D sequence](bSSFP_2D_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_bSSFP_2D_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Diffusion weighted Gradient Echo](DWI_GRE_2D_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_DWI_GRE_2D_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Interactive GRE to FLASH](GRE2FLASH_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_GRE_to_FLASH.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [DREAM STE for B0, B1, TxRx mapping](DREAM_STE_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_DREAM_STE_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [DREAM STID for B0, B1, TxRx mapping](DREAM_STID_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_DREAM_STID_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Burst TSE](burst_TSE_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_burst_TSE.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Compressed Sensing cartesian](mr0_CS_cartesian_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_CS_cartesian_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Compressed Sensing radial](mr0_CS_radial_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_CS_radial_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Pulseq with RF shimming](pulseq_ptx) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/pulseq_rf_shim.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Plot and simulate predifined .seq files

| Sequence |   |
| -------- | - |
| [Simulate pypulseq example sequences](mr0_pypulseq_example) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_pypulseq_exmpls_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Simulate own uploaded seq files](mr0_upload_seq) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_upload_seq.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## MR-zero optimization

Gradient descent optimizations using automatic differentiation by backpropagation

| Sequence |   |
| -------- | - |
| [IR FLASH 2D sequence for T1 mapping using a fit](IR_FLASH_fit) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_opt_FLASH_2D_IR_Fit_T1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [IR FLASH 2D sequence for T1 mapping using a NN](IR_FLASH_NN) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_opt_FLASH_2D_IR_voxelNN_T1.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## MR-double-zero optimization

Gradient-free optimization with [nevergrad](https://github.com/facebookresearch/nevergrad)

| Sequence | Google Colab |
| -------- | ------------ |
| [Ernst angle optimization](mr00_FLASH_2D_ernstAngle_opt) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr00_FLASH_2D_ernstAngle_opt.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## MR plot wall of fame

famous historic plots recreated


## FLASH test notebooks

| Sequence |   |
| -------- | - |
| [Pure `MR0` FLASH](flash) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/flash.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [pulseq FLASH](pulseq_flash) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/pulseq_flash.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [pulseq pTx FLASH](pulseq_pTx) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/pulseq_sim_pTx.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Diffusion weighted FLASH](flash_dwi) | <a target="_blank" href="https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/flash_DWI.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Notebook execution results

::::{toggle}

:::{nb-exec-table}
:::

::::
