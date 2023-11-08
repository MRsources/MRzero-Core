# Playground MR0

Welcome to Playground MR0, a playground to share, vary and simulate MR sequences.
MR sequences are written in the Pulseq standard using the pypulseq library.
Pulseq files are simulated with the efficient Phase Distribution Graph Bloch simulation.
Here we share links to example colabs that contain various MR sequences or let you upload your own seq file for simulation.

Many of the examples are build using [PyPulseq](https://github.com/imr-framework/pypulseq) and simulate the resulting .seq files with `MR0`.
These .seq files could also be measured on any MRI scanner using a Pulseq interpreter.


## Code and simulate PyPulseq

| Sequence | Google Colab |
| -------- | ------------ |
| [Free Induction Decay](FID_seq) | [mr0_FID_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_FID_seq.ipynb) |
| [Spin Echo CPMG](SE_CPMG_seq) | [mr0_SE_CPMG_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_SE_CPMG_seq.ipynb) |
| [Stimulated Echo 3 pulses - 5 echoes](STE_3pulses_5echoes_seq) | [mr0_STE_3pulses_5echoes_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_STE_3pulses_5echoes_seq.ipynb) |
| [FLASH 2D sequence](FLASH_2D_seq) | [mr0_FLASH_2D_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_FLASH_2D_seq.ipynb) |
| [EPI 2D sequence](EPI_2D_seq) | [mr0_EPI_2D_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_EPI_2D_seq.ipynb) |
| [RARE 2D sequence](RARE_2D_seq) | [mr0_RARE_2D_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_RARE_2D_seq.ipynb) |
| [bSSFP 2D sequence](bSSFP_2D_seq) | [mr0_bSSFP_2D_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_bSSFP_2D_seq.ipynb) |
| [Diffusion weighted Gradient Echo](DWI_GRE_2D_seq) | [mr0_DWI_GRE_2D_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_DWI_GRE_2D_seq.ipynb) |
| [Interactive GRE to FLASH](GRE2FLASH_seq) | [mr0_GRE_to_FLASH.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_GRE_to_FLASH.ipynb) |
| [DREAM STE for B0, B1, TxRx mapping](DREAM_STE_seq) | [mr0_DREAM_STE_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_DREAM_STE_seq.ipynb) |
| [DREAM STID for B0, B1, TxRx mapping](DREAM_STID_seq) | [mr0_DREAM_STID_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_DREAM_STID_seq.ipynb) |
| [Burst TSE](burst_TSE_seq) | [mr0_burst_TSE.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_burst_TSE.ipynb) |
| [Compressed Sensing cartesian](mr0_CS_cartesian_seq) | [mr0_CS_cartesian_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_CS_cartesian_seq.ipynb) |
| [Compressed Sensing radial](mr0_CS_radial_seq) | [mr0_CS_radial_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_CS_radial_seq.ipynb) |


## Plot and simulate predifined .seq files

| Sequence | Google Colab |
| -------- | ------------ |
| [Simulate pypulseq example sequences](mr0_pypulseq_example) | [mr0_pypulseq_exmpls_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_pypulseq_exmpls_seq.ipynb) |
| [Simulate own uploaded seq files](mr0_upload_seq) | [mr0_upload_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_upload_seq.ipynb) |


## MR-zero optimization

Gradient descent optimizations using automatic differentiation by backpropagation

| Sequence | Google Colab |
| -------- | ------------ |
| [IR FLASH 2D sequence for T1 mapping using a fit](IR_FLASH_fit) | [mr0_opt_FLASH_2D_IR_Fit_T1.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_opt_FLASH_2D_IR_Fit_T1.ipynb) |
| [IR FLASH 2D sequence for T1 mapping using a NN](IR_FLASH_NN) | [mr0_opt_FLASH_2D_IR_voxelNN_T1.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_opt_FLASH_2D_IR_voxelNN_T1.ipynb) |


## MR-double-zero optimization

Gradient-free optimization with [nevergrad](https://github.com/facebookresearch/nevergrad)

| Sequence | Google Colab |
| -------- | ------------ |
| [Ernst angle optimization](mr00_FLASH_2D_ernstAngle_opt) | [mr00_FLASH_2D_ernstAngle_opt.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr00_FLASH_2D_ernstAngle_opt.ipynb) |


## MR plot wall of fame

famous historic plots recreated


## FLASH (temporary notebooks)

| Sequence | Google Colab |
| -------- | ------------ |
| [Pure `MR0` FLASH](flash) | [flash.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/examples/flash.ipynb) |
| [pulseq FLASH](pulseq_flash) | [pulseq_flash.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/examples/pulseq_flash.ipynb) |
| [pulseq pTx FLASH](pulseq_pTx) | [pulseq_sim_pTx.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/examples/pulseq_sim_pTx.ipynb) |


## Notebook execution results

::::{toggle}

:::{nb-exec-table}
:::

::::
