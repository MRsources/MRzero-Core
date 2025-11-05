# Playground

Welcome to Playground MR0, a playground to share, vary and simulate MR sequences.
MR sequences are written in the Pulseq standard using the pypulseq library.
Pulseq files are simulated with the efficient Phase Distribution Graph Bloch simulation.
Here we share links to example colabs that contain various MR sequences or let you upload your own seq file for simulation.

Many of the examples are build using [PyPulseq](https://github.com/imr-framework/pypulseq) and simulate the resulting .seq files with `MR0`.
These .seq files could also be measured on any MRI scanner using a Pulseq interpreter.


## Code and simulate PyPulseq

| Notebook | ![](https://colab.research.google.com/assets/colab-badge.svg) |
| -------- | ------------------------------------------------------------- |
| Free Induction Decay | [mr0_FID_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_FID_seq.ipynb) |
| Spin Echo CPMG | [mr0_SE_CPMG_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_SE_CPMG_seq.ipynb) |
| Stimulated Echo 3 pulses - 5 echoes | [mr0_STE_3pulses_5echoes_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_STE_3pulses_5echoes_seq.ipynb) |
| FLASH 2D sequence | [mr0_FLASH_2D_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_FLASH_2D_seq.ipynb) |
| GRE EPI 2D sequence | [mr0_EPI_2D_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_EPI_2D_seq.ipynb) |
| DWI SE EPI 2D sequence | [mr0_DWI_SE_EPI.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_DWI_SE_EPI.ipynb) |
| Diffusion prepared STEAM | [mr0_diffusion_prep_STEAM_2D_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_diffusion_prep_STEAM_2D_seq.ipynb) |
| RARE 2D sequence | [mr0_RARE_2D_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_RARE_2D_seq.ipynb) |
| TSE 2D sequence | [mr0_TSE_2D_multi_shot_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_TSE_2D_multi_shot_seq.ipynb) |
| Interactive GRE to FLASH | [mr0_GRE_to_FLASH.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_GRE_to_FLASH.ipynb) |
| balanced SSFP sequence | [mr0_bSSFP_2D_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_bSSFP_2D_seq.ipynb) |
| DREAM STE for B0, B1, TxRx mapping | [mr0_DREAM_STE_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_DREAM_STE_seq.ipynb) |
| DREAM STID for B0, B1, TxRx mapping | [mr0_DREAM_STID_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_DREAM_STID_seq.ipynb) |
| Pulseq with RF shimming | [pulseq_rf_shim.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/pulseq_rf_shim.ipynb) |


## Plot and simulate predifined .seq files

| Notebook | ![](https://colab.research.google.com/assets/colab-badge.svg) |
| -------- | ------------------------------------------------------------- |
| Simulate pypulseq example sequences | [mr0_pypulseq_exmpls_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_pypulseq_exmpls_seq.ipynb) |
| Simulate own uploaded seq files | [mr0_upload_seq.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_upload_seq.ipynb) |


## MR-zero optimization

Gradient descent optimizations using automatic differentiation by backpropagation.
Some notebooks use [pulseq-zero](https://github.com/pulseq-frame/pulseq-zero) for optimizable sequence definitions with PyPulseq.

| Notebook | ![](https://colab.research.google.com/assets/colab-badge.svg) |
| -------- | ------------------------------------------------------------- |
| IR FLASH 2D sequence for T1 mapping using a fit | [mr0_opt_FLASH_2D_IR_Fit_T1.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_opt_FLASH_2D_IR_Fit_T1.ipynb) |
| IR FLASH 2D sequence for T1 mapping using a NN | [mr0_opt_FLASH_2D_IR_voxelNN_T1.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr0_opt_FLASH_2D_IR_voxelNN_T1.ipynb) |
| FLASH flip angle opt. for PSF (with pulseq-zero) | [Pulseq_zero_FLASH_FAopt_PSFtask.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/Pulseq_zero_FLASH_FAopt_PSFtask.ipynb) |
| TSE flip angle opt. for SAR (with pulseq-zero) | [Pulseq_zero_TSE_FAopt_SARtask.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/Pulseq_zero_TSE_FAopt_SARtask.ipynb) |
| DESC with pulseq-zero | [pulseq_zero_DESC_demo.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/pulseq_zero_DESC_demo.ipynb) |


## MR-double-zero optimization

Gradient-free optimization with [nevergrad](https://github.com/facebookresearch/nevergrad)

| Notebook | ![](https://colab.research.google.com/assets/colab-badge.svg) |
| -------- | ------------------------------------------------------------- |
| Ernst angle optimization | [mr00_FLASH_2D_ernstAngle_opt.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/mr00_FLASH_2D_ernstAngle_opt.ipynb) |


## MR plot wall of fame

famous historic plots recreated


## MR0 example notebooks

The following sequences are examples of how to realize various tasks in MR-zero rather than demonstrations of specific MRI sequences.

| Notebook | ![](https://colab.research.google.com/assets/colab-badge.svg) |
| -------- | ------------------------------------------------------------- |
| Pure `MR0` FLASH | [flash.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/flash.ipynb) |
| pulseq FLASH | [pulseq_flash.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/pulseq_flash.ipynb) |
| pulseq pTx FLASH | [pulseq_sim_pTx.ipynb](https://colab.research.google.com/github/MRsources/MRzero-Core/blob/main/documentation/playground_mr0/pulseq_sim_pTx.ipynb) |
