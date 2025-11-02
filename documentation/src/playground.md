# Playground

Welcome to Playground MR0, a playground to share, vary and simulate MR sequences.
MR sequences are written in the Pulseq standard using the pypulseq library.
Pulseq files are simulated with the efficient Phase Distribution Graph Bloch simulation.
Here we share links to example colabs that contain various MR sequences or let you upload your own seq file for simulation.

Many of the examples are build using [PyPulseq](https://github.com/imr-framework/pypulseq) and simulate the resulting .seq files with `MR0`.
These .seq files could also be measured on any MRI scanner using a Pulseq interpreter.


## Code and simulate PyPulseq

| Notebook | {{#include snippets.txt:colab_button}} |
| -------- | -------------------------------------- |
| Free Induction Decay | [mr0_FID_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_FID_seq.ipynb) |
| Spin Echo CPMG | [mr0_SE_CPMG_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_SE_CPMG_seq.ipynb) |
| Stimulated Echo 3 pulses - 5 echoes | [mr0_STE_3pulses_5echoes_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_STE_3pulses_5echoes_seq.ipynb) |
| FLASH 2D sequence | [mr0_FLASH_2D_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_FLASH_2D_seq.ipynb) |
| GRE EPI 2D sequence | [mr0_EPI_2D_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_EPI_2D_seq.ipynb) |
| DWI SE EPI 2D sequence | [mr0_DWI_SE_EPI.ipynb]({{#include snippets.txt:colab_url}}/mr0_DWI_SE_EPI.ipynb) |
| Diffusion prepared STEAM | [mr0_diffusion_prep_STEAM_2D_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_diffusion_prep_STEAM_2D_seq.ipynb) |
| RARE 2D sequence | [mr0_RARE_2D_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_RARE_2D_seq.ipynb) |
| TSE 2D sequence | [mr0_TSE_2D_multi_shot_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_TSE_2D_multi_shot_seq.ipynb) |
| Interactive GRE to FLASH | [mr0_GRE_to_FLASH.ipynb]({{#include snippets.txt:colab_url}}/mr0_GRE_to_FLASH.ipynb) |
| balanced SSFP sequence | [mr0_bSSFP_2D_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_bSSFP_2D_seq.ipynb) |
| DREAM STE for B0, B1, TxRx mapping | [mr0_DREAM_STE_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_DREAM_STE_seq.ipynb) |
| DREAM STID for B0, B1, TxRx mapping | [mr0_DREAM_STID_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_DREAM_STID_seq.ipynb) |
| Pulseq with RF shimming | [pulseq_rf_shim.ipynb]({{#include snippets.txt:colab_url}}/pulseq_rf_shim.ipynb) |


## Plot and simulate predifined .seq files

| Notebook | {{#include snippets.txt:colab_button}} |
| -------- | -------------------------------------- |
| Simulate pypulseq example sequences | [mr0_pypulseq_exmpls_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_pypulseq_exmpls_seq.ipynb) |
| Simulate own uploaded seq files | [mr0_upload_seq.ipynb]({{#include snippets.txt:colab_url}}/mr0_upload_seq.ipynb) |


## MR-zero optimization

Gradient descent optimizations using automatic differentiation by backpropagation.
Some notebooks use [pulseq-zero](https://github.com/pulseq-frame/pulseq-zero) for optimizable sequence definitions with PyPulseq.

| Notebook | {{#include snippets.txt:colab_button}} |
| -------- | -------------------------------------- |
| IR FLASH 2D sequence for T1 mapping using a fit | [mr0_opt_FLASH_2D_IR_Fit_T1.ipynb]({{#include snippets.txt:colab_url}}/mr0_opt_FLASH_2D_IR_Fit_T1.ipynb) |
| IR FLASH 2D sequence for T1 mapping using a NN | [mr0_opt_FLASH_2D_IR_voxelNN_T1.ipynb]({{#include snippets.txt:colab_url}}/mr0_opt_FLASH_2D_IR_voxelNN_T1.ipynb) |
| FLASH flip angle opt. for PSF (with pulseq-zero) | [Pulseq_zero_FLASH_FAopt_PSFtask.ipynb]({{#include snippets.txt:colab_url}}/Pulseq_zero_FLASH_FAopt_PSFtask.ipynb) |
| TSE flip angle opt. for SAR (with pulseq-zero) | [Pulseq_zero_TSE_FAopt_SARtask.ipynb]({{#include snippets.txt:colab_url}}/Pulseq_zero_TSE_FAopt_SARtask.ipynb) |
| DESC with pulseq-zero | [pulseq_zero_DESC_demo.ipynb]({{#include snippets.txt:colab_url}}/pulseq_zero_DESC_demo.ipynb) |


## MR-double-zero optimization

Gradient-free optimization with [nevergrad](https://github.com/facebookresearch/nevergrad)

| Notebook | {{#include snippets.txt:colab_button}} |
| -------- | -------------------------------------- |
| Ernst angle optimization | [mr00_FLASH_2D_ernstAngle_opt.ipynb]({{#include snippets.txt:colab_url}}/mr00_FLASH_2D_ernstAngle_opt.ipynb) |


## MR plot wall of fame

famous historic plots recreated


## MR0 example notebooks

The following sequences are examples of how to realize various tasks in MR-zero rather than demonstrations of specific MRI sequences.

| Notebook | {{#include snippets.txt:colab_button}} |
| -------- | -------------------------------------- |
| Pure `MR0` FLASH | [flash.ipynb]({{#include snippets.txt:colab_url}}/flash.ipynb) |
| pulseq FLASH | [pulseq_flash.ipynb]({{#include snippets.txt:colab_url}}/pulseq_flash.ipynb) |
| pulseq pTx FLASH | [pulseq_sim_pTx.ipynb]({{#include snippets.txt:colab_url}}/pulseq_sim_pTx.ipynb) |
