#!/bin/bash
mode="$1"

pip install pypulseq &> /dev/null
pip install pydisseqt
pip install MRzeroCore --no-deps
pip install ismrmrd
pip install PyWavelets
pip install tqdm
pip install scikit-image

if [ "$mode" == "test" ]; then
    pip install  torchkbnufft
else
    pip install  torchkbnufft --no-deps
fi


wget https://github.com/MRsources/MRzero-Core/raw/main/documentation/playground_mr0/numerical_brain_cropped.mat &> /dev/null