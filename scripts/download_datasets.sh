#!/usr/bin/env bash
# Download the datasets used for training/evaluating.

pip freeze | grep -q "^gdown" || pip install gdown
gdown "1alE33S1GmP5wACMXaLu50rDIoVzBM4ik"
unzip -n -j all_six_datasets.zip -d dataset
rm dataset/.[!.]*
rm all_six_datasets.zip
