#!/bin/bash
## bash commands for creating gns conda environment

conda create -y -n mgn python=3.9
conda activate mgn
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y pyg -c pyg
conda install -y torch_cluster
conda install -y ipykernel
cd work/bleve-2d
conda install -y --file requirements.txt 
python -m ipykernel install --user --name=mgn