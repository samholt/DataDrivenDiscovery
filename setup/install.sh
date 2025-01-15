#!/bin/bash
conda create --name pem python=3.9.7
conda activate pem
pip3 install torch torchvision torchaudio
pip install --upgrade pip
pip install hydra-core --upgrade
# https://github.com/google/jax#installation
pip install -r requirements.txt
pip install hydra-core --upgrade
pip install evosax
pip install -r requirements.txt
sudo apt-get install xvfb
sudo apt-get install python-opengl