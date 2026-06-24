#!/bin/bash
# 
# Jetson Orin Nano (JetPack 6.1) Environment Setup Script
# 
# This script configures a fresh Jetson device to run the luminance-adaptive
# object detection pipeline. It resolves common ARM64 compilation issues, 
# missing JetPack dependencies, and library version mismatches.

echo "=========================================================="
echo " Starting Jetson Hardware Environment Setup (JetPack 6.1) "
echo "=========================================================="

# 1. Update OS and install system-level Python & OpenCV dependencies
echo -e "\n[1/6] Installing system-level dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libopenblas-dev \
    libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

# 2. Fix the missing cuSPARSELt bug in JetPack 6.1
# JetPack 6.1 ships without the CUDA math libraries required by PyTorch 2.5
echo -e "\n[2/6] Applying NVIDIA cuSPARSELt patch..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
rm cuda-keyring_1.1-1_all.deb

# 3. Create and activate a pristine Python virtual environment
echo -e "\n[3/6] Creating Python virtual environment 'env'..."
python3 -m venv env
source env/bin/activate

# 4. Install PyTorch & Torchvision specific to JetPack 6.1 / CUDA 12.6
echo -e "\n[4/6] Installing NVIDIA-compiled PyTorch 2.5 and Torchvision..."
# Install PyTorch from NVIDIA's official Jetson index
pip3 install torch torchaudio --index-url https://developer.download.nvidia.com/compute/redist/jp/v61
# Install pre-compiled Torchvision 0.20.0 from Jetson AI Lab to skip source compilation
pip3 install torchvision==0.20.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 5. Fix the Numpy ABI bug
# The PyTorch 2.5 wheel relies on modern Numpy 2.x, but Jetson installations
# often default to corrupted C-extensions. We force a clean reinstall of a stable version.
echo -e "\n[5/6] Resolving Numpy ABI compatibility..."
pip3 install --force-reinstall numpy==1.26.4

# 6. Install downstream pipeline dependencies
echo -e "\n[6/6] Installing standard CV and inference packages..."
pip3 install pandas opencv-python ultralytics tqdm

echo "=========================================================="
echo " Setup Complete! "
echo " "
echo " To activate your environment, run: "
echo " $ source env/bin/activate "
echo " "
echo " To verify GPU bridge, run: "
echo " $ python3 -c \"import torch; print('CUDA Ready:', torch.cuda.is_available())\" "
echo "=========================================================="
