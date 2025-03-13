# Onboard Deployment Notes
This note is for RL policy deployment for Jetson Orin series from Unitree's Robot.

### Initial Specification

Install a handy tool for jetson status report
```bash
sudo pip install -U jetson-stats
jetson_release
```

This should report the device specifications
```
Software part of jetson-stats 4.3.1 - (c) 2024, Raffaello Bonghi
Model: NVIDIA Orin NX Developer Kit - Jetpack 5.1.1 [L4T 35.3.1]
NV Power Mode[2]: 15W
Hardware:
 - P-Number: p3767-0000
 - Module: NVIDIA Jetson Orin NX (16GB ram)
Platform:
 - Distribution: Ubuntu 20.04 focal
 - Release: 5.10.104-tegra
jtop:
 - Version: 4.3.1
 - Service: Active
Libraries:
 - CUDA: Not installed
 - cuDNN: Not installed
 - TensorRT: Not installed
 - VPI: Not installed
 - Vulkan: 1.3.204
 - OpenCV: 4.2.0 - with CUDA: NO
```

## Sources

### Change APT Sources

Check the apt sources
```
cat /etc/apt/sources.list
```
If they are from tsinghua mirrors, change them to
```
deb http://ports.ubuntu.com/ubuntu-ports/ focal main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports/ focal-updates main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports/ focal-backports main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports/  focal-security main restricted universe multiverse
```
Then
```
sudo apt update
```

### Change CUDA Sources

Check if this file exists, if not, create one
```shell
cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
```

Find the most recent compatible version of [Jetson sourse](https://repo.download.nvidia.cn/jetson/).
In this case, the system is R35.3 and board is t186. However, the most recent source version for t186 is R32.6, so add the sources like below.
```
deb https://repo.download.nvidia.com/jetson/common r35.3 main
deb https://repo.download.nvidia.com/jetson/t186 r32.6 main
```

Then
```
sudo apt update
```


## Dependencies

1. unitree_sdk2_python
2. CUDA
3. cuDNN
4. PyTorch

### Unitree SDK
Clone `cyclonedds==0.10.x` and install it from source
```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
cd ..
export CYCLONEDDS_HOME="$(pwd)/install"
# I suspect only run the following line is enough
pip install cyclonedds --no-binary cyclonedds
```

Clone `unitree_sdk2_python` and install it from source
```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

### CUDA & cuDNN

Remove any incompatible CUDA version if any
```
sudo apt remove --purge '^cuda.*'
sudo apt remove --purge '^libcudnn.*'
sudo apt autoremove
sudo apt update
```

After update, the apt source should contain only one version of `cuda` and `libcudnn`, and they are fully compatible with the Jetson build now

```bash
sudo apt install cuda-toolkit-11-4
sudo apt install libcudnn8
```

### PyTorch

Download the compatible version of [prebuild PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) based on the Jetpack version (5.1.1 in this case) and the training environment (torch 1 or 2).

```bash
pip install torch-*.whl
```

Verify CUDA is recognized by PyTorch via
```
python -c "import torch; print(torch.cuda.is_available())"
```
