# Installation

This repository is built in PyTorch 1.8.1 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6).
Follow these intructions

1. Clone our repository
```
git clone https://github.com/swz30/Restormer.git
cd Restormer
```

2. Make conda environment
```
conda create -n pytorch181 python=3.7
conda activate pytorch181
```

3. Install dependencies
```
conda install pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install packaging causal_conv1d mamba_ssm
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm ptflops
pip install seaborn spectral einops kornia timm gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips timm
```

4. Install basicsr
```
python setup.py develop --no_cuda_ext --record install_files.txt
sudo ~/anaconda3/envs/cu118/bin/python setup.py develop --no_cuda_ext --record install_files.txt
```

### Download datasets from Google Drive

To be able to download datasets automatically you would need `go` and `gdrive` installed. 

1. You can install `go` with the following
```
curl -O https://storage.googleapis.com/golang/go1.11.1.linux-amd64.tar.gz
mkdir -p ~/installed
tar -C ~/installed -xzf go1.11.1.linux-amd64.tar.gz
mkdir -p ~/go
echo "export GOPATH=$HOME/go" >> ~/.bashrc
echo "export PATH=$PATH:$HOME/go/bin:$HOME/installed/go/bin" >> ~/.bashrc
```

2. Install `gdrive` using
```
go get github.com/prasmussen/gdrive
```

3. Close current terminal and open a new terminal. 
