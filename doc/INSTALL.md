# Installation Guide for LMG Model Training

## 1. Create the Conda Environment

```bash
conda create --name meshsplat python=3.8
conda activate meshsplat
```

## 2. Configure CUDA (tested with CUDA 11.7) and PyTorch

We suggest using this way to setup CUDA environment.

Ensure CUDA 11.7 is already installed:

```
cat /usr/local/cuda-*
```

Set environment variables:

```
conda env config vars set CUDA_HOME=/usr/local/cuda-11.7
conda env config vars set PATH=/usr/local/cuda-11.7/bin:$PATH
conda env config vars set LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
conda deactivate
conda activate meshsplat
```

Verify installation:

```
nvcc -V
```

Install PyTorch:

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

Check CUDA availability:

```
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

source: [Cuda and PyTorch Setup Guide \| SYJINTW](https://syjintw.github.io/posts/cuda-and-pytorch/)

## 3. Install Dependencies

```
pip install -r requirements.txt
```

## 4. Setup Submodules

```
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

## 5. Build and Install PyTorch3D

```
mkdir ext
cd ext
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
```
