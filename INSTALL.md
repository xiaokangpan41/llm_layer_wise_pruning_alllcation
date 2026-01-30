# Installation  
## Option A (recommended): uv
Step 1: Create a virtual environment:
```
uv venv -p 3.9
source .venv/bin/activate
```
Step 2: Install dependencies from `pyproject.toml`:
```
uv pip install -e .
```

> Note: Installing CUDA-enabled PyTorch can be platform-specific. If you rely on a specific CUDA toolkit, consider using the conda-based setup below for PyTorch, then run `uv pip install -e .` to install the remaining Python deps.

## Option B: conda (legacy)
Step 1: Create a new conda environment:
```
conda create -n prune_llm python=3.9
conda activate prune_llm
```
Step 2: Install relevant packages
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install transformers==4.28.0 datasets==2.11.0 wandb sentencepiece
pip install accelerate==0.18.0
```
