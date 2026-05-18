# GPU Setup

Use this guide on the university GPU machine where the real dataset is stored.

## 1. Connect to the GPU machine

```bash
ssh <your_username>@<gpu_host>
```

If the university uses a scheduler such as Slurm, request an interactive GPU session after login:

```bash
srun --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=04:00:00 --pty bash
```

Use the command required by your university if it differs.

## 2. Clone the repository

```bash
git clone https://github.com/JamshaidAmjad/Cross-Out-Detection-Project-.git
cd Cross-Out-Detection-Project-
```

## 3. Create a Python environment

With `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the GPU cluster uses Conda:

```bash
conda create -n crossout python=3.11 -y
conda activate crossout
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `torch.cuda.is_available()` is false after installation, install PyTorch using the CUDA-specific command recommended for the cluster's CUDA version.

## 4. Verify GPU access

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## 5. Point the project to the real dataset

The current code expects this structure:

```text
data/ImageFolder/
  train/images/<CLASS_NAME>/*.jpg
  val/images/<CLASS_NAME>/*.jpg
  test/images/<CLASS_NAME>/*.jpg
```

If the dataset is elsewhere on the GPU machine, pass it at runtime:

```bash
python scripts/train.py --data-dir /path/to/ImageFolder
```

For the current code, the expected class folders are:

```text
CLEAN
CROSS
DIAGONAL
DOUBLE_LINE
SCRATCH
SINGLE_LINE
WAVE
ZIG_ZAG
MIXED
```

## 6. Smoke test the training entry point

Run a very short job first:

```bash
python scripts/train.py --data-dir /path/to/ImageFolder --epochs 1 --batch-size 8 --freeze-backbone
```

If that works, run the real training jobs:

```bash
python scripts/train.py --data-dir /path/to/ImageFolder --freeze-backbone
python scripts/train.py --data-dir /path/to/ImageFolder --binary --freeze-backbone
```

Checkpoints are saved separately under `checkpoints/multiclass/` and
`checkpoints/binary/`.
