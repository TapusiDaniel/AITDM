AITDM – Quick Setup

This project uses Python and a virtual environment (venv). Follow the steps below to set up the environment and install dependencies from requirements.txt.

1) Prerequisites
- Python 3.9+ installed (check with: python3 --version)
- Git installed (optional, for cloning)
- Recommended: upgrade pip
  python3 -m pip install --upgrade pip

2) Create and activate a virtual environment

- Linux / macOS:
  python3 -m venv .venv
  source .venv/bin/activate

- Windows (PowerShell):
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  If you get an execution policy error, open PowerShell as Administrator and run:
    Set-ExecutionPolicy RemoteSigned

After activation, your shell prompt should show the (.venv) prefix.

3) Install dependencies
With the venv activated, run:
  pip install -r requirements.txt

4) Running the project
- Example:
  python src/federated_train.py \
  --num_rounds 30 \
  --local_epochs 2 \
  --clients_per_round 1 \
  --device cuda \
  --save_dir checkpoints/federated

  For python scripts:
  ./scripts/split_dataset.py

5) Deactivate the virtual environment
  deactivate
