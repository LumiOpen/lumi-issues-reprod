# Installs necessary packages and runs the GRPO training script.

source /opt/miniconda3/bin/activate pytorch

export PYTHONUSERBASE=./pythonuserbase
pip install trl tensorboard

accelerate launch train_grpo.py