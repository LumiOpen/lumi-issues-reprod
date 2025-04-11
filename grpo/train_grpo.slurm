#!/bin/bash

#SBATCH --job-name=trl_grpo
#SBATCH --time=00-00:30:00
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=480G
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000615
#SBATCH --output=./logs/%x_%j.output
#SBATCH --error=./logs/%x_%j.error

CONTAINER=/scratch/project_462000353/containers/sif_images/rocm-6.2-python-3.10-pytorch-vllm-te-new.sif
SING_BIND="/scratch/project_462000353,/flash/project_462000353,/scratch/project_462000394/containers/for-turkunlp-team,/pfs/lustrep3/scratch/project_462000394/containers/for-turkunlp-team,/usr/lib64"

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export OMP_NUM_THREADS=1

singularity exec -B "$SING_BIND" $CONTAINER ./train_grpo.sh