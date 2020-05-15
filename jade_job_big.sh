#!/bin/bash

# set the number of nodes
#SBATCH --nodes=1

# set name of job
#SBATCH --job-name=vae1

# set number of GPUs
#SBATCH --gres=gpu:4

#Select a partition
#SBATCH --partition=big

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=jehill.parikh@newcastle.ac.uk

module load python3/anaconda
source activate tensorflow2-gpu
python ./model/VAE.py