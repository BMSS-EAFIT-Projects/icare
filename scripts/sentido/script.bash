#!/bin/bash

#SBATCH --chdir=./ # Working directory

#SBATCH --job-name=perplexity # Job name
#SBATCH --mail-type=START,FAIL,END         # Mail notification
#SBATCH --mail-user=afpuertav@eafit.edu.co  # User Email
#SBATCH --output=slurm-serial.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=slurm-serial.%j.err  # Stderr (%j expands to jobId)
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --partition=accel-2        # Partition
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=64G                   # Memory per node             
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --time=5-00:00:0

##### ENVIRONMENT CREATION #####
module load python/3.10_miniconda-23.5.2

##### JOB COMMANDS #### 
pip install pandas xlrd transformers torch

# Ejecutar el script de Python
python /home/afpuertav/mySpace/scripts/sentido/calcularSentido.py