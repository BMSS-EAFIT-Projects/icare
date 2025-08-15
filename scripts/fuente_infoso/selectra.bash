#!/bin/bash

#SBATCH --chdir=./ # Working directory

#SBATCH --job-name=PFI-selectra # Job name
#SBATCH --mail-type=START,FAIL,END         # Mail notification
#SBATCH --mail-user=afpuertav@eafit.edu.co  # User Email
#SBATCH --output=slurm-serial.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=slurm-serial.%j.err  # Stderr (%j expands to jobId)
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --partition=accel-2        # Partition
#SBATCH --gres=gpu:1                # Number of GPUs per node
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=32G                   # Memory per node             # Number of GPUs
#SBATCH --time=3-12:00:00

##### ENVIRONMENT CREATION #####
module load python/3.10_miniconda-23.5.2

##### JOB COMMANDS #### 
pip install --upgrade pip
pip install pandas tqdm transformers torch openpyxl
pip install pandas tqdm transformers torch datasets openpyxl

python /home/afpuertav/mySpace/scripts/fuente_infoso/selectra.py