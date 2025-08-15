#!/bin/bash

#SBATCH --chdir=./ # Working directory

#SBATCH --job-name=dec-perc # Job name
#SBATCH --mail-type=START,FAIL,END         # Mail notification
#SBATCH --mail-user=afpuertav@eafit.edu.co  # User Email
#SBATCH --output=slurm-serial.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=slurm-serial.%j.err  # Stderr (%j expands to jobId)
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --partition=accel        # Partition
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=32G                   # Memory per node             
#SBATCH --gres=gpu:1                # Number of GPUs
#SBATCH --time=5-00:00:0

##### ENVIRONMENT CREATION #####
module load python/3.10_miniconda-23.5.2

export HF_HOME=/home/afpuertav/.cache/huggingface

##### JOB COMMANDS #### 
pip install --upgrade pip

pip install pysentimiento pandas openpyxl

pip install sentencepiece
pip install pandas xlrd
pip install openpyxl
pip install transformers
pip install tiktoken protobuf


# Ejecutar el script de Python
python /home/afpuertav/mySpace/scripts/decretos/tweetsDecretos.py