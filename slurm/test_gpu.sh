#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=gpu_test
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1475870/wafer_project/slurm/logs/%j.out-%N
#SBATCH -e /uufs/chpc.utah.edu/common/home/u1475870/wafer_project/slurm/logs/%j.err-%N

# Set up scratch directory
SCRDIR=/scratch/general/vast/$USER/$SLURM_JOB_ID
mkdir -p $SCRDIR
cd $SCRDIR

# Copy the script and any necessary data from home to scratch
cp /uufs/chpc.utah.edu/common/home/$USER/wafer_project/gpu_test.py .

# Load necessary modules
module load python
module load cuda/12.4.0
module load cudnn

# Set environment variables
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4   

# Activate your virtual environment
source /uufs/chpc.utah.edu/common/home/u1475870/wafer_project/myenv/bin/activate

# Run your Python script
python gpu_test.py

# Deactivate the environment (optional)
deactivate

# Copy results back to the original directory if needed
# cp ... /uufs/chpc.utah.edu/common/home/$USER/wafer_project/outputs/
