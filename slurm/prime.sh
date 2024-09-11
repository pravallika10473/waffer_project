#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=prime_job
#SBATCH --time=02:00:00
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/%j/logs/%j_prime.out
#SBATCH --error=/scratch/general/vast/u1475870/%j/logs/%j_prime.err

# Create a unique directory for this job in scratch space
SCRATCH_DIR="/scratch/general/vast/u1475870/$SLURM_JOB_ID"
LOG_DIR="$SCRATCH_DIR/logs"
mkdir -p $LOG_DIR

echo "Job started on $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Log directory: $LOG_DIR"

# Set up scratch directory for the job
cd $SCRATCH_DIR

# Copy the script and any necessary data from home to scratch
cp /uufs/chpc.utah.edu/common/home/$USER/wafer_project/prime.py .

# Activate the virtual environment
source /uufs/chpc.utah.edu/common/home/$USER/wafer_project/myenv/bin/activate

# Load required modules
module load python
module load cuda/12.4.0
module load cudnn

# Run your Python script
python prime.py > $SCRATCH_DIR/prime_output.txt 2>&1

# Deactivate the virtual environment
deactivate

# Copy results back to the home directory
mkdir -p /uufs/chpc.utah.edu/common/home/$USER/wafer_project/outputs
cp $SCRATCH_DIR/prime.txt /uufs/chpc.utah.edu/common/home/$USER/wafer_project/outputs/

echo "Job finished on $(date)"

# Optional: Clean up scratch directory
# rm -rf $SCRATCH_DIR