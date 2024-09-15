#!/bin/bash
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --job-name=wafer_cgan_job
#SBATCH --time=12:00:00
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/wafer_project/logs/%j/%j_cgan.out
#SBATCH --error=/scratch/general/vast/u1475870/wafer_project/logs/%j/%j_cgan.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL

# Create a unique directory for this job in scratch space
SCRATCH_DIR="/scratch/general/vast/u1475870/wafer_project/"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR

echo "Job started on $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Log directory: $LOG_DIR"

# Set up scratch directory for the job 
cd $SCRATCH_DIR

# Copy the script and any necessary data from home to scratch
cp /uufs/chpc.utah.edu/common/home/$USER/wafer_project/cgan.py .

# Ensure Conda is initialized and in the PATH
export PATH="/uufs/chpc.utah.edu/common/home/$USER/miniconda3/bin:$PATH"

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Activate the Conda environment 
conda activate myenv

# Print Python and Conda info for debugging
which python
python --version
conda info
conda list

# Load required modules
module load cuda/12.1
module load cudnn/8.9.1

# Check GPU availability
nvidia-smi > $LOG_DIR/gpu_info.txt 2>&1
if [ $? -eq 0 ]; then
    echo "GPU is available" >> $LOG_DIR/gpu_info.txt
else
    echo "GPU is not available" >> $LOG_DIR/gpu_info.txt
fi

# Run your Python script
python cgan.py > $LOG_DIR/cgan_output.txt 2>&1

# Deactivate the Conda environment
conda deactivate

# Copy results back to the home directory
cp $SCRATCH_DIR/outputs/wafer_cnn_model.pth /uufs/chpc.utah.edu/common/home/$USER/wafer_project/outputs/
cp $LOG_DIR/cgan_output.txt /uufs/chpc.utah.edu/common/home/$USER/wafer_project/outputs/
cp $LOG_DIR/gpu_info.txt /uufs/chpc.utah.edu/common/home/$USER/wafer_project/outputs/

echo "Job finished on $(date)"

# Optional: Clean up scratch directory
# rm -rf $SCRATCH_DIR