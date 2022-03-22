#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-itk
#SBATCH --time=03-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=12000
#SBATCH --job-name="drl_quadcop_training_ca"
#SBATCH --output=ca.out
#SBATCH --mail-user=ludvigls@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load Anaconda3/2020.07
conda env create -f environment.yml
conda activate SimenAUVtest
pip3 install stable-baselines3
pip3 install tensorboard
python3 train3d.py --exp_id 61


uname -a
