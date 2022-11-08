#!/bin/bash
# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=DWBC2km
#SBATCH --time=48:00:00
# #SBATCH --time=00:19:00
#SBATCH --nodes=4
#SBATCH --tasks-per-node=125
#SBATCH --cpus-per-task=1

# #SBATCH --qos=standard
#SBATCH --qos=long

#SBATCH --account=n01-SiAMOC
#SBATCH --partition=standard

# Setup the job environment (this module needs to be loaded before any other modules)
module load PrgEnv-gnu

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1

# Launch the parallel job
#   srun picks up the distribution from the sbatch options
srun --distribution=block:block --hint=nomultithread ./mitgcmuv
