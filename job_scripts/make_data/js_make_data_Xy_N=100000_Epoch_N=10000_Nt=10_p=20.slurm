#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede2 KNL nodes
#
#   *** MPI Job on Normal Queue ***
#
# Last revised: 20 Oct 2017
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch file_name.slurm" on Stampede2 login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do not use mpirun or mpiexec.
#
#   -- Max recommended MPI tasks per KNL node: 64-68
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      fewer tasks per node to give each task more memory.
#
#----------------------------------------------------

#SBATCH -J make_synth_data       # Job name
#SBATCH -o exp_out/make_synth_data.o%j # Name of stdout output file
#SBATCH -e exp_err/make_synth_data.e%j # Name of stderr error file
#SBATCH -p normal          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes
#SBATCH -n 1               # Total # of mpi tasks
#SBATCH -t 5:00:00         # Run time (hh:mm:ss)
#SBATCH --mail-user=ochavez@utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...

# module list
# pwd
# date

# Launch MPI code...

ibrun python run_simulate_data.py --N_total 100000 --Epoch_N 10000 --predictor_N 20 # Use ibrun instead of mpirun or mpiexec

# ---------------------------------------------------
