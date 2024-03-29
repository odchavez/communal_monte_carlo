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

#SBATCH -J         Xy_N=1000000_Epoch_N=10000_N_Node=20_p=32_particles=1000_GP_version=15-19_expn=0     # Job name
#SBATCH -o exp_out/Xy_N=1000000_Epoch_N=10000_N_Node=20_p=32_particles=1000_GP_version=15-19_expn=0.o%j # stdout output file
#SBATCH -e exp_err/Xy_N=1000000_Epoch_N=10000_N_Node=20_p=32_particles=1000_GP_version=15-19_expn=0.e%j # stderr error file
#SBATCH -p normal                                                # Queue (partition) name
#SBATCH -N 1                                                     # Total # of nodes
#SBATCH -n 20                                                    # Total # of mpi tasks
#SBATCH -t 2:00:00                                               # Run time (hh:mm:ss)
#SBATCH --mail-user=ochavez@utexas.edu
#SBATCH --mail-type=all                                          # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...

#module list
#pwd
#date

# Launch MPI code...
# Use ibrun instead of mpirun or mpiexec

ibrun python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.01 --num_obs 100000000000000 --method_type classification --max_time_in_data 999999 --experiment_number 0 --save_history 0 --files_to_process_path synth_data/classification/Xy_N=1000000_Epoch_N=10000_Nt=1_p=32/GP_version= --version 15 --particles_per_shard 1000 --comm_frequency 10000

ibrun python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.01 --num_obs 100000000000000 --method_type classification --max_time_in_data 999999 --experiment_number 0 --save_history 0 --files_to_process_path synth_data/classification/Xy_N=1000000_Epoch_N=10000_Nt=1_p=32/GP_version= --version 16 --particles_per_shard 1000 --comm_frequency 10000

ibrun python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.01 --num_obs 100000000000000 --method_type classification --max_time_in_data 999999 --experiment_number 0 --save_history 0 --files_to_process_path synth_data/classification/Xy_N=1000000_Epoch_N=10000_Nt=1_p=32/GP_version= --version 17 --particles_per_shard 1000 --comm_frequency 10000

ibrun python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.01 --num_obs 100000000000000 --method_type classification --max_time_in_data 999999 --experiment_number 0 --save_history 0 --files_to_process_path synth_data/classification/Xy_N=1000000_Epoch_N=10000_Nt=1_p=32/GP_version= --version 18 --particles_per_shard 1000 --comm_frequency 10000

ibrun python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.01 --num_obs 100000000000000 --method_type classification --max_time_in_data 999999 --experiment_number 0 --save_history 0 --files_to_process_path synth_data/classification/Xy_N=1000000_Epoch_N=10000_Nt=1_p=32/GP_version= --version 19 --particles_per_shard 1000 --comm_frequency 10000
