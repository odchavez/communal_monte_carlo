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

#SBATCH -J         airline_N_Node=10     # Job name
#SBATCH -o exp_out/airline_N_Node=10.o%j # stdout output file
#SBATCH -e exp_err/airline_N_Node=10.e%j # stderr error file
#SBATCH -p normal                                                # Queue (partition) name
#SBATCH -N 1                                                     # Total # of nodes
#SBATCH -n 20                                                    # Total # of mpi tasks
#SBATCH -t 40:00:00 #4:00:00                                               # Run time (hh:mm:ss)
#SBATCH --mail-user=ochavez@utexas.edu
#SBATCH --mail-type=all                                          # Send email at begin and end of job

# Other commands must follow all #SBATCH directives...

#module list
#pwd
#date

# Launch MPI code...
# Use ibrun instead of mpirun or mpiexec

#ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --#method_type classification --max_time_in_data 525598 --experiment_number 0 --save_history 0 --files_to_process_path #data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 1 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 2 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 3 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 4 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 5 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 6 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 7 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 8 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 9 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 10 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 11 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 12 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 13 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 14 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 15 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 16 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 17 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 18 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100

ibrun python MPI_pf_airline.py --stationary_prior_mean 0 --stationary_prior_std 1.5 --stepsize 0.01 --num_obs 1000000 --method_type classification --max_time_in_data 525598 --experiment_number 19 --save_history 0 --files_to_process_path data/hensman_t_in_min --particles_per_shard 1000 --comm_frequency 100
