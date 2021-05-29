#!/bin/bash

mpirun -np 4 python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.001 --num_obs 100000000000000 --method_type regression --max_time_in_data 999999 --experiment_number 0 --save_history 0 --files_to_process_path synth_data/regression/Xy_N=1000000_Epoch_N=1000_Nt=1_p=32/GP_version=0 --particles_per_shard 10000 --comm_frequency 100000

mpirun -np 4 python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.001 --num_obs 100000000000000 --method_type regression --max_time_in_data 999999 --experiment_number 0 --save_history 0 --files_to_process_path synth_data/regression/Xy_N=1000000_Epoch_N=1000_Nt=1_p=32/GP_version=1 --particles_per_shard 10000 --comm_frequency 100000


mpirun -np 4 python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.001 --num_obs 100000000000000 --method_type regression --max_time_in_data 999999 --experiment_number 0 --save_history 0 --files_to_process_path synth_data/regression/Xy_N=1000000_Epoch_N=1000_Nt=1_p=32/GP_version=0 --particles_per_shard 10000 --comm_frequency 1000000

mpirun -np 4 python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.001 --num_obs 100000000000000 --method_type regression --max_time_in_data 999999 --experiment_number 0 --save_history 0 --files_to_process_path synth_data/regression/Xy_N=1000000_Epoch_N=1000_Nt=1_p=32/GP_version=1 --particles_per_shard 10000 --comm_frequency 1000000