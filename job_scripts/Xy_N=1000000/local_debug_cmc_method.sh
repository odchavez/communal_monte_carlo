#!/bin/bash

mpirun -np 4 python mpi_emb_par_sim_dat_with_comm.py  --Xy_N 1000000 --Epoch_N 100000 --Nt 1000 --p 100 --N_Node 4 --particles_per_shard 2 --experiment_number 0 --save_history 0 --GP_version 0 --randomize_shards 0 --files_to_process_path synth_data/Xy_N=1000000_Epoch_N=10000_Nt=1000_p=100/GP_version=0 --results_sub_folder synth_data --source_folder synth_data

mpirun -np 4 python mpi_emb_par_sim_dat_with_comm.py  --Xy_N 1000000 --Epoch_N 100000 --Nt 1000 --p 100 --N_Node 4 --particles_per_shard 10 --experiment_number 1 --save_history 0 --GP_version 0 --randomize_shards 0 --files_to_process_path synth_data/Xy_N=1000000_Epoch_N=10000_Nt=1000_p=100/GP_version=0 --results_sub_folder synth_data --source_folder synth_data --global_weighting kernel_weighting

mpirun -np 4 python MPI_LFCMC.py  --Xy_N 1000000 --Epoch_N 100000 --Nt 1000 --p 100 --N_Node 4 --particles_per_shard 10 --experiment_number 2 --save_history 0 --GP_version 0 --randomize_shards 0 --files_to_process_path synth_data/Xy_N=1000000_Epoch_N=10000_Nt=1000_p=100/GP_version=0 --results_sub_folder synth_data --source_folder synth_data


mpirun -np 4 python mpi_emb_par_sim_dat_with_comm.py  --Xy_N 1000000 --Epoch_N 100000 --Nt 1000 --p 100 --N_Node 4 --particles_per_shard 10 --experiment_number 3 --save_history 0 --GP_version 0 --randomize_shards 0 --files_to_process_path synth_data/Xy_N=1000000_Epoch_N=10000_Nt=1000_p=100/GP_version=0 --results_sub_folder synth_data --source_folder synth_data --global_weighting normal_consensus_weighting

