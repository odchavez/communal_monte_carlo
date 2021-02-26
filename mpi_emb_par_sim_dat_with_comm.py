"""
RUN WITH:
mpirun -np 3 python mpi_emb_par_sim_dat_with_comm.py  --Xy_N 6000 --Epoch_N 6000 --Nt 30 --p 2 --N_Node 3 --particles_per_shard 20 --experiment_number 999 --save_history 0 --GP_version 0 --randomize_shards 0 --files_to_process_path synth_data/Xy_N=6000_Epoch_N=1200_Nt=30_p=2/GP_version=0 --results_sub_folder synth_data --communicate 1 --source_folder synth_data --global_weighting uniform_weighting
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from mpi4py import MPI
from code_maker import randomString

import particle_filter
import embarrassingly_parallel
import prep_simulation_data
import history
import params
import pf_plots
from files_to_process import files_to_process

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

# Check which version of python is running and make changes accordingly
if (sys.version_info > (3, 0)):
     # Python 3 code in this block
     time.clock = time.process_time

start_time=time.clock()




def get_args():
    parser = argparse.ArgumentParser(
        description='Runs the synthetic data generator.'
    )
    parser.add_argument(
        '--Xy_N', type=str,
        help='Total number of obervations in all files for the particular dataset.',
        required=True
    )
    parser.add_argument(
        '--Epoch_N', type=str,
        help='The number of observations to have per epoch, ie. before a communication step.',
        required=True
    )
    parser.add_argument(
        '--N_Node', type=str,
        help='The number of compute nodes allocated.',
        required=True
    )
    parser.add_argument(
        '--Nt', type=str,
        help='The number of observations with the same timestep.',
        required=True
    )
    parser.add_argument(
        '--p', type=str,
        help='The number of predictor in X.',
        required=True
    )
    parser.add_argument(
        '--p_to_use', type=int,
        help='The number of predictor in X to use.',
        required=False, default=10000
    )
    parser.add_argument(
        '--particles_per_shard', type=int,
        help='The number of particles per shard.',
        required=True
    )
    parser.add_argument(
        '--keep_history', type=int,
        help='keep a full record of sampled particles if 1.  If 0 do not track history.',
        required=False,  default=0
    )
    parser.add_argument(
        '--experiment_number', type=str,
        help='The experimental run number',
        required=True
    )
    parser.add_argument(
        '--GP_version', type=str,
        help='The GP data version number',
        required=False
    )
    parser.add_argument(
        '--test_run', type=int,
        help='number of files to process if running a test',
        required=False, default=99999999999999999
    )
    parser.add_argument(
        '--plot_at_end', type=int,
        help='number of files to process if running a test',
        required=False, default=0
    )
    parser.add_argument(
        '--save_history', type=int,
        help='save history of particles at the end of each epoch if 1 else if 0 only save the last communication state',
        required=False, default=0
    )
    parser.add_argument(
        '--randomize_shards', type=int,
        help='Randomize which of the shards gets data at a particular time step. If 0 shard order is determined by mpi shard rank number.',
        required=False, default=0
    )
    parser.add_argument(
        '--files_to_process_path', type=str,
        help='path to files containing data to be processed.',
        required=True,
        default="files_to_process/synth_data_Xy_N=6000_Epoch_N=1200_Nt=30_p=2_GP_version=0.py"

    ),
    parser.add_argument(
        '--results_sub_folder', type=str,
        help='subfolder for to store results such as synth_data, airline, etc.',
        required=True,
        default=""

    )
    parser.add_argument(
        '--source_folder', type=str,
        help='subfolder for to store results such as synth_data, data, etc.',
        required=True,
        default=""

    )
    parser.add_argument(
        '--communicate', type=int,
        help='communicate and end of each epoch if 1 else 0 (embarrasingly parrallel).',
        required=False,
        default=1

    )
    parser.add_argument(
        '--global_weighting', type=str,
        choices=['uniform_weighting', 'kernel_weighting', 'normal_consensus_weighting'],
        help='the type of weighting in resampling to use at a global communication.',
        required=False,
        default="uniform_weighting"

    )
    return parser.parse_args()


args = get_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

######################################################
first_time = True
run_number = -1
params_obj = params.pf_params_synth_data( size, args)

particle_filter_run_time    = 0
comm_time_scatter_data      = 0
comm_time_gather_particles  = 0
comm_time_scatter_particles = 0
record_keeping_time         = 0


if rank == 0:
    
    if args.source_folder == "synth_data":
        string = "Xy_N={}_Epoch_N={}_Nt={}_p={}{}GP_version={}"
        file_stem = string.format(args.Xy_N, args.Epoch_N, args.Nt, args.p, '_',args.GP_version)
    else:
        file_stem = args.source_folder
    print("file_stem = ", file_stem)
    
    epoch_files_to_process = prep_simulation_data.make_epoch_files(
        files_to_process = files_to_process[args.files_to_process_path][:args.test_run],
        data_type = args.source_folder,
        file_stem = file_stem, 
        Epoch_N = int(args.Epoch_N),
    )
    
    
else:
    epoch_files_to_process = None
    
epoch_files_to_process = comm.bcast(epoch_files_to_process, root=0)

for fn in tqdm(range(len(epoch_files_to_process))):
    
    #prep file names needed for loading data and writing results
    if args.results_sub_folder == "synth_data":
        string = "Xy_N={}_Epoch_N={}_Nt={}_p={}{}GP_version={}"
        output_stem=string.format(args.Xy_N, args.Epoch_N, args.Nt, args.p, '_',args.GP_version)
    else:
        output_stem = args.results_sub_folder
        
    params_results_file_path = (
        'experiment_results/' + 
         args.results_sub_folder + 
        '/results_emb_par_fit_test_with_comm' + 
        '_shard_num=' + str(size) +
        '_' + output_stem + 
        '_part_num=' + str(args.particles_per_shard) +
        '_exp_num=' + args.experiment_number + 
        '_communicate=' + str(True if args.communicate == 1 else False) +
        '.csv')
    
    data_path = epoch_files_to_process[fn]
    exists = os.path.isfile(data_path)
    #. determine which indices to keep in each shard
    if rank == 0:
        if exists: 
            print("processing file ", data_path, " out of ", str( len(epoch_files_to_process) ) )
            data_obj = prep_simulation_data.prep_data()
            data_obj.load_new_data(params_obj.get_params(), data_path, shard_subset=None)
            indices_to_scatter = data_obj.partition_index()
        else:
            indicis_to_scatter = None
            next
    else:
        indices_to_scatter = None

    #print("#SCATTER DATA")
    if exists:
        comm_time_scatter_data -= time.clock()
        shard_data_indices = comm.scatter(indices_to_scatter, root=0)
        data_obj = prep_simulation_data.prep_data()
        data_obj.load_new_data(params_obj.get_params(), data_path, shard_data_indices)
        shard_data =  data_obj.get_data()
        comm_time_scatter_data += time.clock()

    if first_time and exists:
        first_time = False

        shard_pfo = particle_filter.particle_filter(
            shard_data,
            params_obj,
            rank,
            run_number
        )

    if exists:
        run_number+=1
        shard_pfo.update_data(shard_data, run_number)

        particle_filter_run_time -= time.clock()
        shard_pfo.run_particle_filter()
        particle_filter_run_time +=time.clock()
        
        if (args.communicate == 1) or (fn == len(epoch_files_to_process)-1):
            comm_time_gather_particles-=time.clock()
            all_shard_params = comm.gather(shard_pfo.particles, root=0)
            if rank == 0:
                shuffled_particles = (
                    embarrassingly_parallel.shuffel_embarrassingly_parallel_params(
                        all_shard_params=all_shard_params, 
                        weighting_type=args.global_weighting))
            else:
                shuffled_particles = None
            comm_time_gather_particles+=time.clock()            
                
            #  scatter particles and update
            comm_time_scatter_particles-=time.clock()
            post_shuffle_params = comm.scatter(shuffled_particles, root=0)
            shard_pfo.update_params(post_shuffle_params)               
            comm_time_scatter_particles+=time.clock()


# preparing output
particle_filter_run_time_all    = str(comm.gather(particle_filter_run_time, root=0))
comm_time_scatter_data_all      = str(comm.gather(comm_time_scatter_data, root=0))
comm_time_gather_particles_all  = str(comm.gather(comm_time_gather_particles, root=0))
comm_time_scatter_particles_all = str(comm.gather(comm_time_scatter_particles, root=0))
    
if rank == 0:
    output_shuffled_particles = embarrassingly_parallel.convert_to_list_of_type(shuffled_particles)
    stats_results_file = pd.DataFrame(
        {
            'shards'                     : [size],
            'Xy_N='                      : [args.Xy_N],
            'Epoch_N='                   : [args.Epoch_N],
            'N_Node='                    : [args.N_Node],
            'Nt='                        : [args.Nt],
            'p='                         : [args.p],
            'exp_number'                 : [args.experiment_number],
            'data_type'                  : [args.results_sub_folder],
            'particle_number'            : params_obj.get_particles_per_shard(),
            'particle_filter_run_time'   : [particle_filter_run_time_all],
            'comm_time_scatter_data'     : [comm_time_scatter_data_all],
            'comm_time_gather_particles' : [comm_time_gather_particles_all],
            'comm_time_scatter_particles': [comm_time_scatter_particles_all],
            'start_time'                 : [start_time],
            'end_time'                   : [time.clock()],
            'final_params'               : [str(output_shuffled_particles)],
            'run_number'                 : [run_number],
        }
    )
    
    parameter_history_obj = history.parameter_history()
    parameter_history_obj.write_stats_results(
        f_stats_df=stats_results_file, 
        f_other_stats_file=params_results_file_path,
        save_history=args.save_history,
    )

if args.save_history == 1:
    all_histories = comm.gather(shard_pfo.history, root=0)
    if rank == 0:
        mean_history = sum(all_histories)/size
        
        history_path = (
        'experiment_results/history/' + 
         args.results_sub_folder + 
        '/results_emb_par_fit_test_with_comm' + 
        '_shard_num=' + str(size) +
        '_' + output_stem + 
        '_part_num=' + str(args.particles_per_shard) +
        '_exp_num=' + args.experiment_number + 
        '_communicate=' + str(True if args.communicate == 1 else False) +
        '.csv')
        
        pd.DataFrame(mean_history).to_csv(history_path)
