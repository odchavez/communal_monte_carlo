"""
** fix seed to catch error
** post random walk step, intorduce division to makevariance = 1 (sqrt(1+ t*sigma^2)), ie. N(0,1) - DONE
** add functionality to make epocks be more than every 'day' - NOT GOING TO DO
** keep track of time since last flight on pf to not allow too much drift between flights
** email anonio about his arrival
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
from  code_maker import randomString

import particle_filter
import embarrassingly_parallel
import prep_simulation_data
import history
import params
import pf_plots

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

start_time=time.time()


def get_args():
    parser = argparse.ArgumentParser(
        description='Runs the synthetic data generator.'
    )
    parser.add_argument(
        '--N_Node', type=str,
        help='The number of compute nodes allocated.',
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
        help='keep a record of sampled particles if 1.  If 0 do not track history.',
        required=False,  default=0
    )
    parser.add_argument(
        '--experiment_number', type=str,
        help='The experimental run number',
        required=True
    )
    parser.add_argument(
        '--test_run', type=int,
        help='number of files to process if running a test',
        required=False, default=999999999999
    )
    return parser.parse_args()


args = get_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

######################################################
first_time = True
run_number = -1
params_obj = params.pf_params_synth_data( size, args.particles_per_shard, args.p_to_use )

particle_filter_run_time   = 0
comm_time_scatter_data     = 0
comm_time_gather_particles  = 0
comm_time_scatter_particles = 0

year_month_files = [
     'X_2012_November'
    ,'X_2012_December'
    ,'X_2013_January'
    ,'X_2013_February'
    ,'X_2013_March'
    ,'X_2013_April'
    ,'X_2013_May'
    ,'X_2013_June'
    ,'X_2013_July'
    ,'X_2013_August'
    ,'X_2013_September'
    ,'X_2013_October'
]

for fn in tqdm(range(len(year_month_files))):
    file_stem = year_month_files[fn]
    data_path = 'data/' + file_stem + '.csv'
    params_results_file_path = (
        'experiment_results/airline_no_comm' + 
        '_shard_num=' + str(size) +
        '_' + file_stem + 
        '_part_num=' + str(args.particles_per_shard) +
        '_exp_num=' + args.experiment_number + 
        '.csv'
    )

    exists = os.path.isfile(data_path)
    if rank == 0:

        if exists:
            data_obj = prep_simulation_data.prep_data(params_obj.get_params(), data_path)
            to_scatter = data_obj.format_for_scatter(epoch=0)

        else:
            to_scatter = None
            next
    else:
        to_scatter = None
    #print("#SCATTER DATA")
    if exists:
        comm_time_scatter_data -= time.time()
        shard_data = comm.scatter(to_scatter, root=0)
        comm_time_scatter_data += time.time()

    #print("#INITIALIZE PARTICLES IN FIRST PASS WITH DATA")
    if first_time and exists:
        first_time = False

        shard_pfo = particle_filter.particle_filter(
            shard_data,
            params_obj,
            rank,
            run_number
        )

        if rank == 0:
            name_stem_orig = randomString(20)
        else:
            name_stem_orig = None
        name_stem = comm.bcast(name_stem_orig, root=0)

    if exists:
        run_number+=1
        shard_pfo.update_data(shard_data, run_number)

        particle_filter_run_time -= time.time()
        shard_pfo.run_particle_filter()
        particle_filter_run_time +=time.time()
        
        if args.keep_history:
            shard_pfo.write_bo_list(name_stem.code)
        
        shard_pfo.collect_params()
        shard_pfo.collect_history_ids()
        comm_time_gather_particles-=time.time()
        all_shard_params = comm.gather(shard_pfo.params_to_ship, root=0)
        all_shard_particle_history_ids = comm.gather(shard_pfo.particle_history_ids_to_ship, root=0)
        all_shard_machine_history_ids  = comm.gather(shard_pfo.machine_history_ids_to_ship, root=0)
        comm_time_gather_particles+=time.time()
        if rank == 0:
            
            stats_results_file = pd.DataFrame(
                {
                    'shards'                     : [size],
                    #'Xy_N='                      : [args.Xy_N],
                    #'Epoch_N='                   : [args.Epoch_N],
                    'N_Node='                    : [args.N_Node],
                    #'Nt='                        : [args.Nt],
                    #'p='                         : [args.p],
                    'exp_number'                 : [args.experiment_number],
                    'data_type'                  : ["synthetic"],
                    'particle_number'            : params_obj.get_particles_per_shard(),
                    'particle_filter_run_time'   : [999],
                    'comm_time_scatter_data'     : [999],
                    'comm_time_gather_particles' : [999],
                    'comm_time_scatter_particles': [999],
                    'start_time'                 : [start_time],
                    'end_time'                   : [time.time()],
                    'code'                       : [name_stem.code],
                    'final_params'               : [str(all_shard_params)],
                    'run_number'                 : [run_number],
                    'pre_shuffel_params'         : [str(all_shard_params)],
                    'machine_history_ids'        : [str(all_shard_machine_history_ids)],
                    'particle_history_ids'       : [str(all_shard_particle_history_ids)],
                }
            )
            
            #record particles from all shards rather than shuffle to assess fit
            shuffled_particles , shuffled_mach_hist_ids, shuffled_part_hist_ids = embarrassingly_parallel.shuffel_embarrassingly_parallel_params(
                all_shard_params, 
                all_shard_machine_history_ids, 
                all_shard_particle_history_ids
            )
            output_shuffled_particles = embarrassingly_parallel.convert_to_list_of_type(shuffled_particles)
            #stats_results_file.post_shuffel_params=[str(output_shuffled_particles)]
            #stats_results_file.post_machine_history_ids=[str(shuffled_mach_hist_ids)]
            #stats_results_file.post_particle_history_ids=[str(shuffled_part_hist_ids)]
            
            parameter_history_obj = history.parameter_history()
            parameter_history_obj.write_stats_results(
                f_stats_df=stats_results_file, 
                f_other_stats_file=params_results_file_path
            )
        else:
            shuffled_particles = None
        comm_time_scatter_particles-=time.time()
        post_shuffle_params = comm.scatter(all_shard_params, root=0)
        comm_time_scatter_particles+=time.time()
        #shard_pfo.update_params(post_shuffle_params)

# preparing output
post_shuffle_params = comm.scatter(shuffled_particles, root=0)

particle_filter_run_time_all    = str(comm.gather(particle_filter_run_time, root=0))
comm_time_scatter_data_all      = str(comm.gather(comm_time_scatter_data, root=0))
comm_time_gather_particles_all  = str(comm.gather(comm_time_gather_particles, root=0))
comm_time_scatter_particles_all = str(comm.gather(comm_time_scatter_particles, root=0))
    
if rank == 0:
    #float_valued_particles = embarrassingly_parallel.convert_to_list_of_type(shuffled_particles, f_type = float)
    
    stats_results_file = pd.DataFrame(
        {
            'shards'                     : [size],
            #'Xy_N='                      : [args.Xy_N],
            #'Epoch_N='                   : [args.Epoch_N],
            'N_Node='                    : [args.N_Node],
            #'Nt='                        : [args.Nt],
            #'p='                         : [args.p],
            'exp_number'                 : [args.experiment_number],
            'data_type'                  : ["synthetic"],
            'particle_number'            : params_obj.get_particles_per_shard(),
            'particle_filter_run_time'   : [particle_filter_run_time_all],
            'comm_time_scatter_data'     : [comm_time_scatter_data_all],
            'comm_time_gather_particles' : [comm_time_gather_particles_all],
            'comm_time_scatter_particles': [comm_time_scatter_particles_all],
            'start_time'                 : [start_time],
            'end_time'                   : [time.time()],
            'code'                       : [name_stem.code],
            'final_params'               : [str(output_shuffled_particles)],
            'run_number'                 : [run_number],
            'pre_shuffel_params'         : [str(all_shard_params)],
            'machine_history_ids'        : [str(all_shard_machine_history_ids)],
            'particle_history_ids'       : [str(all_shard_particle_history_ids)],
        }
    )
    parameter_history_obj = history.parameter_history()
    parameter_history_obj.write_stats_results(
        f_stats_df=stats_results_file,
        f_other_stats_file=params_results_file_path
    )

