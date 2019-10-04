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


start_time=time.time()


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
    parser.add_argument(
        '--plot_at_end', type=int,
        help='number of files to process if running a test',
        required=False, default=0
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

files_to_process = min(args.test_run, int(args.Xy_N)/int(args.Epoch_N))
for fn in tqdm(range(files_to_process)):

    folder = (
        '/Xy_N=' + args.Xy_N +
        '_Epoch_N=' + args.Epoch_N +
        '_Nt=' + args.Nt +
        '_p=' + args.p
    )
    experiment_path = 'experiment_results/synth_data' + folder
    data_path = 'synth_data' + folder + '/fn='+ str(fn) + '.csv'

    params_results_file = '/params_resutls_' + args.experiment_number

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
        name_stem  = comm.bcast(name_stem_orig, root=0)

    if exists:
        run_number+=1
        shard_pfo.update_data(shard_data, run_number)

        particle_filter_run_time -= time.time()
        shard_pfo.run_particle_filter()
        particle_filter_run_time +=time.time()
        
        if args.keep_history:
            shard_pfo.write_bo_list(name_stem.code)
        
        shard_pfo.collect_params()
        comm_time_gather_particles-=time.time()
        all_shard_params = comm.gather(shard_pfo.params_to_ship, root=0)
        comm_time_gather_particles+=time.time()
        if rank == 0:
            shuffled_particles = embarrassingly_parallel.shuffel_embarrassingly_parallel_params(
                all_shard_params
            )
            print("shuffled_particles = ", shuffled_particles)
            print("*****************")
            print("*****************")
            print("*****************")
            print("*****************")
            print("*****************")
            print("*****************")
            print("*****************")
            print("*****************")
            print("*****************")
            print(float_valued_particles)
        else:
            shuffled_particles = None
        comm_time_scatter_particles-=time.time()
        post_shuffle_params = comm.scatter(shuffled_particles, root=0)
        comm_time_scatter_particles+=time.time()
        shard_pfo.update_params(post_shuffle_params)

    # preparing output

particle_filter_run_time_all    = str(comm.gather(particle_filter_run_time, root=0))
comm_time_scatter_data_all      = str(comm.gather(comm_time_scatter_data, root=0))
comm_time_gather_particles_all  = str(comm.gather(comm_time_gather_particles, root=0))
comm_time_scatter_particles_all = str(comm.gather(comm_time_scatter_particles, root=0))
    
if rank == 0:
    print("shuffled_particles = ", shuffled_particles)
    float_valued_particles = embarrassingly_parallel.convert_to_list_of_type(shuffled_particles, f_type = float)
    print("*****************")
    print("*****************")
    print("*****************")
    print("*****************")
    print("*****************")
    print("*****************")
    print("*****************")
    print("*****************")
    print("*****************")
    print('float_valued_particles = ', float_valued_particles)
    
    stats_results_file = pd.DataFrame(
        {
            'shards'                     : [size],
            'Xy_N='                      : [args.Xy_N],
            'Epoch_N='                   : [args.Epoch_N],
            'N_Node='                    : [args.N_Node],
            'Nt='                        : [args.Nt],
            'p='                         : [args.p],
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
            'final_params'               : [str(float_valued_particles)]
        }
    )
    parameter_history_obj = history.parameter_history()
    parameter_history_obj.write_stats_results(f_stats_df=stats_results_file)

    if args.plot_at_end:
        parameter_history_obj.compile_bo_list_history(name_stem.code)

        parmas_shape = parameter_history_obj.bo_list_history.shape
        print("parmas_shape=",parmas_shape)
        parmas_truth = pd.read_csv(
            'synth_data/Xy_N=' + args.Xy_N +
            '_Epoch_N=' + args.Epoch_N +
            '_Nt=' + args.Nt +
            '_p=' + args.p + '/Beta_t.csv',
            index_col=0
        ).iloc[:parmas_shape[0],:parmas_shape[1]]
        print("parmas_truth.shape=", parmas_truth.shape)
        print("type(shard_data['predictors'])=", type(shard_data['predictors']))
        print("shard_data['predictors']=", shard_data['predictors'])
        embarrassingly_parallel.plot_CMC_parameter_path_(
            parameter_history_obj.bo_list_history,
            shard_data['predictors'],
            parmas_truth
        )
