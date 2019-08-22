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
#np.set_printoptions(threshold=np.nan)

#global_name_stem  = randomString(10)

#if __name__ == '__main__':

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
        '--experiment_number', type=str,
        help='The experimental run number',
        required=True
    )
    return parser.parse_args()

args = get_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#print('size is ' + str(size))    
######################################################
first_time = True
run_number = -1
params_obj = params.pf_params_synth_data( size )
    
particle_filter_run_time   = 0
comm_time_scatter_data     = 0
comm_time_gather_particles  = 0
comm_time_scatter_particles = 0

for fn in tqdm(range(20)):
    #last_file_attempted = 'synth_data/Xy_N=10000000_Epoch_N=1000_Nt=10_p=100/fn='+str(fn)+'.csv'
    #print("attempting file: " + last_file_attempted)
    #print("# LOAD DATA")
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
            #print("workin on" + data_path)
            data_obj = prep_simulation_data.prep_data(params_obj.get_params(), data_path)
            to_scatter = data_obj.format_for_scatter(epoch=0)
            
        else:
            #print("skiping " + data_path)
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
        #print('shard ' + str(rank) + ' initializing particle filter...')
        #Run particle Filter on Workers
        
        shard_pfo = particle_filter.particle_filter(
            shard_data, 
            params_obj,
            rank, 
            run_number
        )
        #print('initializing complete for shard ' + str(rank) )
        
        if rank == 0:
            name_stem_orig = randomString(20)
            #print("in " + str(rank) + " stem name is " + str(name_stem_orig))
        else:
            name_stem_orig = None
        name_stem  = comm.bcast(name_stem_orig, root=0)
        
    if exists:
        #print("updating data on " + str(rank))
        run_number+=1
        shard_pfo.update_data(shard_data, run_number)
        #print("running particle filter on " + str(rank))
        
        particle_filter_run_time -= time.time()
        shard_pfo.run_particle_filter()
        particle_filter_run_time +=time.time()
        
        shard_pfo.write_bo_list(name_stem.code)
        shard_pfo.collect_params()
        
        comm_time_gather_particles-=time.time()
        all_shard_params = comm.gather(shard_pfo.params_to_ship, root=0)
        comm_time_gather_particles+=time.time()
        if rank == 0:
            shuffled_particles = embarrassingly_parallel.shuffel_embarrassingly_parallel_params(
                all_shard_params
            )
        else:
            shuffled_particles = None
        comm_time_scatter_particles-=time.time()
        post_shuffle_params = comm.scatter(shuffled_particles, root=0)
        comm_time_scatter_particles+=time.time()
        shard_pfo.update_params(post_shuffle_params)

    # ploting output

particle_filter_run_time_all    = str(comm.gather(particle_filter_run_time, root=0))
comm_time_scatter_data_all      = str(comm.gather(comm_time_scatter_data, root=0))
comm_time_gather_particles_all  = str(comm.gather(comm_time_gather_particles, root=0))
comm_time_scatter_particles_all = str(comm.gather(comm_time_scatter_particles, root=0))

if rank == 0:
    stats_results_file = pd.DataFrame(
        {
            'shards'                     : [size],
            '/Xy_N='                     : [args.Xy_N],
            '_Epoch_N='                  : [args.Epoch_N],
            '_Nt='                       : [args.Nt],
            '_p='                        : [args.p],
            'exp_number'                 : [args.experiment_number],
            'data_type'                  : ["synthetic"],
            'particle_number'            : params_obj.get_particles_per_shard(),
            'particle_filter_run_time'   : [particle_filter_run_time_all],
            'comm_time_scatter_data'     : [comm_time_scatter_data_all],
            'comm_time_gather_particles' : [comm_time_gather_particles_all],
            'comm_time_scatter_particles': [comm_time_scatter_particles_all],
            'start_time'                 : [start_time],
            'end_time'                   : [time.time()],
        }
    )
    parameter_history_obj = history.parameter_history()
    parameter_history_obj.compile_bo_list_history(name_stem.code)
    
    parameter_history_obj.write_results(
        f_experiment_path = experiment_path,  
        f_params_results_file = params_results_file,
        f_stats_df = stats_results_file,
    )
    
    #parmas_shape = parameter_history_obj.bo_list_history.shape
    #print("parmas_shape=",parmas_shape)
    #parmas_truth = pd.read_csv(
    #    #'synth_data/Xy_N=10000000_Epoch_N=1000_Nt=10_p=100/Beta_t.csv', 
    #     'synth_data/Xy_N=' + args.Xy_N + 
    #    '_Epoch_N=' + args.Epoch_N +
    #    '_Nt=' + args.Nt +
    #    '_p=' + args.p + '/Beta_t.csv',
    #    #low_memory=False, 
    #    index_col=0
    #).iloc[:parmas_shape[0],:parmas_shape[1]]
    #print("parmas_truth.shape=", parmas_truth.shape)
    ##print(parmas_truth.head())
    #print("type(shard_data['predictors'])=", type(shard_data['predictors']))
    #print("shard_data['predictors']=", shard_data['predictors'])
    #embarrassingly_parallel.plot_CMC_parameter_path_(
    #    parameter_history_obj.bo_list_history,
    #    shard_data['predictors'],
    #    parmas_truth
    #)
    