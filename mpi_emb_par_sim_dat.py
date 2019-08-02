"""
** fix seed to catch error
** post random walk step, intorduce division to makevariance = 1 (sqrt(1+ t*sigma^2)), ie. N(0,1) - DONE
** add functionality to make epocks be more than every 'day' - NOT GOING TO DO
** keep track of time since last flight on pf to not allow too much drift between flights
** email anonio about his arrival
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from mpi4py import MPI

import particle_filter
#import simulate_data
import embarrassingly_parallel
import prep_simulation_data
import history
import params
import pf_plots 
from  code_maker import randomString

#np.set_printoptions(threshold=np.nan)

#global_name_stem  = randomString(10)

if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print('size is ' + str(size))    
    ######################################################
    first_time = True
    run_number = -1
    params_obj = params.pf_params_synth_data( size )
    
for fn in range(2):
    last_file_attempted = 'synth_data/Xy_N=10000000_Nt=10_p=100_fn='+str(fn)+'.csv'
    print("attempting file: " + last_file_attempted)
    print("# LOAD DATA")
    path = 'synth_data/Xy_N=10000000_Nt=10_p=100_fn='+str(fn)+'.csv'
    exists = os.path.isfile(path)
    if rank == 0:

        if exists:
            print("workin on" +     path)
            data_obj = prep_simulation_data.prep_data(params_obj.get_params(), path)
            to_scatter = data_obj.format_for_scatter(epoch=0)
            
        else:
            print("skiping " + path)
            to_scatter = None
            next
    else:
        to_scatter = None
    print("#SCATTER DATA")
    if exists:
        shard_data = comm.scatter(to_scatter, root=0)
        
    
    print("#INITIALIZE PARTICLES IN FIRST PASS WITH DATA")
    if first_time and exists:
        first_time = False
        print('shard ' + str(rank) + ' initializing particle filter...')
        #Run particle Filter on Workers
        
        shard_pfo = particle_filter.particle_filter(
            shard_data, 
            params_obj,
            rank, 
            run_number
        )
        print('initializing complete for shard ' + str(rank) )
        
        if rank == 0:
            name_stem_orig = randomString(10)
            print("in " + str(rank) + " stem name is " + str(name_stem_orig))
        else:
            name_stem_orig = None
        name_stem  = comm.bcast(name_stem_orig, root=0)
        
    if exists:
        print("updating data on " + str(rank))
        run_number+=1
        shard_pfo.update_data(shard_data, run_number)
        print("running particle filter on " + str(rank))
        
        shard_pfo.run_particle_filter()
        shard_pfo.write_bo_list(name_stem.code)
        shard_pfo.collect_params()
        
        all_shard_params = comm.gather(shard_pfo.params_to_ship, root=0)
        if rank == 0:
            shuffled_partiles = embarrassingly_parallel.shuffel_embarrassingly_parallel_params(
                all_shard_params
            )
        else:
            shuffled_partiles = None
        post_shuffle_params = comm.scatter(shuffled_partiles, root=0)
        shard_pfo.update_params(post_shuffle_params)

    # ploting output
if rank == 0:
    print('initializing history objects')
    parameter_history_obj = history.parameter_history()
    parameter_history_obj.compile_bo_list_history(name_stem.code)
    
    parmas_shape = parameter_history_obj.bo_list_history.shape
    print("parmas_shape=",parmas_shape)
    parmas_truth = pd.read_csv(
        'synth_data/Beta_t.csv', 
        #low_memory=False, 
        index_col=0
    ).iloc[:parmas_shape[0],:parmas_shape[1]]
    print("parmas_truth.shape=", parmas_truth.shape)
    print(parmas_truth.head())
    embarrassingly_parallel.plot_CMC_parameter_path_(
        parameter_history_obj.bo_list_history,
        shard_data['predictors'],
        parmas_truth
    )
