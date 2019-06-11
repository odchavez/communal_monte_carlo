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
import simulate_data
import embarrassingly_parallel
import prep_airline_data
import history
import params
import pf_plots 
from  code_maker import randomString

np.set_printoptions(threshold=np.nan)

#global_name_stem  = randomString(10)

if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print('size is ' + str(size))    
    ######################################################
    first_time = True
    run_number = -1
    params_obj = params.pf_params( size )
    
    for yr in [1987]:#range(1988, 2009):
        for mo in [10]:#range(1,13):
            for dom in [25, 26,]:# 27,28,29,30]:#range(1,32):
                print("attempting file: "+'Xdata_dom/X'+str(yr)+'mo'+str(mo)+'dom'+str(dom)+'.csv.bz2')
                # LOAD DATA
                path = 'Xdata_dom/X'+str(yr)+'mo'+str(mo)+'dom'+str(dom)+'.csv.bz2'
                exists = os.path.isfile(path)
                if rank == 0:

                    if exists:
                        print("workin on" +     path)
                        data_obj = prep_airline_data.prep_data(params_obj.get_params(), path)
                        to_scatter = data_obj.format_for_scatter(epoch=0)
                        
                    else:
                        print("skiping " + path)
                        to_scatter = None
                        next
                else:
                    to_scatter = None
                #SCATTER DATA
                if exists:
                    shard_data = comm.scatter(to_scatter, root=0)
                    
                
                #INITIALIZE PARTICLES IN FIRST PASS WITH DATA
                if first_time and exists:
                    first_time = False
                    print('shard ' + str(rank) + ' initializing particle filter...')
                    #Run particle Filter on Workers
                    
                    shard_pfo = particle_filter.particle_filter(
                        shard_data, 
                        params_obj.get_particles_per_shard(), 
                        params_obj.get_model(), 
                        params_obj.get_sample_method(),
                        rank, 
                        run_number
                    )
                    print('initializing complete for shard ' + str(rank) )
                    
                    if rank == 0:
                        #parameter_history_obj = history.parameter_history()
                        name_stem_orig = randomString(10)
                        #comm.scatter(name_stem_orig, root=0)
                        print("in " + str(rank) + " stem name is " + str(name_stem_orig))
                    else:
                        name_stem_orig = None
                    name_stem  = comm.bcast(name_stem_orig, root=0)
                    
                if exists:
                    print("updating data on " + str(rank))
                    print("shard "+ str(rank) +" dimention = " + str(shard_data['X_matrix'].shape) )
                    run_number+=1
                    shard_pfo.update_data(shard_data, run_number)
                    print("running particle filter on " + str(rank))
                    
                    
                    shard_pfo.run_particle_filter()
                    shard_pfo.write_bo_list(name_stem.code)
                    shard_pfo.collect_params()
                    
                    #parallel_com_obj = comm.gather(shard_pfo, root=0)
                    all_shard_params = comm.gather(shard_pfo.params_to_ship, root=0)
                    
                    if rank == 0:
                        print("all_shard_params type = ", type(all_shard_params))
                        print("all_shard_params len = ", len(all_shard_params))
                        print("all_shard_params[0] type = ", type(all_shard_params[0]))
                        print("all_shard_params[0] len = ", len(all_shard_params[0]))
                        #parallel_com_obj = embarrassingly_parallel.shuffel_embarrassingly_parallel_particles(
                        #    data_obj.get_data(),
                        #    params_obj.get_particles_per_shard(),
                        #    parallel_com_obj,
                        #)
                        shuffled_partiles = embarrassingly_parallel.shuffel_embarrassingly_parallel_params(
                            all_shard_params
                        )
                        print('len(shuffled_partiles) = ' + str(len(shuffled_partiles)))
                        print('len(shuffled_partiles[0]) = ' + str(len(shuffled_partiles[0])))
                    else:
                        shuffled_partiles = None
                    post_shuffle_params = comm.scatter(shuffled_partiles, root=0)
                    shard_pfo.update_params(post_shuffle_params)
                        #parameter_history_obj.append_bo_list_history(
                        #    parallel_com_obj, 
                        #)
                        #parameter_history_obj.append_bo_machine_list_history(parallel_com_obj)
                       # print("******************************************************************************************")
#                        print("parameter_history_obj.bo_list_history len is:", len(parameter_history_obj.bo_list_history))
#                        for k in range( len(parameter_history_obj.bo_list_history)):
#                            print(
#                                "parameter_history_obj.bo_list_history shape is:",
#                                (parameter_history_obj.bo_list_history[k]).shape
#                            )
#                        #print("******************************************************************************************")
                        
    ######################################################
    # ploting output
    if rank == 0:
        print('initializing history objects')
        parameter_history_obj = history.parameter_history()
        parameter_history_obj.compile_bo_list_history(name_stem.code)
        
        embarrassingly_parallel.plot_CMC_parameter_path_(
            parameter_history_obj.bo_list_history,
            shard_data['predictors']
        )
        
        #embarrassingly_parallel.plot_CMC_parameter_path_by_shard(
        #    data = data_obj.get_data(), 
        #    pf_obj = parallel_com_obj,
        #    PART_NUM = params_obj.get_particles_per_shard(), 
        #    number_of_shards = size, 
        #    particle_prop = 1.0
        #)