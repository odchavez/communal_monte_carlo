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
import params
import pf_plots 

np.set_printoptions(threshold=np.nan)

#file for testing
#yr=str(1987)
#mo=str(10)
#dom=str(23)
#path = 'Xdata_dom/X'+yr+'mo'+mo+'dom'+dom+'.csv.bz2'
#############

if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print('size is ' + str(size))
    #params = params.pf_params( size )
    
    ######################################################
    first_time = True
    params_obj = params.pf_params( size )
    for yr in [1987]:#range(1988, 2009):
        for mo in [11]:#range(1,13):
            for dom in [24]:#range(1,32):
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
                        params_obj.get_sample_method()
                    )
                    print('initializing complete for shard ' + str(rank) )
                
                #
                if exists:
                    print("updating data on " + str(rank))
                    print("shard "+ str(rank) +" dimention = " + str(shard_data['X_matrix'].shape) )
                    shard_pfo.update_data( 
                        dat_X_matrix = shard_data['X_matrix'], 
                        dat_Y = shard_data['Y']
                    )
                    print("running particle filter on " + str(rank))
                    
                    shard_pfo.run_particle_filter()
    
                    #Gather Work from Worders
                    for pinl in range(len(shard_pfo.particle_list)):
                        shard_pfo.particle_list[pinl].print_max()
                    print('shard_pfo size is:' + str(sys.getsizeof(shard_pfo)))
                    parallel_com_obj = comm.gather(shard_pfo,root=0)
                    
                    if rank == 0:
                        parallel_com_obj = embarrassingly_parallel.shuffel_embarrassingly_parallel_particles(
                            data_obj.get_data(),
                            params_obj.get_particles_per_shard(),
                            parallel_com_obj,
                        )
    
                
                
                
    ######################################################
    
    ##preparing data on Master
    #if rank == 0:
    #    print("master setting up...")
    #    data_obj = prep_airline_data.prep_data(params.get_params(), path)
    #    print("data_obj.keys()=", data_obj.get_data().keys())
    #    to_scatter = data_obj.format_for_scatter(epoch=0)
    #else:
    #    to_scatter = None
    #print('scattering begin')
    #shard_data = comm.scatter(to_scatter, root=0)
    #print('scattering success')
    #
    #
    #print('worker '+ str(rank) +'  has keys:', shard_data['data_keys'][:5])
    #print('shard ' + str(rank) + ' initializing particle filter...')
    ##Run particle Filter on Workers
    #shard_pfo = particle_filter.particle_filter(
    #    shard_data, 
    #    params.get_particles_per_shard(), 
    #    params.get_model(), 
    #    params.get_sample_method()
    #)
    #print('initializing complete for shard ' + str(rank) )
    #shard_pfo.run_particle_filter()
    #
    ##Gather Work from Worders
    #parallel_com_obj = comm.gather(shard_pfo,root=0)
    #
    #if rank == 0:
    #    parallel_com_obj = embarrassingly_parallel.shuffel_embarrassingly_parallel_particles(
    #        data_obj.get_data(),
    #        params.get_particles_per_shard(),
    #        parallel_com_obj,
    #    )
    
    
    if rank == 0:
        #embarrassingly_parallel.plot_CMC_parameter_path_(
        #    pf_obj = parallel_com_obj, 
        #    PART_NUM = params_obj.get_particles_per_shard(), 
        #    number_of_shards = size, 
        #    data = data_obj.get_data(), 
        #    params = params_obj, 
        #    particle_prop=0.5
        #)
        embarrassingly_parallel.plot_CMC_parameter_path_by_shard(
            data = data_obj.get_data(), 
            pf_obj = parallel_com_obj,
            PART_NUM = params_obj.get_particles_per_shard(), 
            number_of_shards = size, 
            particle_prop = 1.0
        )