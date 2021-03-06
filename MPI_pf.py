"""
run with: mpirun -np 4 python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.001 --num_obs 100000000000000 --method_type regression --max_time_in_data 999999 --experiment_number 0 --save_history 1 --files_to_process_path synth_data/regression/Xy_N=1000000_Epoch_N=100000_Nt=1_p=32/GP_version= --version 0 --particles_per_shard 500 --comm_frequency 50000
"""

import os
import sys
import time
import itertools
import pdb
import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as random

from tqdm import tqdm
from mpi4py import MPI

from scipy.special import logsumexp
from scipy import stats

import spf as spf
import files_to_process as ftp



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

PARTICLES_PER_SEND = 1000

# time related items
if (sys.version_info > (3, 0)):
    # Python 3 code in this block
    time.clock = time.process_time
start_time=time.clock()
pf_run_time = 0
comm_time   = 0

def get_args():
    parser = argparse.ArgumentParser(
        description='Runs particle filter using MPI in prallel.'
    )
    parser.add_argument(
        '--stationary_prior_mean', type=float,
        help='stationary prior mean - shold be zero.',
        required=True
    )
    parser.add_argument(
        '--stationary_prior_std', type=float,
        help='stationary prior standard deviation',
        required=True
    )
    parser.add_argument(
        '--stepsize', type=float,
        help='size of stepsize between timesteps of GP coefficients',
        required=True
    )
    parser.add_argument(
        '--num_obs', type=int,
        help='how many total observations will be processed in analysis',
        required=False,
        default=9999999999999999
    )
    parser.add_argument(
        '--method_type', type=str,
        help='implemented models are regression or classification',
        required=True
    )
    parser.add_argument(
        '--max_time_in_data', type=int,
        help='max time in data',
        required=True
    )
    parser.add_argument(
        '--experiment_number', type=int,
        help='experiment number',
        required=True
    )
    parser.add_argument(
        '--save_history', type=int,
        help='if 1 saves history from rank=0.  if 0, no history is saved',
        required=True
    )
    parser.add_argument(
        '--files_to_process_path', type=str,
        help='path to the files to process',
        required=True
    )
    parser.add_argument(
        '--particles_per_shard', type=int,
        help='number of particles per shard - must be in multiples of PARTICLES_PER_SEND',
        required=True
    )
    parser.add_argument(
        '--comm_frequency', type=int,
        help='How often nodes/processes should communicate.  Measured in units of time',
        required=True
    )
    parser.add_argument(
        '--version', type=int,
        help='which seed was used to generate the GP',
        required=True
    )
    return parser.parse_args()


args = get_args()


###################################
#  Commmand Line Args Above Here  #
###################################

if False: #args.comm_frequency >= args.max_time_in_data:
    communication_times=[args.max_time_in_data]
else:
    temp = list(range(args.comm_frequency, args.max_time_in_data, args.comm_frequency))
    communication_times = [item - 1 for item in temp]
    if args.max_time_in_data in communication_times:
        communication_times = [ct for ct in communication_times if ct != args.max_time_in_data]
    #if args.max_time_in_data not in communication_times:
    #    communication_times.append(args.max_time_in_data)
print("all communication_times = ", communication_times)
file_paths = ftp.get_files_to_process(args.files_to_process_path, args.version)
print("file_paths=",file_paths)
particles = None
last_times = None
history = None

for fp in tqdm(range(len(file_paths))):
    print("shard:", rank, " opening ", file_paths[fp])
    if rank == 0:
        starting_point = random.sample(range(size), size)#.reshape((size,1))
        starting_point = [[item] for item in starting_point]
    else:
        starting_point = None #np.zeros(1, dtype=int)
    # Broadcast n to all processes
    #print("Process ", rank, " before starting_point = ", starting_point)
    comm.Barrier()
    starting_point = comm.scatter(starting_point, root=0)[0]
    #print("Process ", rank, " after starting_point = ", starting_point)

    with open(file_paths[fp]) as f_in:
        #temp = np.genfromtxt(itertools.islice(f_in, rank, args.num_obs, size), delimiter=',',)
        if args.method_type == "regression":
            data = np.genfromtxt(itertools.islice(f_in, starting_point, args.num_obs, size), delimiter=',',)
        if args.method_type == "classification":
            large_data = np.genfromtxt(itertools.islice(f_in, 0, args.num_obs, 1), delimiter=',',)
            D = large_data.shape[1] - 2
            moded_times=large_data[:, D+1] % size
            data = large_data[moded_times==rank, :]
            del large_data
            del moded_times
    print("shard:", rank, " done...")

   
    D = data.shape[1] - 2
    X = data[:, :D]
    y = data[:, D]
    times = data[:, D+1] 
    epoch_start = 0
    
    # get the proper communication time for each shard
    current_file_comm_times=[]
    current_file_check_times=[]
    
    communication_times = [ct for ct in communication_times if ct>times[0]]
    
    for c_time in range(len(communication_times)):
        try:
            #print("Remaining Global communication_times:", communication_times)
            epoch_end = (next(t[0] for t in enumerate(times) if t[1] > communication_times[c_time]))
            #print("appending to comm and check", times[epoch_end-1])
            current_file_comm_times.append(times[epoch_end-1])
            current_file_check_times.append(times[epoch_end-1])
        except:
            epoch_end = len(times)
            #print("appending to check", times[epoch_end-1])
            current_file_check_times.append(times[epoch_end-1])
            if abs(times[-1] - communication_times[c_time]) < size:
                #print("appending to comm", times[epoch_end-1])
                current_file_comm_times.append(times[epoch_end-1])
                
            #print(rank, " has times[-1] =", times[-1], " communication_times[c_time]=", communication_times[c_time], " diff =",  abs(times[-1] - communication_times[c_time]))
        
        
    current_file_comm_times = sorted(list(set(current_file_comm_times)))    
    current_file_check_times = sorted(list(set(current_file_check_times)))
    #print("Rank:", rank, "current_file_comm_times=", current_file_comm_times)
    #print("Rank:", rank, "current_file_check_times=", current_file_check_times)

    
    c_time=-1
    while epoch_start < len(times):
        c_time+=1
    #for c_time in tqdm(loop_range_size):
        if c_time < len(current_file_check_times):
            try:
                epoch_end = next(t[0] for t in enumerate(times) if t[1] > current_file_check_times[c_time])
                #communicate = True
            except:
                #print(rank, "called exception...")
                epoch_end = len(times)
        else:
            epoch_end = len(times)
            #communicate = False
            
        X_small = X[epoch_start: epoch_end, :]
        y_small = y[epoch_start: epoch_end]
        times_small = times[epoch_start: epoch_end]
        if times_small[-1] in current_file_comm_times:
            #print("first if in exception")
            #print(rank, " communication = True")
            communicate = True
        else:
            #print("else if in exception")
            #print(rank, " communication = False")
            communicate = False
        #print("Rank,", rank, " processing last time of: ", times_small[-1])
        pf_run_time -= time.clock()
        particles, history = (
            spf.pf(
                X_small, y_small, times_small,
                args.particles_per_shard,
                args.stationary_prior_mean,
                args.stationary_prior_std,
                args.stepsize,
                save_history=args.save_history,
                method=args.method_type,
                init_particles=particles,
                last_times=last_times,
                prev_history=history,
                shard_count=size
            )
        )
        pf_run_time += time.clock()
        if communicate == True:
            #print(rank, " communication = True")

            particles = np.hstack((particles, times[epoch_end-1]*np.ones((particles.shape[0], 1))))
            
            particles_per_send_per_shard = int(min(PARTICLES_PER_SEND,args.particles_per_shard))
            number_of_sends = int(args.particles_per_shard / particles_per_send_per_shard)
            
            ################################
            #  Loop for sending particles  #
            ################################
            for nos in range(number_of_sends):
                #print(rank, ":loop for sending particles  " )
                #particlesGathered = np.zeros([particles_per_send_per_shard * size, D+1])
                #split_sizes = np.array([particles_per_send_per_shard*(D+1)]*size)
                #displacements = np.insert(np.cumsum(split_sizes),0,0)[0:-1]
                
                comm.Barrier()
                send_start = nos*particles_per_send_per_shard
                send_stop = (nos+1)*particles_per_send_per_shard
                #print("A")
                #comm.Allgatherv(
                #    [particles[send_start:send_stop,:], MPI.DOUBLE],
                #    [particlesGathered, split_sizes, displacements, MPI.DOUBLE]
                #)
                comm_time -= time.clock()
                all_gathered_stuff = comm.allgather(particles[send_start:send_stop,:])
                comm_time += time.clock()
                #print(rank, "len(all_gathered_stuff)=", len(all_gathered_stuff))
                #print(rank, "all_gathered_stuff[0].shape=", all_gathered_stuff[0].shape)
                vstacked_stuff = np.vstack(all_gathered_stuff)
                #print(rank, "vstacked_stuff.shape=", vstacked_stuff.shape)
                #print("B")
                if nos==0:
                    all_particlesGathered = vstacked_stuff.copy() #particlesGathered.copy()
                else:
                    #all_particlesGathered = np.vstack((all_particlesGathered, particlesGathered.copy()))
                    all_particlesGathered = np.vstack((all_particlesGathered, vstacked_stuff.copy()))
                
            #print(rank, "all_particlesGathered.shape=", all_particlesGathered.shape)
            new_inds = np.random.choice(args.particles_per_shard * size, args.particles_per_shard)
            part_tmp = all_particlesGathered[new_inds, :]
            particles = part_tmp[:, :D]
            last_times = part_tmp[:, D]
            epoch_start = epoch_end
        if communicate == False:
            #print(rank, " communication = False")
            last_times = times[epoch_end-1]*np.ones((particles.shape[0]))
            epoch_start = epoch_end
            
# be smarter about this.  Read at rank 0 and communicate data or read file backwards and stop after time changes
# need something slick
print("Processing last observation in data...")
with open(file_paths[-1]) as f_in:
    data = np.genfromtxt(itertools.islice(f_in, 0, args.num_obs, 1), delimiter=',',)

times = data[:, D+1]
max_time = times[-1]
epoch_start = next(t[0] for t in enumerate(times) if t[1] == max_time)
X_small = data[epoch_start:, :D]
y_small = data[epoch_start:, D]
times_small = data[epoch_start:, D+1]

pf_run_time -= time.clock()
particles, history = (
    spf.pf(
        X_small, y_small, times_small,
        args.particles_per_shard,
        args.stationary_prior_mean,
        args.stationary_prior_std,
        args.stepsize,
        save_history=args.save_history,
        method=args.method_type,
        init_particles=particles,
        last_times=last_times,
        prev_history=history,
        shard_count=1
    )
)
pf_run_time += time.clock()
  
print("shard:", rank, " final communication...")
particles = np.hstack((particles, times[epoch_end-1]*np.ones((particles.shape[0], 1))))

particles_per_send_per_shard = int(min(PARTICLES_PER_SEND,args.particles_per_shard))
number_of_sends = int(args.particles_per_shard / particles_per_send_per_shard)
for nos in range(number_of_sends):
    #particlesGathered = np.zeros([particles_per_send_per_shard * size, D+1])
    #split_sizes = np.array([particles_per_send_per_shard*(D+1)]*size)
    #displacements = np.insert(np.cumsum(split_sizes),0,0)[0:-1]
    
    comm.Barrier()
    send_start = nos*particles_per_send_per_shard
    send_stop = (nos+1)*particles_per_send_per_shard
    #comm.Gatherv(
    #    [particles[send_start:send_stop,:], MPI.DOUBLE],
    #    [particlesGathered, split_sizes, displacements, MPI.DOUBLE],
    #    root=0
    #)
    comm_time -= time.clock()
    final_gathered_stuff = comm.gather(particles[send_start:send_stop,:], root=0)
    comm_time += time.clock()
    if rank == 0:
        print(rank, "len(final_gathered_stuff)=", len(final_gathered_stuff))
        print(rank, "final_gathered_stuff[0].shape=", final_gathered_stuff[0].shape)
        final_vstacked_stuff = np.vstack(final_gathered_stuff)
        print(rank, "final_vstacked_stuff.shape=", final_vstacked_stuff.shape)
        if nos==0:
            final_particlesGathered = final_vstacked_stuff.copy()
        else:
            final_particlesGathered = np.vstack((final_particlesGathered, final_vstacked_stuff.copy()))

print("shard:", rank, " done...")
if rank == 0:
    final_particlesGathered = np.hstack(
        (final_particlesGathered, pf_run_time*np.ones((final_particlesGathered.shape[0], 1))))
    final_particlesGathered = np.hstack(
        (final_particlesGathered, comm_time*np.ones((final_particlesGathered.shape[0], 1))))
    total_run_time = time.clock() - start_time
    final_particlesGathered = np.hstack(
        (final_particlesGathered, total_run_time*np.ones((final_particlesGathered.shape[0], 1))))

    print("writing output")
    file_name = (
        "/exp_num="+str(args.experiment_number) +
        "_part_num="+str(args.particles_per_shard) +
        "_shards="+str(size)+
        "_comm_freq=" + str(args.comm_frequency) +
        "_stepsize=" + str(args.stepsize)
    )
    if not os.path.exists("experiment_results/" + args.files_to_process_path + str(args.version)):
            os.makedirs("experiment_results/" + args.files_to_process_path + str(args.version))
    final_particle_paths = ("experiment_results/" + args.files_to_process_path + str(args.version) + file_name)
    np.save(final_particle_paths, final_particlesGathered)

    if args.save_history:
        history_paths = ("experiment_results/history/" + args.files_to_process_path + str(args.version))
        if not os.path.exists(history_paths):
            os.makedirs(history_paths)
        np.save(history_paths+file_name, history)
