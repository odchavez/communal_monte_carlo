"""
run with: mpirun -np 4 python MPI_pf.py --stationary_prior_mean 0 --stationary_prior_std 1 --stepsize 0.001 --num_obs 100000000000000 --method_type regression --max_time_in_data 999999 --experiment_number 0 --save_history 1 --files_to_process_path synth_data/regression/Xy_N=1000000_Epoch_N=100000_Nt=1_p=32/GP_version=0 --particles_per_shard 500 --comm_frequency 50000
"""

import os
import time
import itertools
import pdb
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm
from mpi4py import MPI

from scipy.special import logsumexp
from scipy import stats

import spf as spf
import files_to_process as ftp



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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
        help='number of particles per shard',
        required=True
    )
    parser.add_argument(
        '--comm_frequency', type=int,
        help='How often nodes/processes should communicate.  Measured in units of time',
        required=True
    )
    return parser.parse_args()


args = get_args()


###################################
#  Commmand Line Args Above Here  #
###################################

if args.comm_frequency >= args.max_time_in_data:
    communication_times=[args.max_time_in_data]
else:
    communication_times = list(range(args.comm_frequency, args.max_time_in_data, args.comm_frequency))
    if args.max_time_in_data not in communication_times:
        communication_times.append(args.max_time_in_data)
file_paths = ftp.files_to_process[args.files_to_process_path]

particles = None
last_times = None
history = None

for fp in tqdm(range(len(file_paths))):
    # Nees a slicker way to run pf with many files.  
    # main problems was with some nodes being out of data while other nodes still had data to process.  
    with open(file_paths[fp]) as f_in:
        temp = np.genfromtxt(itertools.islice(f_in, rank, args.num_obs, size), delimiter=',',)
    if fp==0:
        data=temp
    else:
        data=np.vstack((data, temp))
        
D = data.shape[1] - 2
X = data[:, :D]
y = data[:, D]
times = data[:, D+1] 
epoch_start = 0
current_file_comm_times = communication_times

for c_time in tqdm(range(len(current_file_comm_times))):
    try:
        epoch_end = next(t[0] for t in enumerate(times) if t[1] > current_file_comm_times[c_time])
    except:
        epoch_end = len(times)

    X_small = X[epoch_start: epoch_end, :]
    y_small = y[epoch_start: epoch_end]
    times_small = times[epoch_start: epoch_end]
    print("Rank,", rank, " processing last time of: ", times_small[-1])
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

    particles = np.hstack((particles, times[epoch_end-1]*np.ones((particles.shape[0], 1))))

    particlesGathered = np.zeros([args.particles_per_shard * size, D+1])

    split_sizes = np.array([args.particles_per_shard*(D+1)]*size)

    displacements = np.insert(np.cumsum(split_sizes),0,0)[0:-1]

    comm.Barrier()
    comm.Allgatherv(
        [particles, MPI.DOUBLE],
        [particlesGathered, split_sizes, displacements, MPI.DOUBLE]
    )

    new_inds = np.random.choice(args.particles_per_shard * size, args.particles_per_shard)

    part_tmp = particlesGathered[new_inds, :]
    particles = part_tmp[:, :D]
    last_times = part_tmp[:, D]
    epoch_start = epoch_end

# be smarter about this.  Read at rank 0 and communicate data or read file backwards and stop after time changes
# need something slick
with open(file_paths[-1]) as f_in:
    data = np.genfromtxt(itertools.islice(f_in, 0, args.num_obs, 1), delimiter=',',)

times = data[:, D+1]
max_time = times[-1]
epoch_start = next(t[0] for t in enumerate(times) if t[1] == max_time)
X_small = data[epoch_start:, :D]
y_small = data[epoch_start:, D]
times_small = data[epoch_start:, D+1]
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

particlesGathered = np.zeros([args.particles_per_shard * size, D+1])
split_sizes = np.array([args.particles_per_shard*(D+1)]*size)
displacements = np.insert(np.cumsum(split_sizes),0,0)[0:-1]

comm.Barrier()
comm.Allgatherv( [particles, MPI.DOUBLE], [particlesGathered, split_sizes, displacements, MPI.DOUBLE])

new_inds = np.random.choice(args.particles_per_shard * size, args.particles_per_shard)
final_particles = particlesGathered[new_inds, :]

all_shard_params = comm.gather(final_particles, root=0)

if rank == 0:
    print("writing output")
    file_name = (
        "/exp_num="+str(args.experiment_number) +
        "_part_num="+str(args.particles_per_shard) +
        "_shards="+str(size)+
        "_comm_freq=" + str(args.comm_frequency)
    )
    if not os.path.exists("experiment_results/" + args.files_to_process_path):
            os.makedirs("experiment_results/" + args.files_to_process_path)
    final_particle_paths = ("experiment_results/" + args.files_to_process_path + file_name)
    np.save(final_particle_paths, all_shard_params)

    if args.save_history:
        history_paths = ("experiment_results/history/" + args.files_to_process_path)
        if not os.path.exists(history_paths):
            os.makedirs(history_paths)
        np.save(history_paths+file_name, history)
