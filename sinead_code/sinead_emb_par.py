import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from mpi4py import MPI
import pdb
import sinead_pf

from scipy.special import logsumexp
from scipy import stats


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

stationary_prior_mean = 0.
stationary_prior_std = 1.
stepsize=0.1

particles_per_shard = 1000

filename = "tmp_data.csv" # CSV, first D cols are X, next col is y, next col is times
num_obs = 10000

with open(filename) as f_in:
    data = np.genfromtxt(itertools.islice(f_in, rank, num_obs, size), delimiter=',')


D = data.shape[1] - 2

X = data[:, :D]
y = data[:, D]
times = data[:, D+1]

communication_times = 10*np.arange(1, 10) # list of times for the communications
particles = None
last_times = None
epoch_start = 0

save_history = True


if save_history:
    # currently just for test purposes -- saves the average particle at each time step, on the master swarm only
    history = None
    
for c_time in communication_times:
    try:
        epoch_end = next(t[0] for t in enumerate(times) if t[1] > c_time) 
    except:
        epoch_end = len(times)

    X_small = X[epoch_start: epoch_end, :]
    y_small = y[epoch_start: epoch_end]
    times_small = times[epoch_start: epoch_end]
    if save_history:
        particles, history = sinead_pf.sinead_pf(X_small, y_small, times_small, particles_per_shard, stationary_prior_mean, stationary_prior_std, stepsize, save_history=True, method='classification', init_particles=particles, last_times=last_times, prev_history=history)
    else:
        particles = sinead_pf.sinead_pf(X_small, y_small, times_small, particles_per_shard, stationary_prior_mean, stationary_prior_std, stepsize, save_history=False, method='classification', init_particles=particles, last_times=last_times)

    particles = np.hstack((particles, times[epoch_end-1]*np.ones((particles.shape[0], 1))))

    particlesGathered = np.zeros([particles_per_shard * size, D+1])
    
    split_sizes = np.array([particles_per_shard*(D+1)]*size)
    displacements = np.insert(np.cumsum(split_sizes),0,0)[0:-1]
    
    comm.Barrier()

    comm.Allgatherv([particles, MPI.DOUBLE], [particlesGathered, split_sizes, displacements, MPI.DOUBLE])

    new_inds = np.random.choice(particles_per_shard * size, particles_per_shard)

    part_tmp = particlesGathered[new_inds, :]
    particles = part_tmp[:, :D]
    last_times = part_tmp[:, D]
    epoch_start = epoch_end
    
    

if rank == 0:
    if save_history:
        np.savetxt('tmp_particles.csv', history, delimiter=',')
