import pdb
import time

from scipy.special import logsumexp
from scipy import stats
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import particle_filter
import embarrassingly_parallel
import prep_simulation_data
import history
import params
import pf_plots

from files_to_process import files_to_process


def log_sigmoid(x):
    return -np.logaddexp(0, -x)


def loglik(b, X, y, sigma=None, method='classification', mult_obs=False, shard_count=1):
    # if method == 'regression', sigma is the standard deviation of the residuals (assumed not learned)
    
    bX = np.dot(b, X.T)
    #print("y=",y)
    if isinstance(y,np.float): #type(y) == 'numpy.float64':
        mult_obs = False
    if isinstance(y,np.ndarray): #type(y) == 'numpy.ndarray':
        mult_obs = True
        
    if mult_obs:
        ll = np.zeros(b.shape[0])
        
        for i in range(len(y)):
            # re-write, not very efficient
            if method == 'classification':
                if y[i] == 1:
                    ll += log_sigmoid(bX[:, i])*shard_count
                else:
                    assert y[i] == 0
                    ll += log_sigmoid(-1*bX[:, i])*shard_count
                    
            elif method == 'regression':
                ll += stats.norm.logpdf(y[i], bX[:, i], sigma)*shard_count
            else:
                raise ValueError('{} is not implemented with multiple observations'.format(method))
    else:
        if method == 'classification':
            if y == 1:
                ll = log_sigmoid(bX)*shard_count
            else:
                assert y == 0
                ll = log_sigmoid(-1*bX)*shard_count
        elif method == 'regression':
            ll = stats.norm.logpdf(y, bX, sigma)*shard_count
        else:
            raise ValueError('{} is not implemented'.format(method))
    return ll


def pf(X, y, times, num_particles, b_prior_mean=0., b_prior_std = 1., stepsize=0.05,         
       save_history=True, sigma=1., method='classification', init_particles=None, 
       last_times=None, prev_history=None, shard_count=1, rank=-1):
    unique_times = np.unique(times)
    #if len(unique_times) < len(times):
    #    repeat_obs = True
    #else:
    #    repeat_obs = False
    # initialize particles
    N, D = X.shape
    T = len(unique_times)
    if save_history:
        history = np.zeros((T, D))
        
    if init_particles is None:
        particles = np.random.normal(b_prior_mean, b_prior_std, size=[num_particles, D])
    else:
        particles = init_particles
        if last_times is not None:
            delays = times[0] - last_times
            delays[delays<0]=0
            for d in range(D):
                try:
                    timestepsize = stepsize * delays
                    particles[:, d] = particles[:, d] + np.random.normal(np.zeros(num_particles), timestepsize)
                except ValueError:
                    print("rank:", rank, " error.")
                    print('len(delays): {}; N={}'.format(len(delays), N))
                    #print("delays=",delays)
                    #print("times[0]=",times[0])
                    #print("last_times=",last_times)

    obs_inds = np.argwhere(times==unique_times[0]).squeeze()
    #pdb.set_trace()
    
    #print("times=",times)
    #print("unique_times[0]=",unique_times[0])
    #print("obs_inds=",obs_inds)
    #print("type obs_inds=",type(obs_inds))
    #print("np.argwhere(times==unique_times[0]=",np.argwhere(times==unique_times[0]))
    #print("np.argwhere(times==unique_times[0]).squeeze()=",np.argwhere(times==unique_times[0]).squeeze())
    #if len(obs_inds) > 1:
    #    repeat_obs = True
    #else:
    #    repeat_obs = False
    repeat_obs = False
    
    ll = loglik(
        particles, X[obs_inds, :], y[obs_inds], 
        method=method, mult_obs=repeat_obs, 
        sigma=sigma, shard_count=shard_count
    )
    log_weights = ll - logsumexp(ll)
    
    new_inds = np.random.choice(num_particles, num_particles, p=np.exp(log_weights))
    particles = particles[new_inds, :]
    
    if save_history:
        history[0, :] += np.mean(particles, axis=0)
        
    for j in range(1, len(unique_times)):
        timestepsize = stepsize * (unique_times[j]-unique_times[j-1])
        rescale_const = b_prior_std / np.sqrt(b_prior_std**2 + timestepsize**2)
        particles += np.random.normal(0, timestepsize, size=[num_particles, D])
        particles = rescale_const * particles
        obs_inds = np.argwhere(times==unique_times[j]).squeeze()
        try:
            #if len(obs_inds) > 1:
            #    repeat_obs = True
            #else:
            #    repeat_obs = False
            repeat_obs = False
            ll = loglik(
                particles, X[obs_inds, :], y[obs_inds], 
                method=method, mult_obs=repeat_obs, 
                sigma=sigma, shard_count=shard_count
            )
        except IndexError:
            pdb.set_trace()
        log_weights = ll - logsumexp(ll)
        new_inds = np.random.choice(num_particles, num_particles, p=np.exp(log_weights))
        particles = particles[new_inds, :]
        if save_history:
            history[j, :] = np.mean(particles, axis=0)
    
    if save_history:
        if prev_history is not None:
            history = np.vstack((prev_history, history))
    else:
        history = None
    return particles, history
    
