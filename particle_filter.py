#import pf_models as pf
import numpy as np
import pandas as pd
import time
from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def log_sigmoid(x):
    return -np.logaddexp(0, -x)

def loglik(b, X, y, sigma=None, method='classification', mult_obs=False):
    # if method == 'regression', sigma is the standard deviation of the residuals (assumed not learned)
    bX = np.dot(b, X.T)
    if mult_obs:
        ll = np.zeros(b.shape[0])
        
        for i in range(len(y)):
            # re-write, not very efficient
            if method == 'classification':
                if y[i] == 1:
                    ll += log_sigmoid(bX[:, i])
                else:
                    assert y[i] == 0
                    ll += log_sigmoid(-1*bX[:, i])
                    
            elif method == 'regression':
                ll += stats.norm.logpdf(y[i], bX[:, i], sigma)
            else:
                raise ValueError('{} is not implemented with multiple observations'.format(method))
    else:
        if method == 'classification':
            if y == 1:
                ll = log_sigmoid(bX)
            else:
                assert y == 0
                ll = log_sigmoid(-1*bX)
        elif method == 'regression':
            ll = stats.norm.logpdf(y, bX, sigma)
        else:
            raise ValueError('{} is not implemented'.format(method))
    return ll

class particle_filter:
#particle filter class
    def __init__(self, dat, params_obj, pf_rank = 0, run_number = 0, save_history=True, method='classification'):

        b_prior_mean=0.
        self.b_prior_std = dat['B']#1.
        self.sigma=1.
        self.times = dat['time_value']
        self.method=method
        self.shards = dat['shards']
        self.stepsize = dat['Tau_inv_std']*5
        self.X = dat['X_matrix']
        self.y = dat['Y'] 
        self.save_history = save_history
        self.unique_times = np.unique(self.times)
        self.PART_NUM= params_obj.get_particles_per_shard()
        self.sample_method      = params_obj.get_sample_method()
        
        self.repeat_obs = self.get_repeat_obs_bool()
        
        # initialize particles
        self.N, self.D = self.X.shape
        self.T = len(self.unique_times)
        
        self.particles = np.random.normal(b_prior_mean, self.b_prior_std, size=[self.PART_NUM, self.D])
        
        obs_inds = self.times==np.min(self.unique_times)
        
        ll = loglik(
            self.particles, 
            self.X[obs_inds, :], 
            self.y[obs_inds], 
            method=self.method, 
            mult_obs=self.repeat_obs, 
            sigma=self.sigma
        )
        log_weights = ll - logsumexp(ll)
        
        new_inds = np.random.choice(self.PART_NUM, self.PART_NUM, p=np.exp(log_weights))
        self.particles = self.particles[new_inds, :]
        self.last_processed_time = np.min(self.unique_times)
        if self.save_history:
            self.history = np.mean(self.particles, axis=0)


    def run_particle_filter(self):

        for j in range(0, len(self.unique_times)):            
            if self.last_processed_time < self.unique_times[j]:
                timestepsize = self.stepsize * (self.unique_times[j]-self.last_processed_time)
            else:
                timestepsize = self.stepsize/10000
                
            rescale_const = self.b_prior_std / np.sqrt(self.b_prior_std**2 + timestepsize**2)
            self.particles += np.random.normal(0, timestepsize, size=[self.PART_NUM, self.D])
            self.particles = rescale_const * self.particles
            obs_inds = self.times==self.unique_times[j]
            repeat_obs = self.get_repeat_obs_bool()
            ll = loglik(
                self.particles, 
                self.X[obs_inds, :], 
                self.Y[obs_inds], 
                method=self.method, 
                mult_obs=repeat_obs, 
                sigma=self.sigma
            )
            log_weights = ll - logsumexp(ll)
            new_inds = np.random.choice(self.PART_NUM, self.PART_NUM, p=np.exp(log_weights))
            self.particles = self.particles[new_inds, :]

            if self.save_history:
                new_means = np.mean(self.particles, axis=0)
                self.history = np.vstack((self.history, new_means))
            
            self.last_processed_time = self.unique_times[j]

        return
    
    def update_data(self, dat, run_number):            
        # update data used in new particle filter
        self.run_number = run_number
        self.X = dat['X_matrix']
        self.Y = dat['Y']
        self.times = dat['time_value']
        self.unique_times = np.unique(self.times)

    def update_params(self, updated_params):
        self.particles = np.vstack(updated_params)

    def get_pf_parameter_means(self):
        self.params_to_ship_mean = np.mean(self.params_to_ship, axis=0)
        
    def get_pf_parameter_cov(self):
        self.params_to_ship_cov = np.cov(self.params_to_ship.T)
         
    def compute_particle_kernel_weights(self, mean_params, cov_parmas):

        # get shard inverse covariances * shard count        
        shard_cov_inv_list=[]
        for V_s in range(len(cov_parmas)):
            if np.linalg.matrix_rank(cov_parmas[V_s]) == self.p:
                shard_cov_inv_list.append(np.linalg.inv(cov_parmas[V_s]*self.shards))
            else:
                I = np.identity(self.p)
                #print("I.shape:", I.shape)
                diag_values = cov_parmas[V_s].diagonal()
                max_var = np.nanmax(diag_values)
                I_s = I*max_var/100
                #print("I_s.shape:", I_s.shape)
                Sigma = cov_parmas[V_s]*self.shards + I_s
                #print("Sigma.shape:", Sigma.shape)
                shard_cov_inv_list.append(np.linalg.inv(Sigma))
        
        # get Global covariance
        V_inv = np.zeros(shard_cov_inv_list[0].shape)
        for s in range(len(shard_cov_inv_list)):
            V_inv = np.add(V_inv,shard_cov_inv_list[s])
        V = np.linalg.inv(V_inv) 
        # multiply shard covariances and shard mean and then sum
        S_inv_x_shard_mean = [np.matmul(S_inv, shard_mean) for S_inv, shard_mean in zip(shard_cov_inv_list, mean_params)]
        S_inv_x_shard_mean_sum = np.zeros(np.matmul(shard_cov_inv_list[s],mean_params[s]).shape)
        for s in range(len(shard_cov_inv_list)):
            temp = np.matmul(shard_cov_inv_list[s],mean_params[s])
            S_inv_x_shard_mean_sum = np.add(S_inv_x_shard_mean_sum, temp)
            
        global_mean = np.matmul(V, S_inv_x_shard_mean_sum)
        
        #particles are in self.params_to_ship
        self.not_norm_wts = multivariate_normal.pdf(
            self.params_to_ship, mean=global_mean, cov=V)

    def get_repeat_obs_bool(self):
        return len(self.unique_times) < len(self.times)
