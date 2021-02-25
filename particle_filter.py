import pf_models as pf
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
                    ll += log_sigmoid(-1*bX[:, i])
                else:
                    assert y[i] == 0
                    ll += log_sigmoid(bX[:, i])
                    
            elif method == 'regression':
                ll += stats.norm.logpdf(y[i], bX[:, i], sigma)
            else:
                raise ValueError('{} is not implemented with multiple observations'.format(method))
    else:
        if method == 'classification':
            if y == 1:
                ll = log_sigmoid(-1*bX)
            else:
                assert y == 0
                ll = log_sigmoid(bX)
        elif method == 'regression':
            ll = stats.norm.logpdf(y, bX, sigma)
        else:
            raise ValueError('{} is not implemented'.format(method))
    return ll

class particle_filter:
#particle filter class
    def __init__(self, dat, params_obj, pf_rank = 0, run_number = 0, save_history=True, method='classification'):

        self.PART_NUM           = params_obj.get_particles_per_shard()
        self.particle_list      = list()
        self.model              = params_obj.get_model()
        self.sample_method      = params_obj.get_sample_method()
        self.time_values        = [0]
        self.unique_time_values = np.unique(self.time_values)
        self.rank               = pf_rank
        self.run_number         = run_number
        
        #########################
        # Needed args: X, y, times, num_particles, b_prior_mean=0., b_prior_std = 1., stepsize=0.05,  save_history=True, sigma=1., method='classification'
        b_prior_mean=0.
        self.b_prior_std = 1.
        self.sigma=1.
        self.times = dat['time_value']
        self.method=method
        self.shards = dat['shards']
        self.stepsize = dat['Tau_inv_std']
        self.X = dat['X_matrix']
        self.y = dat['Y'] 
        self.save_history = save_history
        self.unique_times = np.unique(self.times)
        #print("self.unique_times=",self.unique_times)
        if len(self.unique_times) < len(self.times):
            self.repeat_obs = True
        else:
            self.repeat_obs = False
        # initialize particles
        self.N, self.D = self.X.shape
        self.T = len(self.unique_times)
        
        self.particles = np.random.normal(b_prior_mean, self.b_prior_std, size=[self.PART_NUM, self.D])
        #print("self.times=",self.times)
        #print("self.unique_times[0]=", self.unique_times[0])
        #print(self.times==np.min(self.unique_times))
        obs_inds = self.times==np.min(self.unique_times)#np.argwhere(self.times==np.min(self.unique_times)).squeeze()
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
        #########################
        
        if self.model== "probit_sin_wave":
            self.p=dat['p']

            self.shards=dat['shards']
            for pn in range(self.PART_NUM):
                temp_particle=pf.probit_sin_wave_particle( 
                    np.array(dat['b'][0]), dat['B'], dat['Tau_inv_std'], (self.rank, pn)
                )
                temp_particle.set_shard_number(self.shards)
                self.particle_list.append(temp_particle)
        else:
            print(model, " not implemented yet...")

    def run_particle_filter(self):
        #single interation of P particles
        
        self.not_norm_wts=np.ones(self.PART_NUM)
                
        for n in range(len(self.unique_time_values)):
            # Code in PART_NUM loop should be eliminated
            for pn in range(self.PART_NUM):

                row_index = self.time_values == self.unique_time_values[n]
                
                #XtXpre = self.X[row_index,:]
                #XtX = XtXpre.transpose().dot(XtXpre)
                self.particle_list[pn].update_particle_importance(
                    #XtX,
                    self.X[row_index,:],
                    self.Y[row_index],
                    self.unique_time_values[n]
                )
                self.not_norm_wts[pn]=self.particle_list[pn].evaluate_likelihood(self.X[row_index,:], self.Y[row_index])
            
            self.shuffle_particles()
            # REPLACE CODE TO HERE...
        ################################
        # new version of PF
        for j in range(0, len(self.unique_times)):
            
            # get the right sized time step
            if self.last_processed_time < self.unique_times[j]:
                timestepsize = self.stepsize * (self.unique_times[j]-self.last_processed_time)
            else:
                timestepsize = self.stepsize/10000
                
            rescale_const = self.b_prior_std / np.sqrt(self.b_prior_std**2 + timestepsize**2)
            self.particles += np.random.normal(0, self.stepsize, size=[self.PART_NUM, self.D])
            self.particles = rescale_const * self.particles
            obs_inds = self.times==self.unique_times[j]#np.argwhere(times==unique_times[j]).squeeze()
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
            #print(type(self.particles))
            #print(self.particles.shape)
            if self.save_history:
                new_means = np.mean(self.particles, axis=0)
                self.history = np.vstack((self.history, new_means))
            ################################
            self.last_processed_time = self.unique_times[j]
            ################################
        #print(self.history)
        return

    #def resample_locally(weights):
    #    print("resample_locally(weights) Not Implemented...")
    #    return
    
    def shuffle_particles(self):
        
        x=self.not_norm_wts
        finite_values = x[np.isfinite(x)]
        finite_max = np.nanmax(finite_values)
        finite_min = np.nanmin(finite_values)
        x[x>finite_max] = finite_max
        x[x<finite_min] = finite_min
        x[np.isnan(x)] = finite_min
        norm_wts = np.exp(x - logsumexp(x))
        norm_wts = norm_wts/np.sum(norm_wts)
        if any(np.isnan(x)):
            norm_wts = np.ones(len(self.not_norm_wts))/len(self.not_norm_wts)
        #use multinomial vs choice
        particles_kept = np.random.choice(range(self.PART_NUM),size=self.PART_NUM, p=norm_wts)

        base_particle_parameter_matrix = np.zeros(
            (
                self.PART_NUM,  
                self.particle_list[0].get_parameter_dimension()
            )
        )

        for p in range(self.PART_NUM):
            base_particle_parameter_matrix[p, :] = self.particle_list[p].get_parameter_set()
        
        for p in range(base_particle_parameter_matrix.shape[0]):
            params = base_particle_parameter_matrix[particles_kept[p], :]
            self.particle_list[p].set_particle_parameters(params) 
        #print(type(self.particle_list))
        #print(self.particle_list.shape)
        #temp_index=np.zeros(self.PART_NUM)
        #temp_index.astype(int)
#
        #for pn in range(self.PART_NUM):
        #    temp_index[particles_kept[pn]]+=1
#
        #not_chosen=np.where(temp_index==0)[0]
        #for nci in range(len(not_chosen)):
        #    for ti in range(self.PART_NUM):
        #        break_ti_for=False
        #        while(temp_index[ti]>=2):
        #            temp_index[ti]-=1
        #            self.particle_list[not_chosen[nci]].copy_particle_values(self.particle_list[ti])
        #            break_ti_for=True
        #            break
        #        if break_ti_for: break

    #def print_stuff(self):
    #    print("self.not_norm_wts=",self.not_norm_wts)

    #def get_particle_ids(self):
    #    for i in range(len(self.particle_list)):
    #        print(self.particle_list[i].get_particle_id())
    #    return

    #def get_particle_ids_history(self):
    #    for i in range(len(self.particle_list)):
    #        print(self.particle_list[i].get_particle_id_history())
    #    return

    #def get_particle_vals(self):
    #    for i in range(len(self.particle_list)):
    #        print(self.particle_list[i].print_vals())
    #    return

    #def get_particle(self, i):
    #    return(self.particle_list[i])

    def update_data(self, dat, run_number):

        self.run_number = run_number
        self.X = dat['X_matrix']
        self.Y = dat['Y']

        self.time_values = dat['time_value']
        self.unique_time_values = np.unique(self.time_values)
        self.all_shard_unique_time_values = dat['all_shard_unique_time_values']

        for pn in range(self.PART_NUM):
            self.particle_list[pn].set_bo_list(self.all_shard_unique_time_values)
            self.particle_list[pn].this_time = 0
            
        # update data used in new particle filter
        self.run_number = run_number
        self.X = dat['X_matrix']
        self.Y = dat['Y']
        self.times = dat['time_value']
        self.unique_times = np.unique(self.times)

    def update_params(self, updated_params):
        for i in range(len(self.particle_list)):
            self.particle_list[i].bo = updated_params[i]
            
    #def update_particle_id_history(self, updated_machine_history_id, updated_particle_history_id):
    #    for i in range(len(self.particle_list)):
    #        new_tuple = (updated_machine_history_id[i], updated_particle_history_id[i])
    #        self.particle_list[i].particle_id_history = new_tuple

    #def get_predictive_distribution(self, X_new):
    #    self.predictive_distribution = np.zeros(self.PART_NUM)
    #    if  self.model == "probit_sin_wave":
    #        for pn in range(self.PART_NUM):
    #            self.predictive_distribution[pn] = np.exp(self.particle_list[pn].evaluate_likelihood(X_new, Y=1))
    #    else:
    #        print("get_predictive_distribution(self, X_new) not implemented")
#
    #    return(self.predictive_distribution)

    #def plot_particle_path(self, particle_prop=0.1):
    #    print("in plot_particle_path")
#
    #    param_num=self.p#particle_list[0].get_particle(0).bo_list.shape[1]
    #    total_time_steps =self.N# len(self.particle_list[0].get_particle(0).bo_list[:,0])
    #    params=list()
    #    particle_indices = np.random.choice(self.PART_NUM, max(int(self.PART_NUM*particle_prop), 1))
#
    #    for os in range(param_num):
    #        temp_all_parts = np.zeros((len(particle_indices), total_time_steps))
    #        #for sn in range(M):
    #        for pn in range(len(particle_indices)):
    #            #particle=self.particle_list[pn].get_particle(particle_indices[pn])
    #            #p_temp = particle.bo_list[:,os].copy()
    #            p_temp = self.particle_list[pn].bo_list[:,os].copy()
    #            p_temp[np.isnan(p_temp)]=0
    #            temp_all_parts[pn,:]=np.add(temp_all_parts[pn,:],p_temp)
    #        params.append(temp_all_parts)
#
#
    #    for par_n in range(param_num):
    #        avg_param_0=np.mean(params[par_n], axis=0)
    #        std_parma_0=np.std(params[par_n], axis=0)
    #        above=np.add(avg_param_0,std_parma_0*2)
    #        below=np.add(avg_param_0,-std_parma_0*2)
#
    #        truth=self.dat['b'][:,par_n]#test['shard_0']['b'][:,par_n]
#
    #        x = np.arange(len(avg_param_0)) # the points on the x axis for plotting
#
    #        fig, ax1 = plt.subplots()
    #        plt.plot(x,truth,'black')
    #        ax1.fill_between(x, below, above, facecolor='green',  alpha=0.3)
    #        plt.plot(x,avg_param_0, 'b', alpha=.8)
    #        min_tic=np.min([np.min(below),np.min(truth)])
    #        max_tic=np.max([np.max(above),np.max(truth)])
    #        plt.yticks(np.linspace(start=min_tic, stop=max_tic, num=12))
    #        plt.grid(True)
    #        plt.show()

    #def write_bo_list(self, f_file_stem = ''):
    #    output = self.get_temp_all_particles()
    #    print((f_file_stem))
    #    print((str(self.run_number)))
    #    print((str(self.rank)))
    #    np.save(
    #        "particle_hold/file_" +
    #        f_file_stem +
    #        "_" +
    #        str(self.run_number) +
    #        "_" +
    #        str(self.rank),
    #        output
    #    )

    #def get_temp_all_particles(self):
    #    particle_number = self.PART_NUM
    #    Z_dim = particle_number
    #    bo_shape =  self.particle_list[0].bo_list.shape
    #    temp_all_parts = np.zeros((bo_shape[0], bo_shape[1], Z_dim))
#
    #    for pn in range(particle_number):
    #        particle = self.get_particle(pn)
    #        temp_all_parts[:,:,pn] = self.particle_list[pn].bo_list
#
    #    return temp_all_parts

    def collect_params(self):
        #list comprehention form: new_list = [expression for member in iterable]
        #output = [particle.bo for particle in self.particle_list]
        #if len(output)>0:
        #    self.params_to_ship = np.reshape(output, (self.PART_NUM, self.p))
        #else:
        #    self.params_to_ship = []
        self.params_to_ship = self.particles
        #print("self.params_to_ship.shape=",self.params_to_ship.shape)
        #print("self.particles.shape=", self.particles.shape)
#
    #def collect_history_ids(self):
    #    self.machine_history_ids_to_ship = np.zeros((self.PART_NUM))
    #    self.particle_history_ids_to_ship = np.zeros((self.PART_NUM))
    #    for pn in range(self.PART_NUM):
    #        self.machine_history_ids_to_ship[pn] = self.particle_list[pn].particle_id_history[0]
    #        self.particle_history_ids_to_ship[pn] = self.particle_list[pn].particle_id_history[1]
    
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
