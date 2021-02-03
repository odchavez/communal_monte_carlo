import pf_models as pf
import numpy as np
import pandas as pd
import time
from scipy.stats import multivariate_normal

from matplotlib import pyplot as plt

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


class particle_filter:
#particle filter class
    def __init__(self, dat, params_obj, pf_rank = 0, run_number = 0):

        self.PART_NUM           = params_obj.get_particles_per_shard()
        self.particle_list      = list()
        self.model              = params_obj.get_model()
        self.sample_method      = params_obj.get_sample_method()
        self.time_values        = [0]
        self.unique_time_values = np.unique(self.time_values)
        self.rank               = pf_rank
        self.run_number         = run_number

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
            
            for pn in range(self.PART_NUM):

                row_index = self.time_values == self.unique_time_values[n]
                
                XtXpre = self.X[row_index,:]
                XtX = XtXpre.transpose().dot(XtXpre)
                self.particle_list[pn].update_particle_importance(
                    XtX,
                    self.X[row_index,:],
                    self.Y[row_index],
                    self.unique_time_values[n]
                )
                self.not_norm_wts[pn]=self.particle_list[pn].evaluate_likelihood(self.X[row_index,:], self.Y[row_index])

            self.shuffle_particles()
        return

    def resample_locally(weights):
        print("resample_locally(weights) Not Implemented...")
        return
    
    def shuffle_particles(self):
        
        self.not_norm_wts[np.isnan(self.not_norm_wts)] = -100.0
        
        top    = np.exp(self.not_norm_wts)
        top_min = np.nanmin(top)
        fill_min = np.min([top_min, 1/len(top)])
        top[np.isnan(top)]=fill_min

        bottom = np.sum(top)
        if bottom == 0:
            max_val  = np.max(self.not_norm_wts)
            top      = np.exp(self.not_norm_wts - max_val)
            bottom   = np.sum(top)
            norm_wts = top/bottom
        else:
            norm_wts=top/bottom
        #print("Class: particle_filter, function: shuffle_particles -  norm_wts:", norm_wts)
        particles_kept = np.random.choice(range(self.PART_NUM),size=self.PART_NUM, p=norm_wts)
        temp_index=np.zeros(self.PART_NUM)
        temp_index.astype(int)

        for pn in range(self.PART_NUM):
            temp_index[particles_kept[pn]]+=1

        not_chosen=np.where(temp_index==0)[0]
        for nci in range(len(not_chosen)):
            for ti in range(self.PART_NUM):
                break_ti_for=False
                while(temp_index[ti]>=2):
                    temp_index[ti]-=1
                    self.particle_list[not_chosen[nci]].copy_particle_values(self.particle_list[ti])
                    break_ti_for=True
                    break
                if break_ti_for: break

    def print_stuff(self):
        print("self.not_norm_wts=",self.not_norm_wts)

    def get_particle_ids(self):
        for i in range(len(self.particle_list)):
            print(self.particle_list[i].get_particle_id())
        return

    def get_particle_ids_history(self):
        for i in range(len(self.particle_list)):
            print(self.particle_list[i].get_particle_id_history())
        return

    def get_particle_vals(self):
        for i in range(len(self.particle_list)):
            print(self.particle_list[i].print_vals())
        return

    def get_particle(self, i):
        return(self.particle_list[i])

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

    def update_params(self, updated_params):
        for i in range(len(self.particle_list)):
            self.particle_list[i].bo = updated_params[i]
            
    def update_particle_id_history(self, updated_machine_history_id, updated_particle_history_id):
        for i in range(len(self.particle_list)):
            new_tuple = (updated_machine_history_id[i], updated_particle_history_id[i])
            self.particle_list[i].particle_id_history = new_tuple

    def get_predictive_distribution(self, X_new):
        self.predictive_distribution = np.zeros(self.PART_NUM)
        if  self.model == "probit_sin_wave":
            for pn in range(self.PART_NUM):
                self.predictive_distribution[pn] = np.exp(self.particle_list[pn].evaluate_likelihood(X_new, Y=1))
        else:
            print("get_predictive_distribution(self, X_new) not implemented")

        return(self.predictive_distribution)

    def plot_particle_path(self, particle_prop=0.1):
        print("in plot_particle_path")

        param_num=self.p#particle_list[0].get_particle(0).bo_list.shape[1]
        total_time_steps =self.N# len(self.particle_list[0].get_particle(0).bo_list[:,0])
        params=list()
        particle_indices = np.random.choice(self.PART_NUM, max(int(self.PART_NUM*particle_prop), 1))

        for os in range(param_num):
            temp_all_parts = np.zeros((len(particle_indices), total_time_steps))
            #for sn in range(M):
            for pn in range(len(particle_indices)):
                #particle=self.particle_list[pn].get_particle(particle_indices[pn])
                #p_temp = particle.bo_list[:,os].copy()
                p_temp = self.particle_list[pn].bo_list[:,os].copy()
                p_temp[np.isnan(p_temp)]=0
                temp_all_parts[pn,:]=np.add(temp_all_parts[pn,:],p_temp)
            params.append(temp_all_parts)


        for par_n in range(param_num):
            avg_param_0=np.mean(params[par_n], axis=0)
            std_parma_0=np.std(params[par_n], axis=0)
            above=np.add(avg_param_0,std_parma_0*2)
            below=np.add(avg_param_0,-std_parma_0*2)

            truth=self.dat['b'][:,par_n]#test['shard_0']['b'][:,par_n]

            x = np.arange(len(avg_param_0)) # the points on the x axis for plotting

            fig, ax1 = plt.subplots()
            plt.plot(x,truth,'black')
            ax1.fill_between(x, below, above, facecolor='green',  alpha=0.3)
            plt.plot(x,avg_param_0, 'b', alpha=.8)
            min_tic=np.min([np.min(below),np.min(truth)])
            max_tic=np.max([np.max(above),np.max(truth)])
            plt.yticks(np.linspace(start=min_tic, stop=max_tic, num=12))
            plt.grid(True)
            plt.show()

    def write_bo_list(self, f_file_stem = ''):
        output = self.get_temp_all_particles()
        print((f_file_stem))
        print((str(self.run_number)))
        print((str(self.rank)))
        np.save(
            "particle_hold/file_" +
            f_file_stem +
            "_" +
            str(self.run_number) +
            "_" +
            str(self.rank),
            output
        )

    def get_temp_all_particles(self):
        particle_number = self.PART_NUM
        Z_dim = particle_number
        bo_shape =  self.particle_list[0].bo_list.shape
        temp_all_parts = np.zeros((bo_shape[0], bo_shape[1], Z_dim))

        for pn in range(particle_number):
            particle = self.get_particle(pn)
            temp_all_parts[:,:,pn] = self.particle_list[pn].bo_list

        return temp_all_parts

    def collect_params(self):
        #list comprehention form: new_list = [expression for member in iterable]
        output = [particle.bo for particle in self.particle_list]
        if len(output)>0:
            self.params_to_ship = np.reshape(output, (self.PART_NUM, self.p))
        else:
            self.params_to_ship = []
        #print("self.params_to_ship=",self.params_to_ship)

    def collect_history_ids(self):
        self.machine_history_ids_to_ship = np.zeros((self.PART_NUM))
        self.particle_history_ids_to_ship = np.zeros((self.PART_NUM))
        for pn in range(self.PART_NUM):
            self.machine_history_ids_to_ship[pn] = self.particle_list[pn].particle_id_history[0]
            self.particle_history_ids_to_ship[pn] = self.particle_list[pn].particle_id_history[1]
    
    def get_pf_parameter_means(self):
        self.params_to_ship_mean = np.mean(self.params_to_ship, axis=0)
        
    def get_pf_parameter_cov(self):
        self.params_to_ship_cov = np.cov(self.params_to_ship.T)
         
    def compute_particle_kernel_weights(self, mean_params, cov_parmas):

        # get shard inverse covariances * shard count
        
        #print("rank of cov_parmas",[np.linalg.matrix_rank(V_s) for V_s in cov_parmas])
        shard_cov_inv_list = [np.linalg.inv(V_s*self.shards) for V_s in cov_parmas]
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
