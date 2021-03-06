from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import particle_filter
import pf_models as pfm
import math as m
import random
from scipy.optimize import linprog
from scipy.stats import multivariate_normal

#import seaborn as sns

#from joblib import Parallel, delayed
#from joblib import parallel_backend


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

#def particle_filter_init_wrapper(f_shards_data, f_part_num, f_model, f_sample_method):
#    pfo = particle_filter.particle_filter(f_shards_data, f_part_num, f_model, f_sample_method)
#    return(pfo)
#    
#def run_particle_filter_wrapper(pfo):
#    pfo.run_particle_filter()
#    return(pfo)
#
#def update_data_wrapper(pfo, f_data_X_matrix, f_data_Y):
#    pfo.update_data(f_data_X_matrix, f_data_Y)
#    return(pfo)
#
#def plot_CMC_parameter_path_(ObsN_ParamN_Part_N, predictor_names, ground_truth = None):
#    print("plot_parameter_path...")
#    total_time_steps, param_num, Z_dim = ObsN_ParamN_Part_N.shape
#    #params=list()
#    params     = np.nanmean(ObsN_ParamN_Part_N, axis=2)
#    params_std = np.nanstd(ObsN_ParamN_Part_N, axis=2)
#    
#    plot_column_count = 4
#    plot_row_count = int(m.ceil(param_num/plot_column_count))
#    fig_outer, axes_outer = plt.subplots(nrows=plot_row_count, ncols=plot_column_count)
#
#    
#    for par_n in range(param_num):
#        avg_param_0=params[:,par_n]
#        avg_param_0=pd.Series(avg_param_0).fillna(method='ffill')
#        avg_param_0=pd.Series(avg_param_0).fillna(method='bfill')
#            
#        std_parma_0=params_std[:,par_n]
#        std_parma_0=pd.Series(std_parma_0).fillna(method='ffill')
#        std_parma_0=pd.Series(std_parma_0).fillna(method='bfill')                
#        
#        above=np.add(avg_param_0,std_parma_0*2)
#        below=np.add(avg_param_0,-std_parma_0*2)
#
#        x = np.arange(len(avg_param_0)) 
#        
#        ax_row = int(m.floor(par_n/plot_column_count))
#        ax_col = par_n % plot_column_count
#        
#        axes_outer[ax_row][ax_col].fill_between(x, below, above, facecolor='green',  alpha=0.3)
#        axes_outer[ax_row][ax_col].plot(x,avg_param_0, 'b', alpha=.8)
#        axes_outer[ax_row][ax_col].axhline(y=0.0, color='r', linestyle='-')
#        axes_outer[ax_row][ax_col].title.set_text(predictor_names[par_n])
#        
#        if ground_truth is not None:
#             axes_outer[ax_row][ax_col].plot(x, ground_truth.iloc[:,par_n], '--')
#                
#    plt.show()
#
#        
#def plot_CMC_parameter_path_by_shard(data, pf_obj, PART_NUM, number_of_shards, particle_prop=0.01):
#    print("plot_parameter_path...")
#    print(data['predictors'])
#    color_list = sns.color_palette(None, data['parallel_shards'])
#    param_num = len(data['predictors'])#pf_obj[0].get_particle(0).bo_machine_list.shape[1]
#    total_time_steps = len( pf_obj[0].get_particle(0).bo_machine_list[:,0] )
#    max_val=max(int(PART_NUM*particle_prop), 1)
#    particle_indices = np.random.choice(PART_NUM, max_val)
#    print("entering first loop")
#    for par_n in range(param_num):
#        
#        min_tic = 9999999999999
#        max_tic = -9999999999999
#        print("entering second loop")
#        for sn in range(data['parallel_shards']):
#            print("working on shard " + str(sn))
#            #plt.title('parameter {}, shard {}'.format(par_n+1,sn+1))
#            truth = data['b'][:,par_n]
#            # the points on the x axis for plotting
#            x = np.arange(len(data['b'][:,par_n])) 
#            plt.plot(x,truth,'black')
#            params=list()
#            Z_dim = number_of_shards * PART_NUM
#            temp_all_parts = np.zeros((total_time_steps, param_num, Z_dim))
#            print("temp_all_parts.shape=", temp_all_parts.shape)
#            temp_all_parts[temp_all_parts==0]=np.NaN
#            counter=0
#            print("entering third loop")
#            for pn in range(len(particle_indices)):
#                particle = pf_obj[sn].get_particle(particle_indices[pn])
#                temp_all_parts[:,:,counter] = particle.bo_machine_list.copy()
#                counter+=1
#
#            params=np.nanmean(temp_all_parts, axis=2)
#            params_std=np.nanstd(temp_all_parts, axis=2)
#
#            avg_param_0=avg_param_0=params[:,par_n]
#            avg_param_0=pd.Series(avg_param_0).fillna(method='ffill')
#            avg_param_0=pd.Series(avg_param_0).fillna(method='bfill')
#
#            std_parma_0=params_std[:,par_n]
#            std_parma_0=pd.Series(std_parma_0).fillna(method='ffill')
#            std_parma_0=pd.Series(std_parma_0).fillna(method='bfill')                
#            above=np.add(avg_param_0,std_parma_0*2)
#            below=np.add(avg_param_0,-std_parma_0*2)
#            x = np.arange(len(avg_param_0)) 
#
#            plt.fill_between(x, below, above, facecolor=color_list[sn],  alpha=0.3)
#            plt.plot(x,avg_param_0, c=color_list[sn], marker='.', alpha=.8)
#            #for line_tick in self.params['epoch_at']:
#            #    plt.axvline(x=line_tick, color='r', alpha=0.25)
#            
#            min_max_scale=1.0
#            min_tic = min( min_tic, m.floor(np.min(below)/min_max_scale)*min_max_scale)
#            max_tic = max( max_tic, m.ceil(np.max(above)/min_max_scale)*min_max_scale)
#            #if np.min(below) < 0:
#            #    min_tic = m.ceil(np.min(below)/10.0)*10.0
#            #else:
#            #    min_tic= m.floor(np.min(below)/10.0)*10.0 #  floor(np.min(below))#np.min([np.min(below),np.min(truth)])
#            #
#            #if np.max(above) < 0:
#            #    max_tic = m.floor(np.min(above)/10.0)*10.0
#            #else:
#                
#            #max_tic=np.max([np.max(above),np.max(truth)])
#        print("end of second loop")
#        print("check point")    
#        #plt.yticks(np.linspace(start=min_tic, stop=max_tic, num= 1 + round(max_tic-min_tic)/min_max_scale ))
#        print(1)
#        plt.grid(True)
#        #print(2)
#        plt.title(data['predictors'][par_n])      
#        print(3)
#        plt.show()
#        print("end point")
#        
#def shuffel_embarrassingly_parallel_particles(data,
#                                              PART_NUM,
#                                              pf_obj,
#                                              machine_list=None, 
#                                              method='uniform', 
#                                              wass_n=None):
#            
#    all_particles=list()
#    for m in range(data['parallel_shards']):
#        for p in range(PART_NUM):
#
#            first  = np.array(data['b'][0]), 
#            second = data['B']
#            temp_particle = pfm.probit_sin_wave_particle(first, second , -1)
#
#            temp_particle.copy_particle_values(pf_obj[m].particle_list[p])
#            all_particles.append(temp_particle)
#         
#    if method == "uniform":        
#        for m in range(data['parallel_shards']):
#            for p in range(PART_NUM):
#                index = random.randint(0, len(all_particles)-1)
#                first = all_particles[index] 
#                pf_obj[m].particle_list[p].copy_particle_values(first)
#
#    return pf_obj

def get_Communal_Monte_Carlo_mu_Sigma(all_shard_params, particle_count, shard_count):
    
    dim = all_shard_params[0].shape[1]
    Sig_i_inv_x_mu_i = np.zeros((particle_count,dim,shard_count))
    
    Sig_i_inv = np.zeros((dim,dim,shard_count))

    for m in range(shard_count):
        shard_cov=np.cov(all_shard_params[m].T)*shard_count
        
        if np.linalg.matrix_rank(shard_cov) == shard_cov.shape[1]:
                Sig_i_inv[:,:,m] = np.linalg.inv(shard_cov)
        else: 
            I = np.identity(shard_cov.shape[1])
            diag_values = shard_cov.diagonal()
            max_var = np.nanmax(diag_values)
            I_s = I*max_var/100
            Sigma = shard_cov + I_s
            Sig_i_inv[:,:,m] = np.linalg.inv(Sigma)

        Sig_i_inv_x_mu_i[:,:,m] = np.matmul(Sig_i_inv[:,:,m], all_shard_params[m].T).T
        
    summed_Sig_i_inv_x_mu_i = np.sum(np.sum(Sig_i_inv_x_mu_i, axis=0), axis=1)
    V_inv = np.sum(Sig_i_inv, axis=2)*particle_count          
    V = np.linalg.inv(V_inv)
    
    # ensure V maintains positive semi-definite property
    # code here
    
    combined_mean = np.matmul(V, summed_Sig_i_inv_x_mu_i.T).T

    return combined_mean, V

def shuffel_embarrassingly_parallel_params(all_shard_params, weighting_type="uniform_weighting",):
    
    unlisted = list()
    particle_count = len(all_shard_params[0])
    shard_count = len(all_shard_params)

    #use appropriate weighting scheme
    if weighting_type == "uniform_weighting":        
        rows = np.random.randint(particle_count*shard_count, size = particle_count*shard_count)
        sampled_unlisted = np.vstack(all_shard_params)[rows,:]
        output = np.array_split(sampled_unlisted, shard_count)
        return(output)
        
    if weighting_type == "kernel_weighting":
        #print("if weighting_type == kernel_weighting")
        """
            this will determin the mean and covariance of the model estimated parameters and weight 
            particles according to a gaussian density for resampling purposes.
            
            THE NEXT 5 LINES OF CODE REALLY SHOULD LIVE IN THE pf_model.py FILE.
        """
        
        mu, Sigma = get_Communal_Monte_Carlo_mu_Sigma(all_shard_params, particle_count, shard_count)
        #for m in range(shard_count):
        #    x = multivariate_normal.logpdf(all_shard_params[m], mean=mu, cov=Sigma)
        unlisted = np.vstack(all_shard_params)
        #print(unlisted.shape)
        #print(mu.shape)
        #print(Sigma.shape)
        x = multivariate_normal.logpdf(unlisted, mean=mu, cov=Sigma)
        finite_values = x[np.isfinite(x)]
        finite_max = np.nanmax(finite_values)
        finite_min = np.nanmin(finite_values)
        x[x>finite_max] = finite_max
        x[x<finite_min] = finite_min
        x[np.isnan(x)] = finite_min
        normalized_kernel_weights = np.exp(x - logsumexp(x))
        normalized_kernel_weights = normalized_kernel_weights/np.sum(normalized_kernel_weights)
        if any(np.isnan(x)):
            normalized_kernel_weights = np.ones(len(x))/len(x)

        idx=list(range(len(unlisted)))
        rows = np.random.choice(idx, size = len(idx), p=normalized_kernel_weights)
        sampled_unlisted = unlisted[rows,:]
        output = np.array_split(sampled_unlisted, shard_count)
        return(output)
    
    if weighting_type == "normal_consensus_weighting":
        
        combined_mean, V = get_Communal_Monte_Carlo_mu_Sigma(all_shard_params, particle_count, shard_count)
        
        sampled_unlisted = np.random.multivariate_normal(
            combined_mean, V, size=particle_count*shard_count, check_valid='warn', tol=1e-8)
        
        output = np.array_split(sampled_unlisted, shard_count)
        return(output)
        

def convert_to_list_of_type(params_nd_list, f_type = float):
    all_shards = list()
    for s in range(len(params_nd_list)):
        all_particles = list()
        for p_n in range(len(params_nd_list[s])):
            float_params = np.array(params_nd_list[s][p_n]).tolist()
            all_particles.append(float_params)
        all_shards.append(all_particles)
    
    return all_shards
          
#class embarrassingly_parallel:
#    
#    def __init__(self, data, params):
#        
#        self.data=data
#        self.pf_obj=list()
#        self.params=params
#        self.model=params['model']        
#        self.number_of_shards=params['shards']
#        self.sample_method= params['sample_method']
#        self.PART_NUM=params['particles_per_shard']
#        print('initializing workers')
#        self.parallel_workers = Parallel(n_jobs=self.number_of_shards)
#        print('done\ninitializing particle filters')
#        #self.parallel_workers(delayed(np.sqrt)(m) for m in range(self.number_of_shards))
#        
#        
#        #self.pf_obj = self.parallel_workers(delayed(particle_filter.particle_filter
#        #                               )(self.data['shard_'+str(m)], 
#        #                                 self.PART_NUM, 
#        #                                 self.model,
#        #                                 self.sample_method
#        #                                ) for m in range(self.number_of_shards)
#        #                       )
#        self.pf_obj = self.parallel_workers(
#            delayed(
#                particle_filter_init_wrapper
#            )(
#                f_shards_data = self.data['shard_'+str(m)], 
#                f_part_num = self.PART_NUM, 
#                f_model = self.model,
#                f_sample_method = self.sample_method
#            ) for m in range(self.number_of_shards)
#        )
#        print('initialized')
#        #self.pf_obj = self.parallel_workers(delayed(run_particle_filter_wrapper
#        #                               )(pfo = self.pf_obj[m]
#        #                                ) for m in range(self.number_of_shards)
#        #                       )
#    
#    
#    def run_batch(self, data):
#        
#        self.parallel_workers = Parallel(n_jobs=self.number_of_shards)
#
#        self.pf_obj = self.parallel_workers(
#            delayed(
#                update_data_wrapper
#            )(
#                pfo = self.pf_obj[m], 
#                f_data_X_matrix = data['shard_'+str(m)]['X_matrix'],
#                f_data_Y = data['shard_'+str(m)]['Y']
#            ) for m in range(self.number_of_shards)
#        )
#        
#        #self.pf_obj = self.parallel_workers(
#        #    delayed(
#        #        run_particle_filter_wrapper
#        #    )(
#        #        pfo = self.pf_obj[m]
#        #    ) for m in range(self.number_of_shards)
#        #)
#            
#    def plot_parameter_path(self, particle_prop=0.01):
#        print("plot_parameter_path...")
#        
#        param_num=self.pf_obj[0].get_particle(0).bo_list.shape[1]
#        print("param_num=",param_num)
#        total_time_steps = len(self.pf_obj[0].get_particle(0).bo_list[:,0])
#        print("total_time_steps=",total_time_steps)
#        params=list()
#        particle_indices = np.random.choice(self.PART_NUM, max(int(self.PART_NUM*particle_prop), 1))
#        print("particle_indices=",particle_indices)
#        
#        for os in range(param_num):
#          temp_all_parts = np.zeros((len(particle_indices), total_time_steps))
#          #print("data keys = ", self.data.keys())
#          for sn in range(self.data['parallel_shards']):
#            for pn in range(len(particle_indices)):
#              particle=self.pf_obj[sn].get_particle(particle_indices[pn])
#              p_temp = particle.bo_list[:,os].copy()
#              p_temp[np.isnan(p_temp)]=0
#              temp_all_parts[pn,:]=np.add(temp_all_parts[pn,:],p_temp)
#                     
#          params.append(temp_all_parts)
#        
#        for par_n in range(param_num):
#            avg_param_0=np.mean(params[par_n], axis=0)
#            std_parma_0=np.std(params[par_n], axis=0)
#            above=np.add(avg_param_0,std_parma_0*2)
#            below=np.add(avg_param_0,-std_parma_0*2)
#            
#            truth=self.data['b'][:,par_n]
#            
#            x = np.arange(len(avg_param_0)) # the points on the x axis for plotting
#            
#            fig, ax1 = plt.subplots()
#            plt.plot(x,truth,'black')
#            ax1.fill_between(x, below, above, facecolor='green',  alpha=0.3)
#            plt.plot(x,avg_param_0, 'b', alpha=.8)
#            for line_tick in self.params['epoch_at']:
#                plt.axvline(x=line_tick, color='r', alpha=0.25)
#            min_tic=np.min([np.min(below),np.min(truth)])
#            max_tic=np.max([np.max(above),np.max(truth)])
#            plt.yticks(np.linspace(start=min_tic, stop=max_tic, num=12))
#            plt.grid(True)
#            plt.show()
#
#    def plot_CMC_parameter_path(self, particle_prop=0.01):
#        print("plot_parameter_path...")
#        
#        param_num=self.pf_obj[0].get_particle(0).bo_list.shape[1]
#        total_time_steps = len(self.pf_obj[0].get_particle(0).bo_list[:,0])
#        params=list()
#        particle_indices = np.random.choice(self.PART_NUM, max(int(self.PART_NUM*particle_prop), 1))
#        
#        Z_dim=self.number_of_shards*self.PART_NUM
#        temp_all_parts = np.zeros((total_time_steps, param_num, Z_dim))
#        temp_all_parts[temp_all_parts==0]=np.NaN
#
#        counter=0
#        for sn in range(self.data['parallel_shards']):
#            for pn in range(len(particle_indices)):
#                particle=self.pf_obj[sn].get_particle(particle_indices[pn])
#                temp_all_parts[:,:,counter] = particle.bo_list.copy()
#                counter+=1
#
#        params=np.nanmean(temp_all_parts, axis=2)
#        params_std=np.nanstd(temp_all_parts, axis=2)
#
#        for par_n in range(param_num):
#            avg_param_0=params[:,par_n]
#            avg_param_0=pd.Series(avg_param_0).fillna(method='ffill')
#            avg_param_0=pd.Series(avg_param_0).fillna(method='bfill')
#                
#            std_parma_0=params_std[:,par_n]
#            std_parma_0=pd.Series(std_parma_0).fillna(method='ffill')
#            std_parma_0=pd.Series(std_parma_0).fillna(method='bfill')                
#            
#            above=np.add(avg_param_0,std_parma_0*2)
#            below=np.add(avg_param_0,-std_parma_0*2)
#            
#            truth=self.data['b'][:,par_n]
#            
#            x = np.arange(len(avg_param_0)) 
#            
#            fig, ax1 = plt.subplots()
#            plt.plot(x,truth,'black')
#            ax1.fill_between(x, below, above, facecolor='green',  alpha=0.3)
#            plt.plot(x,avg_param_0, 'b', alpha=.8)
#            for line_tick in self.params['epoch_at']:
#                plt.axvline(x=line_tick, color='r', alpha=0.25)
#                min_tic=np.min([np.min(below),np.min(truth)])
#                max_tic=np.max([np.max(above),np.max(truth)])
#                plt.yticks(np.linspace(start=min_tic, stop=max_tic, num=12))
#                plt.grid(True)
#                    
#            plt.show()
#            
#
#    def plot_CMC_parameter_path_by_shard(self, particle_prop=0.01):
#        print("plot_parameter_path...")
#        color_list = sns.color_palette(None, self.data['parallel_shards'])
#        param_num=self.pf_obj[0].get_particle(0).bo_machine_list.shape[1]
#        total_time_steps = len(self.pf_obj[0].get_particle(0).bo_machine_list[:,0])
#        max_val=max(int(self.PART_NUM*particle_prop), 1)
#        particle_indices = np.random.choice(self.PART_NUM, max_val)
#        for par_n in range(param_num):
#            
#
#            for sn in range(self.data['parallel_shards']):
#                plt.title('parameter {}, shard {}'.format(par_n+1,sn+1))
#                truth=self.data['b'][:,par_n]
#                # the points on the x axis for plotting
#                x = np.arange(len(self.data['b'][:,par_n])) 
#                plt.plot(x,truth,'black')
#                params=list()
#                Z_dim=self.number_of_shards*self.PART_NUM
#                temp_all_parts = np.zeros((total_time_steps, param_num, Z_dim))
#                temp_all_parts[temp_all_parts==0]=np.NaN
#                counter=0
#                for pn in range(len(particle_indices)):
#                    particle=self.pf_obj[sn].get_particle(particle_indices[pn])
#                    temp_all_parts[:,:,counter] = particle.bo_machine_list.copy()
#                    counter+=1
#
#                params=np.nanmean(temp_all_parts, axis=2)
#                params_std=np.nanstd(temp_all_parts, axis=2)
#
#                avg_param_0=avg_param_0=params[:,par_n]
#                avg_param_0=pd.Series(avg_param_0).fillna(method='ffill')
#                avg_param_0=pd.Series(avg_param_0).fillna(method='bfill')
#
#                std_parma_0=params_std[:,par_n]
#                std_parma_0=pd.Series(std_parma_0).fillna(method='ffill')
#                std_parma_0=pd.Series(std_parma_0).fillna(method='bfill')                
#                above=np.add(avg_param_0,std_parma_0*2)
#                below=np.add(avg_param_0,-std_parma_0*2)
#                x = np.arange(len(avg_param_0)) 
#
#                plt.fill_between(x, below, above, facecolor=color_list[sn],  alpha=0.3)
#                plt.plot(x,avg_param_0, c=color_list[sn], marker='.', alpha=.8)
#                for line_tick in self.params['epoch_at']:
#                    plt.axvline(x=line_tick, color='r', alpha=0.25)
#                min_tic=np.min([np.min(below),np.min(truth)])
#                max_tic=np.max([np.max(above),np.max(truth)])
#                plt.yticks(np.linspace(start=min_tic, stop=max_tic, num=12))
#                plt.grid(True)
#                    
#            plt.show()
#                
#            
#    def shuffel_embarrassingly_parallel_particles(self, 
#                                                  machine_list=None, 
#                                                  method=None, 
#                                                  wass_n=None):
#                
#        self.all_particles=list()
#        for m in range(self.data['parallel_shards']):
#            for p in range(self.PART_NUM):
#
#                first  = np.array(self.data['b'][0]), 
#                second = self.data['B']
#                temp_particle = pfm.probit_sin_wave_particle(first, second , -1)
#
#                temp_particle.copy_particle_values(self.pf_obj[m].particle_list[p])
#                self.all_particles.append(temp_particle)
#
#        if method == "wasserstein":
#            print("computing waserstein barrycenter...")
#            print("collecting parameter info from shards...")
#            f_wts = 0.01+self.get_approx_shard_wasserstein_barycenter(wass_n)[0]
#            print("data successfully prepared...")  
#            print('machine weights: ', f_wts)
#            temp_all_wts=np.repeat(f_wts,self.PART_NUM)
#            all_wts = temp_all_wts/np.sum(temp_all_wts)
#            index_vals = np.random.choice(range(len(all_wts)), len(all_wts), p=all_wts)
#            count=0
#            for m in range(self.data['parallel_shards']):
#                for p in range(self.PART_NUM):
#                    first=self.all_particles[index_vals[count]] 
#                    self.pf_obj[m].particle_list[p].copy_particle_values(first)
#                    count+=1
#                    
#        if method == "uniform":        
#            for m in range(self.data['parallel_shards']):
#                for p in range(self.PART_NUM):
#                    index=randint(0, len(self.all_particles)-1)
#                    first=self.all_particles[index] 
#                    self.pf_obj[m].particle_list[p].copy_particle_values(first)
#                                    
#    def update_data(self, data_from_outside):
#        self.data = data_from_outside
#        
#    def prep_particles_for_wassber(self, n):
#
#        K=self.number_of_shards
#        myTheta=list()
#        theta = np.zeros((n*K,self.params['p']))#['omega_shift'])))
#        row=0
#        for m in range(self.number_of_shards):
#            myTheta.append(np.zeros((n,self.params['p'])))#['omega_shift']))))
#            for p in range(n):
#                particle_index = randint(0, self.PART_NUM-1)
#                theta[row,:] = myTheta[m][p,:] = self.pf_obj[m].particle_list[particle_index].bo
#                row+=1
#        return theta, myTheta
#    
#    def get_wasserstein_barycenter(self, n, K, d, theta, myTheta):
#        
#        Nk1 = np.ones((1,n))    # one vector of length n
#        N1  = np.ones((K*n,1))  # one vector of length K*n
#        IN  = np.identity(K*n)  # Diagonal matrix of size K*n
#        INk = np.identity(n)    # Diagonal matrix of size n
#        
#        # cost vector
#        # each thetak is the matrix of samples from subset posterior k=1,...,K
#        # theta is the overall sample matrix, formed by stacking the thetak
#        #cost = c()
#        cost = np.array([])
#        #for (i in 1:K){
#        for i in range(K):
#            thetak = myTheta[i]
#            theta_theta_transpose = np.matmul(theta, np.transpose(theta))
#            diag_of_theta_theta_transpose=np.diag(theta_theta_transpose)
#            diag_of_theta_theta_transpose=diag_of_theta_theta_transpose.reshape((len(diag_of_theta_theta_transpose),1))
#            sec1 = np.matmul(diag_of_theta_theta_transpose, Nk1)
#            
#            thetak_thetak_transpose=np.matmul(thetak, np.transpose(thetak))
#            tdiag_thetak_thetak_transpose=np.transpose(np.diag(thetak_thetak_transpose))
#            tdiag_thetak_thetak_transpose=tdiag_thetak_thetak_transpose.reshape((1,len(tdiag_thetak_thetak_transpose)))
#            sec2 = np.matmul(N1, tdiag_thetak_thetak_transpose)
#            
#            theta_thetak_transpose = np.matmul(theta, np.transpose(thetak))
#            sec3=-2.0*theta_thetak_transpose
#            Mk = sec1 + sec2 + sec3
#    
#            cost = np.concatenate((cost, np.transpose(Mk).reshape(1,Mk.size)[0]))
#    
#        cost = np.concatenate((cost, np.zeros(K*n)))
#        
#        # constraint matrix 
#        # A1-A6 are the 6 components of the A constraint matrix in the first Srivastava paper.
#        A1 = np.zeros((1,(K*n)**2)) 
#        A2 = np.ones((1, K*n))
#        a3=np.kron(Nk1, IN)
#        A3 = np.kron(np.identity(K), a3)
#        A4 = np.kron(np.ones((K,1)), IN)
#        a5 = np.transpose(np.kron(INk, N1))
#        A5 = np.transpose(np.kron(np.identity(K), np.transpose(a5)))
#        A6 = np.zeros((K*n, K*n))
#        cbA1A2  = np.concatenate((A1, A2), axis=1)
#        cbA3mA4 = np.concatenate((A3, -A4), axis=1)
#        cbA5A6  = np.concatenate((A5, A6), axis=1)
#        A = np.concatenate((cbA1A2, cbA3mA4, cbA5A6)) 
#    
#        # the right hand side of constraints for the matrix A
#        consRHS = np.concatenate((np.ones(1), 
#                                  np.zeros((K**2)*n), 
#                                  1.0/n*np.ones(K*n)))
#        
#        # direction of the constraints.
#        
#        # note the constraint that the output vector 'a' of probabilities is >=0 is not explicitly put into the solver
#        # since this solver already inputs this constraint, other solver packages MAY REQUIRE THIS EXPLICITLY
#        
#        out = linprog(cost, A_eq=A, b_eq=consRHS, options={"disp": False})
#        # this just extracts the relevant output containing the posterior vector of probabilities 'a' for the consensus posterior
#        end = len(out['x'])
#        start = len(out['x'])-K*n
#        sol = out['x'][start:end]
#        
#        return sol
#    
#    def get_approx_shard_wasserstein_barycenter(self, f_n):
#        wsbc_num       = 20
#        wssber_wts     = {}
#        theta_list     = list()
#        myTheta_list   = list()
#        apx_bc_mac_wts = list()
#        f_K = self.number_of_shards
#        f_d = len(self.params['p'])#['omega_shift'])
#        for i in range(wsbc_num):
#            theta, myTheta = self.prep_particles_for_wassber(f_n)
#            wssber_wts[i] = self.get_wasserstein_barycenter(f_n, f_K, f_d, theta, myTheta)
#            temp_wts=np.sum(np.array_split(wssber_wts[i], f_K), axis=1)
#            apx_bc_mac_wts.append(temp_wts)
#            theta_list.append(theta)
#            myTheta_list.append(myTheta)
#        
#        f_shard_wts_0 = np.array(apx_bc_mac_wts).reshape((wsbc_num,f_K))
#        f_shard_wts   = np.mean(f_shard_wts_0, axis=0)
#        return f_shard_wts, apx_bc_mac_wts, theta_list, myTheta_list