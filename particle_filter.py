import pf_models as pf
import numpy as np
#from scipy.stats import truncnorm
#from scipy.stats import norm
#from scipy.stats import truncnorm
#from scipy.special import logsumexp
#from tqdm import tqdm
import numpy as np
import pandas as pd
#import csv
from matplotlib import pyplot as plt

class particle_filter:
#particle filter class    
    def __init__(self, dat, PART_NUM, model, sample_method, pf_rank = 0, run_number = 0):
        self.PART_NUM=PART_NUM
        self.particle_list=list()
        self.model=model
        self.sample_method=sample_method
        self.dat=dat
        self.data_keys=dat['data_keys']
        self.time_values = [0]#dat['time_value']
        self.unique_time_values = np.unique(self.time_values)
        self.rank = pf_rank
        self.run_number = run_number
        #print(self.unique_time_values)
        if self.model=="probit":
            #create particles
            self.X=dat.X
            self.Y=dat.Y
            self.p=dat.p
            for pn in range(self.PART_NUM):
                temp_particle=pf.probit_particle(np.zeros((1,p)),np.identity(p)*1000.0, pn)
                self.particle_list.append(temp_particle)
                
        elif self.model== "probit_sin_wave":
            print("working on model ", model)
            self.X=dat['X_matrix']
            self.Y=dat['Y']
            self.p=dat['p']
            self.N=dat['N']
            #self.time_values = dat['time_value']
            
            self.shards=dat['shards']
            for pn in range(self.PART_NUM):
                temp_particle=pf.probit_sin_wave_particle( np.array(dat['b'][0]), dat['B'], pn)#( np.array(dat['b'][0]), dat['B'], pn)
                temp_particle.set_N(self.N)
                temp_particle.set_Zi(self.X)
                #temp_particle.set_bo_list(int(1+max(self.time_values)))
                temp_particle.set_shard_number(self.shards)
                self.particle_list.append(temp_particle)
        else:
            print(model, " not implemented yet...")
                        
    def run_particle_filter(self):
        #single interation of P particles
        self.not_norm_wts=np.ones(self.PART_NUM)
        if self.model=="probit":
            for pn in range(self.PART_NUM):
                #update particle
                self.particle_list[pn].update_particle(self.X, self.Y)
                self.not_norm_wts[pn]=particle_list[pn].evaluate_likelihood(self.X, self.Y)
        if self.model=="probit_sin_wave":
            x_keys = self.data_keys
            y_keys = self.data_keys
            
            for n in range(len(self.unique_time_values)):#self.X.shape[0]):
                n_check_point = np.floor(self.N*0.10)
                if n % n_check_point ==0:
                    print(str(100*n/self.N)+"% of total file complete...")
                    print(str(100*n/self.X.shape[0])+"% of shard complete...")
                    
                for pn in range(self.PART_NUM):
                    if self.sample_method=='importance':
                        row_index = self.time_values == self.unique_time_values[n]
                        #print(self.time_values)
                        #print(type(self.X))

                        self.particle_list[pn].update_particle_importance(
                            self.X[row_index,:], 
                            self.Y[row_index], 
                            int(x_keys[n].split(":")[0]),
                            self.unique_time_values[n]
                        )
                    else:
                        self.particle_list[pn].update_particle(self.X[n,:], self.Y[n], n)
                    self.not_norm_wts[pn]=self.particle_list[pn].evaluate_likelihood(self.X[n,:], self.Y[n])
                self.shuffle_particles()
            #print("max(self.not_norm_wts)=", max(self.not_norm_wts))
        return
    
    def shuffle_particles(self):#,n):
        #print("enter shuffle_particle")
        #self.not_norm_wts=self.not_norm_wts*100000000
        #(np.exp(xxx- np.max(xxx))+np.exp(np.max(xxx)))/
        
        #exp_max_val = np.exp(max_val)
        
        top    = np.exp(self.not_norm_wts)
        bottom = np.sum(top)
        if bottom == 0:
            max_val=np.max(self.not_norm_wts)
            top    = np.exp(self.not_norm_wts - max_val)
            bottom = np.sum(top)
            norm_wts=top/bottom#np.exp(logsumexp(self.not_norm_wts)))
        else:
            norm_wts=top/bottom#np.exp(logsumexp(self.not_norm_wts)))
        #print("not_norm  = ", top)
        #print("norm_wts  = ", norm_wts)
        #if n==1:
        #    print("max_val=",max_val)
        #    print("self.not_norm_wts - max_val=",self.not_norm_wts - max_val)
        #    print("logsumexp(self.not_norm_wts)=",logsumexp(self.not_norm_wts))
        #    print("top=",top)
        #    print("bottom=",bottom)
        if np.sum(norm_wts) < 0.999999 or np.sum(norm_wts) > 1.000001:
            print("norm_wts=", norm_wts)
                #easy_hist(norm_wts, "norm weights")
                
        particles_kept=np.random.choice(range(self.PART_NUM),size=self.PART_NUM, p=norm_wts)
        temp_index=np.zeros(self.PART_NUM)
        temp_index.astype(int)
        for pn in range(self.PART_NUM):
            temp_index[particles_kept[pn]]+=1
            
        #print(particles_kept)
        #print(temp_index)
        #print("np.where(temp_index==0)=",np.where(temp_index==0))
        
        not_chosen=np.where(temp_index==0)[0]
        #print("not_chosen=",not_chosen)
        for nci in range(len(not_chosen)):
            #print("not_chosen[nci]=",not_chosen[nci])
            for ti in range(self.PART_NUM):
                break_ti_for=False
                while(temp_index[ti]>=2):
                    #print("duplicated",ti)
                    temp_index[ti]-=1
                    #copy_particle_values(not_this_particle)
                    self.particle_list[not_chosen[nci]].copy_particle_values(self.particle_list[ti])
                    break_ti_for=True
                    break
                if break_ti_for: break
        #print("exit shuffle_particle")
        #print("max(norm_wts)=", max(norm_wts))

        
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
    
    def update_data(self, dat, run_number):#dat_X_matrix, dat_Y):
            self.run_number = run_number
            self.X = dat['X_matrix']#dat_X_matrix#dat['X_matrix']
            self.Y = dat['Y']#dat_Y#dat['Y']
            #print("dat['time_value'] = ", dat['time_value'])
            
            self.time_values = dat['time_value']
            # [x+max(self.time_values) for x in dat['time_value']]# dat['time_value'] + max(self.time_values)
            self.unique_time_values = np.unique(self.time_values)
            for pn in range(self.PART_NUM):
                self.particle_list[pn].set_bo_list(int(1+max(self.time_values)))
                self.particle_list[pn].this_time = 0
            #self.data_keys=dat['data_keys']
            #self.pf_obj[m].update_data(self.data['shard_'+str(m)])
    
    def update_params(self, updated_params):
        for i in range(len(self.particle_list)):
            self.particle_list[i].bo = updated_params[i]
    
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
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%write_bo_list called successfully")
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
        )#, compression='bz2')
        
    def get_temp_all_particles(self):
        particle_number = self.PART_NUM #len(parallel_com_obj[0].particle_list)
        #number_of_shards = len(parallel_com_obj)
        Z_dim = particle_number #number_of_shards * particle_number
        #total_time_steps = len( parallel_com_obj[0].get_particle(0).bo_machine_list[:,0] )
        bo_shape =  self.particle_list[0].bo_list.shape #parallel_com_obj[0].particle_list[0].bo_list.shape
        print("base bo list shape:", bo_shape )
        temp_all_parts = np.zeros((bo_shape[0], bo_shape[1], Z_dim))

        #counter = 0
        #for sn in range(len(parallel_com_obj)):
            #print(sn)
            #print(
            #    "(parallel_com_obj[i].particle_list[0].shape) = " ,
            #    parallel_com_obj[sn].particle_list[0].bo_list.shape
            #)
        for pn in range(particle_number):
            particle = self.get_particle(pn) #parallel_com_obj[sn].get_particle(pn)
            temp_all_parts[:,:,pn] = self.particle_list[pn].bo_list #parallel_com_obj[sn].particle_list[pn].bo_list
            #counter+=1
                
        return temp_all_parts
    
    def collect_params(self):
        self.params_to_ship = np.zeros((self.PART_NUM, self.p))
        for pn in range(self.PART_NUM):
            self.params_to_ship[pn,:] = self.particle_list[pn].bo
        print("params collected...")