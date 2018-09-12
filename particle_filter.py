import pf_models as pf
import numpy as np
#from scipy.stats import truncnorm
#from scipy.stats import norm
#from scipy.stats import truncnorm
#from scipy.special import logsumexp
import numpy as np
import pandas as pd
#import csv
from matplotlib import pyplot as plt

class particle_filter:
#particle filter class    
    def __init__(self, dat, PART_NUM, model, sample_method):
        self.PART_NUM=PART_NUM
        self.particle_list=list()
        self.model=model
        self.sample_method=sample_method
        self.dat=dat
        if self.model=="probit":
            #create particles
            #PART_NUM=20
            self.X=dat.X
            self.Y=dat.Y
            self.p=dat.p
            for pn in range(self.PART_NUM):
                temp_particle=pf.probit_particle(np.zeros((1,p)),np.identity(p)*1000.0, pn)
                self.particle_list.append(temp_particle)
                
        elif self.model== "probit_sin_wave":
            print("working on model ", model)
            self.X=dat['X']
            self.Y=dat['Y']
            self.p=dat['p']
            self.N=dat['N']
            self.batch_num=dat['batch_number']
            self.shards=dat['shards']
            for pn in range(self.PART_NUM):
                #print("particle number:",pn)
                #print("dat.b=",dat.b)
                #print("dat.b[0]=",dat.b[0])
                temp_particle=pf.probit_sin_wave_particle( np.array(dat['b'][0]), dat['B'], pn)#dat.b[0].dot(np.ones((self.p,1)))
                temp_particle.set_N(self.N)
                temp_particle.set_Zi(self.X)
                temp_particle.set_bo_list()
                temp_particle.set_shard_number(self.shards)
                self.particle_list.append(temp_particle)
                #print("particle_list=",self.particle_list)
        else:
            print(model, " not implemented yet...")
            
    def run_multi_batch_particle_filter(self, X, y):
        if self.model=="probit_sin_wave":
            x_keys = list(X.keys())
            y_keys = list(Y.keys())
            for n in range(len(x_keys)):
                if n in list(range(0,len(x_keys), int(0.2*len(x_keys)))):
                    print("batch ", x_keys[n])
                for pn in range(self.PART_NUM):
                    if self.sample_method=='importance':
                        self.particle_list[pn].update_particle_importance(X[x_keys[n]], 
                                                                          Y[y_keys[n]], 
                                                                          int(x_keys[n].split(":")[0]))
                    else:
                        print("else in run_multi_batch_particle_filter not implemented")

                self.shuffle_particles(n)
        else:
            print("did nothing...")
        return
                        
    def run_particle_filter(self):
        #single interation of P particles
        self.not_norm_wts=np.ones(self.PART_NUM)
        if self.model=="probit":
            for pn in range(self.PART_NUM):
                #update particle
                self.particle_list[pn].update_particle(self.X, self.Y)
                self.not_norm_wts[pn]=particle_list[pn].evaluate_likelihood(self.X, self.Y)
        if self.model=="probit_sin_wave":
            #print("in run_particle_filter, ", self.model)
            x_keys = list(self.X.keys())
            #print('x_keys=',x_keys)
            y_keys = list(self.Y.keys())
            #print('y_keys=',y_keys)
            #for n in range(self.batch_num):
            for n in range(len(x_keys)):# self.batch_num):
                
                #print("batch ", x_keys[n])
                if n in list(range(0,len(x_keys), int(0.2*len(x_keys)))):#%np.floor(self.batch_num*0.20)==0:
                #if n%np.floor(self.*0.20)==0:
                    print("batch ", x_keys[n])
                
                for pn in range(self.PART_NUM):
                    #print("particle ", pn)
                    if self.sample_method=='importance':
                        #print("got fix this --->>> int(x_keys[n].split(':')[0]):", int(x_keys[n].split(":")[0]))
                        #print('n=',n)
                        #print('x_keys[n]=',x_keys[n])
                        #print('y_keys[n]=',y_keys[n])
                        self.particle_list[pn].update_particle_importance(self.X[x_keys[n]], self.Y[y_keys[n]], int(x_keys[n].split(":")[0]))
                    else:
                        self.particle_list[pn].update_particle(self.X[x_keys[n]], self.Y[y_keys[n]], n)
                    self.not_norm_wts[pn]=self.particle_list[pn].evaluate_likelihood(self.X[x_keys[n]], self.Y[y_keys[n]])
                
                #print('self.not_norm_wts=',self.not_norm_wts)
                #print("n before shuffle:",n)
                self.shuffle_particles()#n)
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