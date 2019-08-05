import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import math
from matplotlib import pyplot as plt
import random
import os

class simulated_data:
    def __init__(self, params, model, show=True):
        self.model=model
        if model=="probit":
            params={'N': 10000, 'b': [-2,-1,0,1,2], 'B':np.identity(5)}
            self.p = len(params['b'])
            self.N = params['N']
            self.X = np.random.uniform(-1,1,self.p*self.N).reshape((self.N,self.p))
            self.Y = np.zeros(self.N)
            self.b=params['b']
            self.B=params['B']
        
            for i in range(self.N):
                inner=self.X[i].dot(self.b)
                samp=np.random.normal(loc=inner, scale=1.0, size=1)[0]
                if samp>=0:
                    self.Y[i]=1
            self.output={'X':self.X, 'Y':self.Y, 'N': self.N, 'b': self.b, 'B':self.B}
                
        if model== "probit_sin_wave":
            
            omega_shift=params['omega_shift']
            temp_params={'p':len(omega_shift), 'b': omega_shift, 'B':np.identity(len(omega_shift))}

            params.update(temp_params)
            self.p=params['p']
            self.N=params['N']
            self.N_batch = params['N_batch']
            self.shards=params['shards']
            self.epoch_at=params['epoch_at']
            self.epoch_number=len(self.epoch_at)+1
            self.X={}
            self.Y={}
            self.data_keys=list()
            f=1.5
            x = np.arange(self.N) # the points on the x axis for plotting
            self.b=np.zeros((self.N,self.p))
            self.b_oos=np.zeros((1,self.p))
            
            output = {}
            for m in range(self.shards):
                output["shard_"+str(m)]     ={}
                output["shard_"+str(m)]['X']={}
                output["shard_"+str(m)]['Y']={}
                output["shard_"+str(m)]['data_keys']=list()
            
            epoch_output = {}
            for ep in range(self.epoch_number):
                epoch_output["epoch"+str(ep)]={}
                for m in range(self.shards):
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]     ={}
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X']={}
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['Y']={}
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['data_keys']=list()
            
            for os in range(self.p):
                self.b[:,os] = [ 1.0*np.sin(np.pi*f * (i/(self.N*1.0)) + omega_shift[os]) for i in x]
                #% matplotlib inline
                # showing the exact location of the smaples
                if show:
                    plt.stem(x,self.b[:,os], 'r', )
                    plt.plot(x,self.b[:,os])
                    plt.yticks(np.linspace(start=-2, stop=2, num=9))
                    plt.grid(True)
                    plt.show()
            
            for os in range(self.p):
                self.b_oos[:,os] =  [ 1.0*np.sin(np.pi*f * (i/(self.N*1.0)) + omega_shift[os]) for i in [max(x)+1]]
                
            data_index=0
            epoch_counter=0
            for i in range(self.N):
                key=str(i)+":"+str(data_index)
                all_key=str(i)+":"+str(i)
                self.data_keys.append(all_key)
                s="shard_"+str(i%self.shards)

                temp_X = np.random.uniform(-1,1,self.p*self.N_batch).reshape((self.N_batch,self.p))
                self.X[all_key]     = temp_X
                self.Y[all_key]     = np.zeros(self.N_batch)
                
                output[s]['X'][key] = temp_X
                output[s]['Y'][key] = np.zeros(self.N_batch)
                output[s]['data_keys'].append(key)
                
                inner=temp_X.dot(self.b[i,])
                samp=np.random.normal(loc=inner, scale=1.0)
                for j in range(self.N_batch):
                    if samp[j]>=0:
                        output[s]['Y'][key][j]=self.Y[all_key][j]=1
                
                if i not in self.epoch_at:
                    epoch_output["epoch"+str(epoch_counter)][s]['X'][key]=temp_X
                    epoch_output["epoch"+str(epoch_counter)][s]['Y'][key]=self.Y[all_key] #np.zeros(self.N_batch)
                    epoch_output["epoch"+str(epoch_counter)][s]['data_keys'].append(key)
                else:
                    for m in range(self.shards):
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['X'][all_key] = self.X[all_key]
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['Y'][all_key] = self.Y[all_key]
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['data_keys'].append(all_key)
                    epoch_counter+=1
                
                if i%self.shards == self.shards-1:
                    data_index+=1
                    
            tttemp=np.max([100,self.N_batch])
            self.X_oos=np.random.uniform(-1,1,self.p*tttemp).reshape((tttemp,self.p))
            
            sig=np.max(np.var(self.b[0:(self.N-1),:]-self.b[1:,:], axis=0))
            self.B=30*sig*params['B']
            
            for m in range(self.shards):
                output["shard_"+str(m)]['N'] = self.N
                output["shard_"+str(m)]['b'] = self.b
                output["shard_"+str(m)]['B'] = self.B*self.shards
                output["shard_"+str(m)]['p'] = self.p
                output["shard_"+str(m)]['model'] = self.model
                output["shard_"+str(m)]['shards']=self.shards
                
                L =list(output["shard_"+str(m)]['X'].keys())
                L1=L[len(L)-1].split(":")
                max_val=int(L1[len(L1)-1])
                output["shard_"+str(m)]['batch_number']=max_val+1
            
            for ep in range(self.epoch_number):
                epoch_output["epoch"+str(ep)]['parallel_shards'] = self.shards
                epoch_output["epoch"+str(ep)]['b']               = self.b
                epoch_output["epoch"+str(ep)]['B'] = self.B*self.shards

                for m in range(self.shards):
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['N'] = self.N
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['b'] = self.b
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['B'] = self.B*self.shards
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['p'] = self.p
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['model'] = self.model
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['shards']=self.shards

            
            self.output=output
            self.output['epoch_data']=epoch_output
            self.output['X']=self.X
            self.output['Y']=self.Y
            self.output['N']=self.N
            self.output['b']=self.b
            self.output['B']=self.B
            self.output['p']=self.p
            self.output['b_oos']=self.b_oos
            self.output['X_oos']=self.X_oos
            self.output['batch_number']=self.N_batch
            self.output['model']=self.model
            self.output['shards']=1
            self.output['parallel_shards']=self.shards
            self.output['data_keys']=self.data_keys
            
    def get_data(self):
        return(self.output)
     
def temp_make_data_function(params, model, show):
    test = simulated_data(params, model, show).get_data()
    return test, params


class  simulated_data2:
    
    def __init__(self, n_per_file, N_total = 10000000, n_per_tic = 10, pred_number = 100):
        
        random.seed(30)
        self.time_tics = np.array(range(int(N_total/n_per_tic)))
        self.row = len(self.time_tics) * n_per_tic
        self.n_per_file = n_per_file/n_per_tic
        self.pred_number =pred_number
        self.N_total = N_total
        self.n_per_tic = n_per_tic
        self.n_per_epoch = n_per_file
        
        self.output_folder_name = (
                    "synth_data/"
                    "Xy_N=" + str(self.N_total) + 
                    "_Epoch_N=" + str(self.n_per_epoch) + 
                    "_Nt=" + str(self.n_per_tic) + 
                    "_p=" + str(self.pred_number) +
                    "/"
                )

        if not os.path.exists(self.output_folder_name):
            os.makedirs(self.output_folder_name)
        
    def make_linear_trajectory(self, f_time_tic, y_2 = 1.0, y_1 = -1.0,):
        
        T_max = max(f_time_tic)
        T_min = min(f_time_tic)
        m = (y_2 - y_1)/(T_max - T_min)
        T = np.array(range(int(T_min), int(T_max+1.0)))
        output = m*(T-y_1) + y_1
        return output

    def make_logrithmic_trajectory(self, f_time_tic,):
        
        T_max = max(f_time_tic)
        T_min = min(f_time_tic)
        T = np.array(range(int(T_min), int(T_max+1.0)))
        output = np.log(T+1)
        return output
    
    def make_sin_wave_trajectory(self, f_time_tic, y_2 = 1.0, y_1 = -1.0,):
        
        T_max = max(f_time_tic)
        T_min = min(f_time_tic)
        T = np.array(range(int(T_min), int(T_max+1.0)))
        output = np.sin((T-T_min)*2*math.pi/T_max)
        return output

    def generate_Betas(self, path="synth_data/"):
        print("generating regression coefficients...")
        self.Beta_vals_base = {
            'B_0': self.make_linear_trajectory(f_time_tic = self.time_tics, y_2 = 1.0, y_1 = -1.0),
            'B_1': self.make_linear_trajectory(f_time_tic = self.time_tics, y_2 = -2.0, y_1 = 2.0),
            'B_2': self.make_logrithmic_trajectory(f_time_tic = self.time_tics),
            'B_3': self.make_sin_wave_trajectory(f_time_tic = self.time_tics)
        }
        beta_cnames = list(range(self.pred_number))
        Beta_vals={}
        for pn in range(self.pred_number):
            Beta_vals['B_'+str(pn)] = self.Beta_vals_base['B_'+str(pn%len(self.Beta_vals_base))]
            beta_cnames[pn] = 'B_'+str(pn)
        #Beta_vals.head()
        self.Beta_vals_df = pd.DataFrame(Beta_vals)[beta_cnames]
        
        if path == '':
            self.Beta_vals_df.to_csv(self.output_folder_name + "Beta_t.csv" )
        else:
            self.Beta_vals_df.to_csv(self.output_folder_name + "Beta_t.csv" )
        print("regression coefficients generation complete...")

    def generate_data(self, path=None):
        
        if ~(path == None):
            path = self.output_folder_name
        
        print("generating data...")
        print("writing data to " + path)
        X_i_all = pd.DataFrame()
        vcnames=list(range(self.pred_number))
        #pnames={}
        for i in range(self.pred_number):
            vn="v_"+str(i)
            #pnames["v_"+str(i)] = vn
            vcnames[i]=vn
        file_num = 0
        for tt in tqdm(range(len(self.time_tics))):
            
            X_i =  pd.DataFrame(
                np.random.multivariate_normal(
                    np.zeros(self.pred_number),
                    np.identity(self.pred_number)*.01,
                    self.n_per_tic
                )
            )
            X_i.columns=vcnames
        
            
            #X_i.columns=pnames
            
            Beta_t = self.Beta_vals_df.iloc[tt].T#Beta_vals[tt,:]
                
            X_B_t = X_i.dot(np.array(Beta_t))
            event_prob = norm.cdf(X_B_t)
            #print(event_prob)
            X_i['y'] = np.random.binomial(1,event_prob)
            X_i['time'] = tt
            
            if X_i_all.shape[0]==0:
                X_i_all = X_i
            else:
                X_i_all = pd.concat([X_i_all, X_i], axis=0)
                
            if tt % self.n_per_file == self.n_per_file-1:
                file_name = (
                    "fn=" +str(file_num) +
                    ".csv"
                )
                
                X_i_all.to_csv(path + file_name )
                
                file_num+=1
                X_i_all = pd.DataFrame()
            
        print("data generation complete...")
