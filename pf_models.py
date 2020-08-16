from scipy.stats import norm
import numpy as np
import pandas as pd
import math as m
from matplotlib import pyplot as plt
from scipy.stats import invwishart
import operator

class probit_sin_wave_particle:
    def __init__(self, bo, Bo, Tau_inv, idval):#
        self.bo=bo
        #self.Bo=Bo
        self.Tau_inv=Tau_inv
        self.Bo = Bo
        self.std_rescale = Bo/(m.sqrt(Bo*Bo + Tau_inv*Tau_inv))
        self.df=2
        self.particle_id=[idval]
        self.particle_id_history=idval
        self.log_lik=-100.0
        self.p=len(bo)
        self.shards=1
        self.this_time=0
        self.sudo_epoch_time = 0


    def evaluate_likelihood(self, X, Y):
        x_j_tB          = X.dot(self.bo)
        p_of_x_i = 1.0/(1.0+np.exp(-x_j_tB))
        A = np.dot(Y, np.log(p_of_x_i))
        B = np.dot(1.0-Y, np.log(1-p_of_x_i))
        log_lik      = (A+B)*self.shards
        return_value = np.max([log_lik, self.log_lik])
        return(return_value)

    def update_particle_importance(self,XtX, X, Y, time_value = None):
        if time_value != None:
            self.time_delta = time_value - self.this_time
            self.this_time  += self.time_delta
            self.sudo_epoch_time += self.time_delta
        else:
            self.this_time=1.0
            self.sudo_epoch_time = 1.0

        #self.useful_calcs(X, XtX)

        self.bo = np.random.multivariate_normal(
            np.transpose(self.bo),
            self.Tau_inv*np.identity(len(self.bo)),
            1
        ).astype(float).reshape(self.p,1).flatten()*self.std_rescale

        if time_value != None:
            idx = int(self.this_time)
        else:
            idx = j
        self.bo_list.loc[idx,:]=np.transpose(self.bo)

    def get_particle_id(self):
        return(self.particle_id)

    def get_particle_id_history(self):
        return(self.particle_id_history)

    def useful_calcs(self, X, XtX):
        self.XtX       = XtX
        self.Bo_inv    = np.linalg.inv(self.Bo)
        self.B_cov     = np.linalg.inv(
            self.Bo_inv + (1.0/self.sudo_epoch_time) * self.XtX
        )

    def set_N(self, N):
        self.N=N
        return

    #def set_Zi(self, full_data):
    #    self.Zi=np.zeros((1,1))

    def set_shard_number(self, shards):
        self.shards=shards

    def set_bo_list(self, time_values):
        temp_bo_list=np.zeros((len(time_values), len(self.bo)))
        temp_bo_list[temp_bo_list==0]=np.NaN
        self.bo_list = pd.DataFrame(temp_bo_list, index=time_values)

    def get_truncated_normal(self, mean, sd, low, upp):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def copy_particle_values(self, not_this_particle):
        self.bo=not_this_particle.bo.copy()
        self.particle_id_history=not_this_particle.particle_id_history #[0]
        self.N=not_this_particle.N

    def print_vals(self):
        print(self.bo)
        print(self.Bo)
        print("particle_id=",self.particle_id)
        print("particle_id_history=",self.particle_id_history)
        print(self.Bo_inv)
        print(self.B_cov)
        print(self.N)
        print("self.bo_list=",self.bo_list)

    def print_max(self):

        L = [
                (np.amax( self.bo ))
                ,(np.amax( self.Bo ))
                ,(np.amax( self.particle_id ))
                ,(np.amax( self.particle_id_history ))
                ,(np.amax( self.XtX ))
                ,(np.amax( self.B_cov ))
                ,(np.amax( self.Bo_inv))
                ,(np.amax( self.N ))
                ,(np.amax( self.bo_list ))
                ,(np.amax( self.Bo_suf_stat ))
                ,(np.amax( self.log_lik ))
                ,(np.amin(self.log_lik))
            ]
        print(max(L))

    def print_min(self):

        L = [
                 (np.amin( self.bo ))
                ,(np.amin( self.Bo ))
                ,(np.amin( self.particle_id ))
                ,(np.amin( self.particle_id_history ))
                ,(np.amin( self.XtX ))
                ,(np.amin( self.B_cov ))
                ,(np.amin( self.Bo_inv))
                ,(np.amin( self.N ))
                ,(np.amin( self.bo_list ))
                ,(np.amin( self.Bo_suf_stat ))
                ,(np.amin( self.log_lik ))
                ,(np.amin(self.log_lik))
            ]
        print(min(L))

    def plot_parameters(self):
        x = np.arange(self.N) # the points on the x axis for plotting
        for os in range(self.p):
            #% matplotlib inline
            plt.stem(x,self.bo_list[:,os], 'r', )
            plt.plot(x,self.bo_list[:,os])
            plt.yticks(np.linspace(start=-2, stop=2, num=9))
            plt.grid(True)
            plt.show()
        return
