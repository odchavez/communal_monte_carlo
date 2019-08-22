from scipy.stats import norm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import invwishart
import operator

class probit_sin_wave_particle:
    def __init__(self, bo, Bo, idval):#
        self.bo=bo
        self.Bo=Bo
        self.Bo_suf_stat=Bo.copy()
        self.df=2
        self.particle_id=[idval]
        self.particle_id_history=idval
        self.log_lik=-100.0
        self.p=len(bo)
        self.shards=1
        self.this_time=0
        self.sudo_epoch_time = 0
        

    def evaluate_likelihood(self, X, Y):
        x_j_tB=X.dot(self.bo)
        log_PHI=np.log(norm.cdf(x_j_tB))
        log_1_minus_PHI=np.log(1.0-norm.cdf(x_j_tB))
        one_minus_Y=1.0-Y 
        if Y == 0:
            A = 0.0
        else:
            A=np.dot(Y,log_PHI)
        if one_minus_Y == 0:
            B=0
        else:
            B=np.dot(one_minus_Y,log_1_minus_PHI)
        log_lik=(A+B)*self.shards
        return_value = np.max([log_lik, self.log_lik])
        return(return_value)
    
    def update_particle_importance(self, X, Y, j, time_value = None):
        if time_value != None:
            self.time_delta = time_value - self.this_time
            self.this_time  += self.time_delta
            self.sudo_epoch_time += self.time_delta
        else:
            self.this_time=1.0
            self.sudo_epoch_time = 1.0

        self.useful_calcs(X)
        
        mu = np.transpose(self.bo)
        B_mu_proposed = np.random.multivariate_normal(
            mu, 
            self.B_cov,
            1
        ).reshape(self.p,1).flatten()
        
        self.bo = B_mu_proposed
        if time_value != None:
            idx = int(self.this_time)
        else:
            idx = j
        self.bo_list.loc[idx,:]=np.transpose(self.bo)
        
    def update_particle(self,X,Y, j):
        self.useful_calcs(X)
        x_j_tB=X.dot(self.bo)
        n_rows_of_X = X.shape[0]
        for n_batch in range(X.shape[0]):
            #print( Y[n_batch],'==',1)
            if Y[n_batch]==1:
                self.Zi=self.get_truncated_normal(mean=x_j_tB[n_batch], sd=1, low=0, upp=100).rvs(1)
            else:
                self.Zi=self.get_truncated_normal(mean=-x_j_tB[n_batch], sd=1, low=-100, upp=0).rvs(1)
                
        ########################################################################
        #
        #model values need to accommodate knowing the coavariance matrix 
        #
        ########################################################################
                    
        Bo_inv_bo=self.Bo_inv.dot(self.bo)
        B_mu = self.B_cov.dot(Bo_inv_bo + np.transpose(X).dot(self.Zi))
        m=np.transpose(B_mu)[0]
        B = np.random.multivariate_normal(m, self.B_cov,1).reshape(len(m),1)
        self.bo=B
        self.bo_list[j,:]=np.transpose(B)
        self.bo_machine_list[j,:]=np.transpose(B)
        self.Bo_suf_stat+=self.Bo_suf_stat+(B_mu-self.bo).dot(np.transpose(B_mu-self.bo))
        temp_cov = invwishart.rvs(df=self.df, scale=self.Bo_suf_stat)
        self.Bo  = np.linalg.inv(temp_cov)
        self.df += n_rows_of_X#1
        
    def get_particle_id(self):
        return(self.particle_id)

    def get_particle_id_history(self):
        return(self.particle_id_history)
    
    def useful_calcs(self, X):
        self.XtX       =X.transpose().dot(X)
        self.Bo_inv    =np.linalg.inv(self.Bo)
        #self.B_cov     = np.linalg.inv(
        #    self.Bo_inv + ( 
        #        (1.0/.0000007)/self.sudo_epoch_time 
        #    ) * 
        #    np.identity(self.p)*.000000025
        #)
        #print("self.sudo_epoch_time=", self.sudo_epoch_time)
        self.B_cov     = np.linalg.inv(
            self.Bo_inv + (1/self.sudo_epoch_time) * self.XtX #( (1.0/.0000007)/self.sudo_epoch_time ) * np.identity(self.p)*.000000025
        )
    
    def set_N(self, N):
        self.N=N
        return
    
    def set_Zi(self, full_data):
        self.Zi=np.zeros((1,1))
    
    def set_shard_number(self, shards):
        self.shards=shards
            
    def set_bo_list(self, time_values):
        temp_bo_list=np.zeros((len(time_values), len(self.bo)))
        temp_bo_list[temp_bo_list==0]=np.NaN
        self.bo_list = pd.DataFrame(temp_bo_list, index=time_values)
        self.bo_machine_list=np.zeros((self.N, len(self.bo)))
        self.bo_machine_list[self.bo_machine_list==0]=np.NaN

    def get_truncated_normal(self, mean, sd, low, upp):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)    
    
    def copy_particle_values(self, not_this_particle):
        self.bo=not_this_particle.bo.copy()
        self.Bo=not_this_particle.Bo.copy()
        self.bo_list=not_this_particle.bo_list.copy()
        self.particle_id_history=not_this_particle.particle_id[0]
        self.B_cov=np.identity(2)*0.00005
        self.N=not_this_particle.N
        self.Zi=not_this_particle.Zi.copy()
        
    def print_vals(self):
        print(self.bo)
        print(self.Bo)
        print("particle_id=",self.particle_id)
        print("particle_id_history=",self.particle_id_history)
        #print(self.XtX)
        print(self.Bo_inv)
        print(self.B_cov)
        print(self.N)
        print(self.Zi)
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
                ,(np.amax( self.Zi ))
                ,(np.amax( self.bo_list ))
                ,(np.amax( self.Bo_suf_stat ))
                ,(np.amax( self.log_lik ))
                ,(np.amax( self.bo_machine_list))
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
                ,(np.amin( self.Zi ))
                ,(np.amin( self.bo_list ))
                ,(np.amin( self.Bo_suf_stat ))
                ,(np.amin( self.log_lik ))
                ,(np.amin( self.bo_machine_list))
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
        

class probit_particle:
#particles for prbit model class
    def __init__(self, bo, Bo, idval):
        self.bo=bo
        self.Bo=Bo
        self.particle_id=[idval]
        self.particle_id_history=idval
        self.log_lik=-100.0

    def update_particle(self,X,Y):
        self.useful_calcs(X)
        for j in range(self.N):
            x_j_tB=(np.matrix(X.iloc[j])).dot(self.bo)
            if Y[j]==1:
                self.Zi[j]=self.get_truncated_normal(mean=x_j_tB, sd=1, low=0, upp=100).rvs(1)
            else:
                self.Zi[j]=self.get_truncated_normal(mean=-x_j_tB, sd=1, low=-100, upp=0).rvs(1)
        B_mu = self.B_cov.dot(self.Bo_inv_bo+np.transpose(X).dot(self.Zi))
        m=np.transpose(B_mu)[0]
        B = np.random.multivariate_normal(m, 0.000001*self.B_cov,1)
        self.bo=B[0]

    def useful_calcs(self,X):
        self.XtX       =X.transpose().dot(X)
        self.Bo_inv_bo =np.linalg.inv(self.Bo).dot(self.bo)
        self.B_cov     =np.linalg.inv(np.linalg.inv(self.Bo) + self.XtX)
        self.N         =X.shape[0]
        self.Zi        =np.zeros((self.N,1))
        
    def set_N(self, N):
        self.N=N
        
    def print_vals(self):
        print(self.bo)
        print(self.Bo)
        print("particle_id=",self.particle_id)
        print("particle_id_history=",self.particle_id_history)
        print(self.XtX)
        print(self.Bo_inv_bo)
        print(self.B_cov)
        print(self.N)
        print(self.Zi)
        
    def print_max(self):
        L = [ 
            self.bo
            ,self.Bo
            ,self.particle_id
            ,self.particle_id_history
            ,self.XtX
            ,self.Bo_inv_bo
            ,self.B_cov
            ,self.N
            ,self.Zi
        ]   
        output = reduce(operator.concat, L)
        return(output)
    
    def get_truncated_normal(self, mean, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    
    def evaluate_likelihood(self, X, Y):
        x_j_tB=(np.matrix(X)).dot(self.bo)
        log_PHI=np.log(norm.cdf(x_j_tB))[0]
        log_1_minus_PHI=np.log(1.0-norm.cdf(x_j_tB))[0]
        one_minus_Y=1.0-Y        
        A=np.dot(Y,log_PHI)
        B=np.dot(one_minus_Y,log_1_minus_PHI)
        self.log_lik=A+B
        return(self.log_lik)
    
    def copy_particle_values(self, not_this_particle):
        
        ################################################################################################################
        #need to copy the values in a list not the list themselves - may not be able to use the same code for matrix
        ################################################################################################################
        self.bo=not_this_particle.bo.copy()
        self.Bo=not_this_particle.Bo.copy()
        self.particle_id_history=not_this_particle.particle_id[0]
        self.XtX=not_this_particle.XtX.copy()
        self.Bo_inv_bo=not_this_particle.Bo_inv_bo.copy()
        self.B_cov=not_this_particle.B_cov.copy()
        self.N=not_this_particle.N
        self.Zi=not_this_particle.Zi.copy()
        
    def get_particle_id(self):
        return(self.particle_id)

    def get_particle_id_history(self):
        return(self.particle_id_history)