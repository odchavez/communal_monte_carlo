#particle classes
#from scipy.stats import truncnorm
from scipy.stats import norm
#from scipy.stats import truncnorm
#from scipy.special import logsumexp
import numpy as np
#import pandas as pd
#import csv
from matplotlib import pyplot as plt
from scipy.stats import invwishart#, invgamma
import operator

class probit_sin_wave_particle:
    def __init__(self, bo, Bo, idval):#
        #print("bo=",bo)
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
        #print("in evaluate likelihood ")
        #print("X.shape=",X.shape)
        #print('self.bo.shape=',self.bo.shape)
        x_j_tB=X.dot(self.bo)#(np.transpose(self.bo))
        #print("X=", X)
        #print("self.bo=",self.bo)
        #print('x_j_tB=',x_j_tB)
        log_PHI=np.log(norm.cdf(x_j_tB))#[0]
        #print("log_PHI=",log_PHI)
        log_1_minus_PHI=np.log(1.0-norm.cdf(x_j_tB))#[0]
        #print("log_1_minus_PHI=",log_1_minus_PHI)
        one_minus_Y=1.0-Y 
        #print("one_minus_Y=",one_minus_Y)
        if Y == 0:
            A = 0.0
        else:
            A=np.dot(Y,log_PHI)
        #print("np.dot(Y,log_PHI)=",np.dot(Y,log_PHI))
        if one_minus_Y == 0:
            B=0
        else:
            B=np.dot(one_minus_Y,log_1_minus_PHI)
        #print("np.dot(one_minus_Y,log_1_minus_PHI)=",np.dot(one_minus_Y,log_1_minus_PHI))
        #print("A=", A)
        #print("B=", B)
        log_lik=(A+B)*self.shards
        #print("log_lik=",log_lik)
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
        #print("this time = ", self.this_time)
        #print("just entered update_particle_importance in pf_models.py")
        #print("X looks like:")
        #print(X)
        #print("select index looks like:")
        #print(~(X==0))
        self.useful_calcs(X)
        #print("np.transpose(self.bo).shape: ", np.transpose(self.bo).shape)
        #print("self.B_cov.shape: ", self.B_cov.shape)
        
        mu = np.transpose(self.bo) #np.zeros(self.p)
        B_mu_proposed = np.random.multivariate_normal(
            mu, 
            self.B_cov,
            1
        ).reshape(self.p,1).flatten()
        
        #B_mu_proposed=np.random.multivariate_normal(np.transpose(self.bo), self.B_cov,1).reshape(self.p,1).flatten()
        #B_mu_proposed = B_mu_proposed / np.sqrt(np.diag(self.B_cov))
        #print("np.transpose(self.bo)=", np.transpose(self.bo))
        #B_mu_proposed=np.random.multivariate_normal(np.zeros(self.p), self.B_cov,1).reshape(self.p,1).flatten()
        #print("B_mu_proposed:", B_mu_proposed)
        #select_index = ~(X==0)
        #print("select_index=", select_index)
        #if len(select_index) == 0:
        #    self.bo = B_mu_proposed
        #else:
        #    print("type(B_mu_proposed)=", type(B_mu_proposed))
        #    print("B_mu_proposed[select_index]=", B_mu_proposed[:,select_index])
        #    print("self.bo[select_index]=", self.bo[select_index])
        #    self.bo[select_index]=B_mu_proposed[select_index]
        self.bo = B_mu_proposed
        if time_value != None:
            idx = int(self.this_time)
        else:
            idx = j
        #print("idx = ", idx)
        self.bo_list[idx,:]=np.transpose(self.bo)#np.transpose(B_mu_proposed)#.copy()
        #self.bo_machine_list[j,:]=np.transpose(self.bo)#np.transpose(B_mu_proposed)#.copy()
        
    def update_particle(self,X,Y, j):
        self.useful_calcs(X)
        x_j_tB=X.dot(self.bo)#(np.matrix(X.iloc[j])).dot(self.bo)
        #print("***********************in update_particle***********************")
        #print("X.shape=",X.shape)
        for n_batch in range(X.shape[0]):
            print( Y[n_batch],'==',1)
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
        #print("temp_cov=", temp_cov)
        self.Bo= np.linalg.inv(temp_cov)
        self.df+=1
        
    def get_particle_id(self):
        return(self.particle_id)

    def get_particle_id_history(self):
        return(self.particle_id_history)
    
    def useful_calcs(self, X):
        self.XtX       =X.transpose().dot(X)
        #print("self.XtX.shape= " , self.XtX.shape )
        self.Bo_inv    =np.linalg.inv(self.Bo)
        #print("self.Bo_inv.shape=", self.Bo_inv.shape)
        self.B_cov     = np.linalg.inv(
            self.Bo_inv + ( 
                (1.0/.0000007)/self.sudo_epoch_time 
            ) * 
            np.identity(self.p)*.000000025
        )# np.linalg.inv(self.Bo_inv + self.XtX)
        #print("self.sudo_epoch_time = ", self.sudo_epoch_time)
        #print(self.B_cov)
        #self.B_cov     = Sigma_prior + t * Epsilon_Cov
        #print("self.B_cov.shape=", self.B_cov.shape)
    
    def set_N(self, N):
        self.N=N
        return
    
    def set_Zi(self, full_data):
        self.Zi=np.zeros((1,1))
    
    def set_shard_number(self, shards):
        self.shards=shards
            
    def set_bo_list(self, row_num):
        #print("setting bo_list")
        self.bo_list=np.zeros((row_num, len(self.bo)))
        self.bo_list[self.bo_list==0]=np.NaN
        
        self.bo_machine_list=np.zeros((self.N, len(self.bo)))
        self.bo_machine_list[self.bo_machine_list==0]=np.NaN

    def get_truncated_normal(self, mean, sd, low, upp):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)    
    
    def copy_particle_values(self, not_this_particle):
        self.bo=not_this_particle.bo.copy()
        self.Bo=not_this_particle.Bo.copy()
        self.bo_list=not_this_particle.bo_list.copy()
        self.particle_id_history=not_this_particle.particle_id[0]
        #self.XtX=not_this_particle.XtX
        #self.Bo_inv_bo=not_this_particle.Bo_inv_bo
        self.B_cov=np.identity(2)*0.00005#not_this_particle.B_cov.copy()
        self.N=not_this_particle.N
        self.Zi=not_this_particle.Zi.copy()
    
    #def copy_bo_list(self):
    
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
        #plot parameters
        #print("self.N=",self.N)
        #print("np.arange(self.N)=",np.arange(self.N))
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