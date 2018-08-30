#particle classes
#from scipy.stats import truncnorm
from scipy.stats import norm
#from scipy.stats import truncnorm
#from scipy.special import logsumexp
import numpy as np
#import pandas as pd
#import csv
from matplotlib import pyplot as plt

class probit_sin_wave_particle:
    def __init__(self, bo, Bo, idval):#
        #print("bo=",bo)
        self.bo=bo
        self.Bo=Bo
        self.particle_id=[idval]
        self.particle_id_history=idval
        self.log_lik=-99999999999.0
        self.p=len(bo)

    def evaluate_likelihood(self, X, Y):
        #print("in evaluate likelihood ")
        #print("X=",X)
        #print('self.bo=',self.bo)
        x_j_tB=X.dot(self.bo)#(np.transpose(self.bo))
        #print('x_j_tB=',x_j_tB)
        log_PHI=np.log(norm.cdf(x_j_tB))#[0]
        #print("log_PHI=",log_PHI)
        log_1_minus_PHI=np.log(1.0-norm.cdf(x_j_tB))#[0]
        #print("log_1_minus_PHI=",log_1_minus_PHI)
        one_minus_Y=1.0-Y 
        #print("one_minus_Y=",one_minus_Y)
        A=np.dot(Y,log_PHI)
        #print("np.dot(Y,log_PHI)=",np.dot(Y,log_PHI))
        B=np.dot(one_minus_Y,log_1_minus_PHI)
        #print("np.dot(one_minus_Y,log_1_minus_PHI)=",np.dot(one_minus_Y,log_1_minus_PHI))
        log_lik=A+B
        #print("log_lik=",log_lik)
        return(log_lik)
    
    def update_particle_importance(self, X, Y, j):
        #print("check this:", np.array(self.bo))
        self.useful_calcs(X)
        B_mu_proposed=np.random.multivariate_normal(np.transpose(self.bo), self.B_cov,1).reshape(self.p,1).flatten()
        self.bo=B_mu_proposed
        self.bo_list[j,:]=np.transpose(B_mu_proposed).copy()
    
    def update_particle(self,X,Y, j):
        self.useful_calcs(X)
        #for j in range(self.N):
        #print("X=",X)
        #print("np.matrix(X)=",np.matrix(X))
        
        #print("np.matrix(X[j])=",np.matrix(X[j]))
        #print("self.bo=", self.bo)
        #print("*******************")
        #print("matmul=", np.matmul(X, self.bo))
        x_j_tB=X.dot(self.bo)#(np.matrix(X.iloc[j])).dot(self.bo)
        #print("x_j_tB=",x_j_tB)
        for n_batch in range(X.shape[0]):
            if Y[n_batch]==1:
                self.Zi[str(j)][n_batch]=self.get_truncated_normal(mean=x_j_tB[n_batch], sd=1, low=0, upp=100).rvs(1)
            else:
                self.Zi[str(j)][n_batch]=self.get_truncated_normal(mean=-x_j_tB[n_batch], sd=1, low=-100, upp=0).rvs(1)
            
        ########################################################################
        #
        #model values need to accommodate knowing the coavariance matrix 
        #
        ########################################################################
                    
        Bo_inv_bo=self.Bo_inv.dot(self.bo)
        #print("Bo_inv_bo=",Bo_inv_bo)
        #print("self.B_cov=",self.B_cov)
        #print("X.reshape(len(X,1)=",X.reshape(len(X),1))
        #print("np.transpose(X)=",np.transpose(X))
        #print("Zi[str(j)])=", self.Zi[str(j)])
        #print(self.Zi.keys())
        #print("Zi=", self.Zi)
        
        
        B_mu = self.B_cov.dot(Bo_inv_bo + np.transpose(X).dot(self.Zi[str(j)]))
        #print("B_mu=", B_mu)
        #print("np.transpose(B_mu)[0]=",np.transpose(B_mu)[0])
        m=np.transpose(B_mu)[0]
        #print("m=",m)
        #print("self.B_cov=",self.B_cov)
        B = np.random.multivariate_normal(m, self.B_cov,1).reshape(len(m),1)
        #print("the new self.bo", B)
        self.bo=B.copy()
        self.bo_list[j,:]=np.transpose(B).copy()#[0]
        #self.bo_list[j,1]=B[1]
        #self.bo_list[j,2]=B[2]

    def get_particle_id(self):
        return(self.particle_id)

    def get_particle_id_history(self):
        return(self.particle_id_history)
    
    def useful_calcs(self, X):
        self.XtX       =X.transpose().dot(X)
        #self.Bo_inv_bo =np.linalg.inv(self.Bo).dot(self.bo)
        #self.B_cov     =np.linalg.inv(np.linalg.inv(self.Bo) + self.XtX)
        #print("just called useful_calcs shape of X=", X.shape)
        self.Bo_inv    =np.linalg.inv(self.Bo)
        self.B_cov     =np.linalg.inv(self.Bo_inv + self.XtX)#self.N_batch*self.Bo)
        #print('self.B_cov=', self.B_cov)
        return
    
    def set_N(self, N):
        self.N=N
        return
    
    def set_Zi(self, full_data):
        keys=full_data.keys()
        N=len(keys)
        self.Zi={}
        for key in keys:
            self.Zi[key]=np.zeros((full_data[key].shape[0],1))
            
    def set_bo_list(self):
        self.bo_list=np.zeros((self.N, len(self.bo)))


    def get_truncated_normal(self, mean, sd, low, upp):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)    
    
    def copy_particle_values(self, not_this_particle):
        self.bo=not_this_particle.bo.copy()
        self.Bo=not_this_particle.Bo.copy()
        self.bo_list=not_this_particle.bo_list.copy()
        self.particle_id_history=not_this_particle.particle_id[0]
        #self.XtX=not_this_particle.XtX
        #self.Bo_inv_bo=not_this_particle.Bo_inv_bo
        #self.B_cov=not_this_particle.B_cov.copy()
        self.N=not_this_particle.N
        self.Zi=not_this_particle.Zi.copy()
    
    #def copy_bo_list(self):
    
    def print_vals(self):
        print(self.bo)
        print(self.Bo)
        print("particle_id=",self.particle_id)
        print("particle_id_history=",self.particle_id_history)
        #print(self.XtX)
        #print(self.Bo_inv_bo)
        print(self.B_cov)
        print(self.N)
        print(self.Zi)
        print("self.bo_list=",self.bo_list)
        
    def plot_parameters(self):
        #plot parameters
        print("self.N=",self.N)
        print("np.arange(self.N)=",np.arange(self.N))
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
        self.log_lik=-99999999999.0

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