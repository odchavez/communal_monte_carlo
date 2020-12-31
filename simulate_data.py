import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import math
from matplotlib import pyplot as plt
import os


class simulated_data2:

    def __init__(self, n_per_file, N_total = 10000000, n_per_tic = 10, pred_number = 100, seed = 0, GP=0, GB_version=0):

        self.time_tics = np.array(range(int(N_total/n_per_tic)))
        self.row = len(self.time_tics) * n_per_tic
        self.n_per_file = n_per_file/n_per_tic
        self.pred_number =pred_number
        self.N_total = N_total
        self.n_per_tic = n_per_tic
        self.n_per_epoch = n_per_file
        self.seed = seed
        self.GP = GP
        self.GB_version = GB_version
        
        if GP == 0:
            self.output_folder_name = (
                "synth_data/"
                "Xy_N=" + str(self.N_total) +
                "_Epoch_N=" + str(self.n_per_epoch) +
                "_Nt=" + str(self.n_per_tic) +
                "_p=" + str(self.pred_number) +
                "/"
            )
        if GP == 1:
            self.output_folder_name = (
                "synth_data/"
                "Xy_N=" + str(self.N_total) +
                "_Epoch_N=" + str(self.n_per_epoch) +
                "_Nt=" + str(self.n_per_tic) +
                "_p=" + str(self.pred_number) +
                "/GP_version=" + str(self.GB_version) +
                "/"
            )
        
            self.Beta_file_name = (
                "Beta_t" + 
                "_Xy_N=" + str(self.N_total) +
                "_Epoch_N=" + str(self.n_per_epoch) +
                "_Nt=" + str(self.n_per_tic) +
                "_p=" + str(self.pred_number) +
                "_GP_version=" + str(self.GB_version) +
                ".csv"
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
        output = np.log((T+300000)/300000)
        return output

    def make_sin_wave_trajectory(self, f_time_tic, sin_or_cos='cos',):

        T_max = max(f_time_tic)
        T_min = min(f_time_tic)
        T = np.array(range(int(T_min), int(T_max+1.0)))
        if sin_or_cos == 'cos':
            output = np.cos((T-T_min)*2*math.pi/T_max)
        else:
            output = np.sin((T-T_min)*2*math.pi/T_max)
        return output

    def make_K(self, x, h, lam):
        """
        Make covariance matrix from covariance kernel
        """
        # for a data array of length x, make a covariance matrix x*x:
        K = np.zeros((len(x),len(x)))
        for i in range(0,len(x)):
            for j in range(0,len(x)):
                # calculate value of K for each separation:
                K[i,j] = self.cov_kernel(x[i],x[j],h,lam)
    
        return K

    def cov_kernel(self, x1,x2,h,lam):
        """
        Squared-Exponential covariance kernel
        """
        k12 = h**2*np.exp(-1.*(x1 - x2)**2/lam**2)
        return k12
    
    def make_GP_trajectory(self, f_time_tic):
        #t = np.arange(0, f_time_tic)
        h=1.0
        lam=len(f_time_tic)/2 #100
        K = self.make_K(f_time_tic,h,lam)
        y = np.random.multivariate_normal(np.zeros(len(f_time_tic)),K)
        return y-y[0]

    def generate_Betas(self, sin_or_cos, intercepts_1, intercepts_2, intercepts_3):
        print("generating regression coefficients...")
        if self.GP==0:
            self.Beta_vals_base = {
                'B_0': self.make_sin_wave_trajectory(f_time_tic = self.time_tics, sin_or_cos=sin_or_cos),
                'B_1': self.make_linear_trajectory(f_time_tic = self.time_tics, y_2 = 2.0, y_1 =  intercepts_1),
                'B_2': self.make_linear_trajectory(f_time_tic = self.time_tics, y_2 = -3.0, y_1 = intercepts_2),
                'B_3': self.make_linear_trajectory(f_time_tic = self.time_tics, y_2 = -3.0, y_1 = intercepts_3),
            }
            beta_cnames = list(range(self.pred_number))
            Beta_vals={}
            for pn in range(self.pred_number):
                Beta_vals['B_'+str(pn)] = self.Beta_vals_base['B_'+str(pn%len(self.Beta_vals_base))]
                beta_cnames[pn] = 'B_'+str(pn)
            #Beta_vals.head()
            self.Beta_vals_df = pd.DataFrame(Beta_vals)[beta_cnames]
    
            self.Beta_vals_df.to_csv(self.output_folder_name + "Beta_t.csv" )
        if self.GP==1:
            beta_cnames = list(range(self.pred_number))
            Beta_vals={}
            for pn in tqdm(range(self.pred_number)):
                Beta_vals['B_'+str(pn)] = self.make_GP_trajectory(f_time_tic= self.time_tics)
                beta_cnames[pn] = 'B_'+str(pn)
            self.Beta_vals_df = pd.DataFrame(Beta_vals)[beta_cnames]
            
            self.Beta_vals_df.to_csv(self.output_folder_name + self.Beta_file_name )

        print("regression coefficients generation complete...")
        print("estimating Tau_inv_std parameter...")
        self.Tau_inv_std = np.max(self.Beta_vals_df.diff().std())
        self.Bo_std = self.Beta_vals_df.values.std()
        print("Tau_inv_std = ", self.Tau_inv_std)
        print("Bo_std = ", self.Bo_std)
        

    def generate_data(self):

        print("generating data...")
        print("writing data to " + self.output_folder_name)
        X_i_all = pd.DataFrame()
        vcnames=list(range(self.pred_number))
        #pnames={}
        for i in range(self.pred_number):
            vn="v_"+str(i)
            #pnames["v_"+str(i)] = vn
            vcnames[i]=vn
        file_num = 0
        for tt in tqdm(range(len(self.time_tics))):
            np.random.seed(tt+self.seed)

            X_i =  pd.DataFrame(
                np.random.uniform(
                    low = -1,
                    high = 1,
                    size = (self.n_per_tic, self.pred_number)
                )
            )
            X_i.columns=vcnames


            #X_i.columns=pnames

            Beta_t = self.Beta_vals_df.iloc[tt].T

            X_B_t = X_i.dot(np.array(Beta_t))
            event_prob = 1.0/(1.0+np.exp(-X_B_t))
            Y_vals = np.random.binomial(n=1, p=event_prob, size=None)

            X_i['y'] = Y_vals
            X_i['Tau_inv_std'] = self.Tau_inv_std
            X_i['Bo_std'] = self.Bo_std
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

                X_i_all.to_csv(self.output_folder_name + file_name )

                file_num+=1
                X_i_all = pd.DataFrame()

        print("data generation complete...")
        
        
class simulated_data_Dynamic_Gaussian_Mixture:
    """
    Generate a 2-D Dynamic Gausian Mixture with a fixed covariance I*sig^2.  means and sigma are 
    stored in the output file.  Generate data with the following code:
    
    DGM_obj = simulated_data_Dynamic_Gaussian_Mixture (
    n_per_file=1000, N_total = 10000, n_per_tic = 100, dimension = 2, mix_comp_num=5, seed = 0, GP_version=0)
    
    """
    def __init__(
        self, n_per_file, N_total = 10000, n_per_tic = 100, dimension = 2, mix_comp_num=5, 
        seed = 0, GP_version=0):

        self.time_tics = np.array(range(int(N_total/n_per_tic)))
        self.row = len(self.time_tics) * n_per_tic
        self.n_per_file = n_per_file/n_per_tic
        self.dimension = dimension 
        self.mix_comp_num = mix_comp_num
        self.N_total = N_total
        self.n_per_tic = n_per_tic
        self.n_per_epoch = n_per_file
        self.seed = seed
        self.GP_version = GP_version
        
        self.output_folder_name = (
            "synth_data/DGM/"
            "Xy_N=" + str(self.N_total) +
            "_Epoch_N=" + str(self.n_per_epoch) +
            "_Nt=" + str(self.n_per_tic) +
            "_p=" + str(self.dimension) +
            "_mix_comp_num=" + str(self.mix_comp_num) +
            "/GP_version=" + str(self.GP_version) +
            "/"
        )
                
        if not os.path.exists(self.output_folder_name):
            os.makedirs(self.output_folder_name)
            
        self.make_dynamic_GM_path()
        self.generate_DGM_data()
            
    def make_K(self, x, h, lam):
        """
        Make covariance matrix from covariance kernel
        """
        # for a data array of length x, make a covariance matrix x*x:
        K = np.zeros((len(x),len(x)))
        for i in range(0,len(x)):
            for j in range(0,len(x)):
                # calculate value of K for each separation:
                K[i,j] = self.cov_kernel(x[i],x[j],h,lam)
    
        return K

    def cov_kernel(self, x1,x2,h,lam):
        """
        Squared-Exponential covariance kernel
        """
        k12 = h**2*np.exp(-1.*(x1 - x2)**2/lam**2)
        return k12
    
    def make_GP_trajectory(self, f_time_tic):
        #t = np.arange(0, f_time_tic)
        h=1.0
        lam=len(f_time_tic)/2 #100
        K = self.make_K(f_time_tic,h,lam)
        y = np.random.multivariate_normal(np.zeros(len(f_time_tic)),K)
        return y-y[0]
    
    def make_dynamic_GM_path(self):
        #initialize path containers
        x1_vec = np.zeros((len(self.time_tics), self.mix_comp_num))
        x2_vec = np.zeros((len(self.time_tics), self.mix_comp_num))
        
        #add starting points
        starting_points = np.random.uniform(low = -1, high = 1, size = (2,self.mix_comp_num))
        
        #generate path
        for i in range(self.mix_comp_num):
            x1_vec[:,i] = self.make_GP_trajectory(self.time_tics) + starting_points[0,i]
            x2_vec[:,i] = self.make_GP_trajectory(self.time_tics) + starting_points[1,i]
        
        self.x1 = x1_vec
        self.x2 = x2_vec
        #return x1_vec, x2_vec
    
    def get_mixture_component_weights(self):
        #known mixture weights
        return np.array([.1,.1,.2,.25,.35])

    def allocate_data_to_components(self):
        time_tic_allocation_counts = (
            np.random.multinomial(self.n_per_tic,self.get_mixture_component_weights()))
        return time_tic_allocation_counts
    
    def generate_DGM_data(self):
        initial=True
        all_component_scale = np.random.uniform(low=.05,high=0.6)
        for tt in range(len(self.time_tics)):
            # get component data allocation
            comp_aloc = self.allocate_data_to_components()
            # sample data for each component
            location = np.array([self.x1[tt], self.x2[tt]])
            
            for mc in range(self.mix_comp_num):
                data = np.random.normal(loc=location[:,mc], scale=all_component_scale, size=(comp_aloc[mc],2))
                if initial:
                    output = pd.DataFrame(data)
                    output['time'] = tt
                    output['mu_0'] = location[0,mc]
                    output['mu_1'] = location[1,mc]
                    output['scale'] = all_component_scale
                    
                    initial=False
                else:
                    temp_df = pd.DataFrame(data)
                    temp_df['time'] = tt
                    temp_df['mu_0'] = location[0,mc]
                    temp_df['mu_1'] = location[1,mc]
                    temp_df['scale'] = all_component_scale
                    output = output.append(temp_df)
        self.data = output.reset_index(drop=True)
        self.data.to_csv(self.output_folder_name + "fn=0.csv" )
