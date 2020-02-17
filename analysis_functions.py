import analysis_functions as af

import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt

def plot_between_shard_var_param_i(params_A, params_B, param_i):
    plt.figure(figsize=(30,3))
    A = params_A.between_shard_var_pred_i(pred_i=param_i)
    plt.plot(range(len(A)), A, '-', label="A")
    B = params_B.between_shard_var_pred_i(pred_i=param_i)
    plt.plot(range(len(B)), B, '-', label="B")
    legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
    
def plot_within_shard_var_param_i(params_A, params_B, param_i, shard):
    plt.figure(figsize=(30,3))
    A = params_A.within_shard_var_pred_i_shard_s(pred_i=param_i, shard_s=shard)
    plt.plot(range(len(A)), A, '-', label="A")
    B = params_B.within_shard_var_pred_i_shard_s(pred_i=param_i, shard_s=shard)
    plt.plot(range(len(B)), B, '-', label="B")
    legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')


def make_like_path_plots_by_shard(no_comm, with_comm):
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        for s in range(no_comm.number_of_shards):

            axes[0].plot(no_comm.true_shard_likelihoods[s], 'x', alpha=.3, label="Shard "+str(s)+", True", color='red')
            axes[0].plot(no_comm.est_shard_likelihoods[s], 'o', alpha=.3, label="Shard "+str(s)+", Estimate", color='blue')
            axes[0].plot(with_comm.est_shard_likelihoods[s], 'v', alpha=.3, label="Shard "+str(s)+", Estimate", color='green')
            #legend = axes[0].legend(loc='upper left', shadow=True, fontsize='x-large')
            
            diff = np.subtract(no_comm.true_shard_likelihoods[s], no_comm.est_shard_likelihoods[s])
            axes[1].plot( 
                diff , 
                '-', 
                alpha=.3, 
                label="Shard "+str(s)+", True - Estimate, mead diff = " + str(np.mean(diff))
            )
            diff = np.subtract(with_comm.true_shard_likelihoods[s], with_comm.est_shard_likelihoods[s])
            axes[1].plot( 
                diff , 
                '-', 
                alpha=.3, 
                label="Shard "+str(s)+", True - Estimate, mead diff = " + str(np.mean(diff))
            )
            #legend = axes[1].legend(loc='upper left', shadow=True, fontsize='x-large')
        
        axes[0].plot(no_comm.avg_true_shard_likelihoods, 'x', alpha=.6, label="True", color='black')
        axes[0].plot(no_comm.avg_est_shard_likelihoods, 'o', alpha=.6, label="Estimate", color='blue')
        axes[0].plot(with_comm.avg_est_shard_likelihoods, 'v', alpha=.6, label="Estimate", color='green')
        diff = np.subtract(no_comm.avg_true_shard_likelihoods, no_comm.avg_est_shard_likelihoods)
        axes[1].plot( 
            diff , 
            'o', 
            alpha=.6, 
            label="True - Estimate, mead diff = " + str(np.mean(diff))
        )
        diff = np.subtract(with_comm.avg_true_shard_likelihoods, with_comm.avg_est_shard_likelihoods)
        axes[1].plot( 
            diff , 
            'v', 
            alpha=.6, 
            label="True - Estimate, mead diff = " + str(np.mean(diff))
        )
        plt.show()
        #print("********************************")


class analyze_run:
    
    def __init__(self, f_path, f_beta_file_path, f_step_size, true_cols, comm, col='final_params'):
        self.comm = comm
        self.true_cols = true_cols
        self.col = col
        if comm:
            self.Betas_in_columns = self.get_params_from_results_with_comm(f_path)
        else:
            self.Betas_in_columns = self.get_params_from_results_no_comm(f_path)
        self.beta_i_avg = self.get_beta_i_avg(self.Betas_in_columns)
        self.beta_i_var = self.get_beta_i_var(self.Betas_in_columns)
        self.skip_size = f_step_size # SKIP SIZE BASED ON TIME STEPS SKIPPED
        self.Beta_true_data = pd.read_csv(f_beta_file_path)
        #print("True Beta shape = ", self.Beta_true_data.shape)
        self.Beta_com = self.Beta_true_data[self.true_cols][(self.skip_size-1)::self.skip_size]#Beta_true_data.B_0.index % skip_size == 0]
        self.Beta_com.reset_index(inplace=True)
        self.predictor_number = self.Betas_in_columns[0].shape[0]
        self.epoch_number = len(self.Betas_in_columns)
        
        #print("len(f_Betas_in_columns = self.Betas_in_columns)=", len(self.Betas_in_columns))
        #print("len(f_Betas_in_columns = self.Betas_in_columns[0])=", len(self.Betas_in_columns[0]))
        
        self.params_PREDxPARTxSHARDxEPOCH = self.params_PREDxPARTxSHARDxEPOCH(
            f_Betas_in_columns = self.Betas_in_columns, 
            f_pred_num  = self.number_of_predictors, 
            f_part_num  = self.number_of_particles, 
            f_shard_num = self.number_of_shards, 
            f_epoch_num = self.epoch_number
        )
        
        self.true_lik, self.esti_lik = self.get_plot_likelihoods(self.Beta_com, self.true_cols, self.beta_i_avg)
        
        self.particle_history_ids = self.id_cleaner(path=f_path, column_name='particle_history_ids')
        self.machine_history_ids = self.id_cleaner(path=f_path, column_name='machine_history_ids')
        if comm:
            self.post_particle_history_ids = self.id_cleaner(path=f_path, column_name='post_particle_history_ids')
            self.post_machine_history_ids = self.id_cleaner(path=f_path, column_name='post_machine_history_ids')
        
    def get_particle_id_counts(self):
        flat_particle_history_ids = self.particle_history_ids.flatten()
        flat_machine_history_ids = self.machine_history_ids.flatten()
        
        history_df = pd.DataFrame(
            {
                machine_id: flat_machine_history_ids, 
                partilce_id:flat_particle_history_ids
            }
        )
        return history_df.groupby(['machine_id', 'partilce_id']).size()

    def get_Beta_t(path, Beta_index = -1):
        B_t = pd.read_csv(path, index_col=0).iloc[Beta_index,:]
        return np.array(B_t.T)
    
    
    def compute_lik(self, f_X, f_Y, f_B):
        x_j_tB = np.matmul(f_X.as_matrix() , f_B)
        p_of_x_i = 1.0/(1.0+np.exp(-1*x_j_tB))
        likelihood =  f_Y*p_of_x_i + (1-f_Y)*(1-p_of_x_i)
        return likelihood
    
    
    def generate_OOS_X_y(self, f_B_t):
        X_i =  pd.DataFrame(
            np.random.uniform(
                low=-1,
                high = 1,
                size=(10000,len(f_B_t))
            )
        )
        X_B_t = X_i.dot(np.array(f_B_t))
        event_prob = 1.0/(1.0+np.exp(-1*X_B_t))
        Y_vals = np.random.binomial(n=1, p=event_prob, size=None)
        return X_i, Y_vals
    
    def params_PREDxPARTxSHARDxEPOCH(self, f_Betas_in_columns, f_pred_num, f_part_num, f_shard_num, f_epoch_num):
        f_params_PREDxPARTxSHARDxEPOCH = np.zeros(
            (f_pred_num, f_part_num, f_shard_num, f_epoch_num)
        )
        for tt in range(len(f_Betas_in_columns)):
            for i in range(f_pred_num):
                #print("f_pred_num=", f_pred_num)
                single_beta_all_part_and_s = np.array(f_Betas_in_columns[tt][i,:])
                for s in range(f_shard_num):
                    f_params_PREDxPARTxSHARDxEPOCH[i,:,s,tt] = (
                        single_beta_all_part_and_s[s*f_part_num:(s+1)*f_part_num]
                    )            
        return f_params_PREDxPARTxSHARDxEPOCH
    
    def get_params_from_results_with_comm(self, path):
        results_output = pd.read_csv(path)
        #print(results_output.shape)
        f_Betas_in_columns = list()
        if self.col == 'final_params':
            for nr in range(len(results_output.final_params[:-1])):
                output = list()
                dirty_list = results_output.final_params[nr].split(',')
                
                self.number_of_particles = int(results_output.particle_number[nr])
                self.number_of_shards = int(results_output.shards[nr])
                self.number_of_predictors = int(results_output['p='][nr])
                for i in range(len(dirty_list)):
                    single_particle_params = re.findall(r'-?\d+\.?\d*',dirty_list[i])
                    if len(single_particle_params)==0: 
                        continue
                    test_list = list(map(float, single_particle_params))[0] 
                    output.append(test_list)
                f_Betas_in_columns.append(
                    np.array(output).reshape(
                        (self.number_of_particles*self.number_of_shards,self.number_of_predictors)
                    ).T
                )
        if self.col == 'pre_shuffel_params':
            for nr in range(len(results_output.pre_shuffel_params[:-1])):
                output = list()
                dirty_list = results_output.pre_shuffel_params[nr].split(',')
                
                self.number_of_particles = int(results_output.particle_number[nr])
                self.number_of_shards = int(results_output.shards[nr])
                self.number_of_predictors = int(results_output['p='][nr])
                for i in range(len(dirty_list)):
                    single_particle_params = re.findall(r'-?\d+\.?\d*',dirty_list[i])
                    if len(single_particle_params)==0: 
                        continue
                    test_list = list(map(float, single_particle_params))[0] 
                    output.append(test_list)
                f_Betas_in_columns.append(
                    np.array(output).reshape(
                        (self.number_of_particles*self.number_of_shards,self.number_of_predictors)
                    ).T
                )
        if self.col == 'post_shuffel_params':
            for nr in range(len(results_output.post_shuffel_params[:-1])):
                output = list()
                dirty_list = results_output.post_shuffel_params[nr].split(',')
                
                self.number_of_particles = int(results_output.particle_number[nr])
                self.number_of_shards = int(results_output.shards[nr])
                self.number_of_predictors = int(results_output['p='][nr])
                for i in range(len(dirty_list)):
                    single_particle_params = re.findall(r'-?\d+\.?\d*',dirty_list[i])
                    if len(single_particle_params)==0: 
                        continue
                    test_list = list(map(float, single_particle_params))[0] 
                    output.append(test_list)
                f_Betas_in_columns.append(
                    np.array(output).reshape(
                        (self.number_of_particles*self.number_of_shards,self.number_of_predictors)
                    ).T
                )
        #print("number of time epochs = " , len(f_Betas_in_columns))
        #print("number of predictors = ", len(f_Betas_in_columns[0]))
        #print("number of particles accorss allshards = ", len(f_Betas_in_columns[0][0]))
        #print(f_Betas_in_columns[0].shape)
        return f_Betas_in_columns
    
    
    def get_params_from_results_no_comm(self, path):
        results_output = pd.read_csv(path)
        #print(results_output.shape)
        f_Betas_in_columns = list()
        for nr in range(len(results_output.final_params[:-1])):
            output = list()
            dirty_list = results_output.final_params[nr].split(',')
            
            self.number_of_particles = int(results_output.particle_number[nr])
            self.number_of_shards = int(results_output.shards[nr])
            self.number_of_predictors = int(results_output['p='][nr])
            for i in range(len(dirty_list)):
                single_particle_params = re.findall(r'-?\d+\.?\d*',dirty_list[i])
                if len(single_particle_params)==0: 
                    continue
                test_list = list(map(float, single_particle_params))[0] 
                output.append(test_list)
            #print(output[:10])
            f_Betas_in_columns.append(
                np.array(output).reshape(
                    (self.number_of_particles*self.number_of_shards,self.number_of_predictors)
                ).T
            )
        #print("number of time epochs = " , len(f_Betas_in_columns))
        #print("number of predictors = ", len(f_Betas_in_columns[0]))
        #print("number of particles accorss allshards = ", len(f_Betas_in_columns[0][0]))
        #print(f_Betas_in_columns[0].shape)
        return f_Betas_in_columns
    
    
    def get_beta_i_avg(self, f_Betas_in_columns):
        #print("len(f_Betas_in_columns)", len(f_Betas_in_columns))
        #print("len(f_Betas_in_columns[0])", len(f_Betas_in_columns[0]))
        #print("len(f_Betas_in_columns[0][0])", len(f_Betas_in_columns[0][0]))
        p_num = len(f_Betas_in_columns[0])
        all_bi_avg = list()
        for i in range(p_num):
            all_bi_avg.append(list())
        
        for tt in range(len(f_Betas_in_columns)):
            for i in range(p_num):
                all_bi_avg[i].append(np.mean(f_Betas_in_columns[tt][i,:]))
                
        return np.array(all_bi_avg)
    
    def get_beta_i_var(self, f_Betas_in_columns):
        
        p_num = len(f_Betas_in_columns[0])
        all_bi_avg = list()
        for i in range(p_num):
            all_bi_avg.append(list())
        
        for tt in range(len(f_Betas_in_columns)):
            for i in range(p_num):
                all_bi_avg[i].append(np.var(f_Betas_in_columns[tt][i,:]))
                
        return np.array(all_bi_avg)
    
    
    def get_plot_likelihoods(self, f_Beta_com, f_cols, f_beta_i_avg):
        #print("In get_plot_likelihoods")
        #print("f_Beta_com.shape=", f_Beta_com.shape)
        #print("f_beta_i_avg.shape", f_beta_i_avg.shape)
        f_true_lik = list()
        f_esti_lik = list()
        #print("type(f_Beta_com)=", type(f_Beta_com))
        #print("type(f_beta_i_avg)=", type(f_beta_i_avg))
        #print("f_Beta_com.shape=", f_Beta_com.shape)
        #print("f_beta_i_avg.shape=", f_beta_i_avg.shape)
        for i in range(f_Beta_com.shape[0]):
            Beta_t = f_Beta_com[f_cols].loc[i]
            Beta_fit = f_beta_i_avg[:,i]
    
            X, y = self.generate_OOS_X_y(f_B_t=Beta_t)
            f_true_lik.append(np.mean(self.compute_lik(f_X=X, f_Y=y, f_B=Beta_t)))
            f_esti_lik.append(np.mean(self.compute_lik(f_X=X, f_Y=y, f_B=Beta_fit)))
        return f_true_lik, f_esti_lik
    
    
    def make_like_path_plots(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
            
        axes[0].plot(self.true_lik, 'x', alpha=1, label="True")
        axes[0].plot(self.esti_lik, 'o', alpha=1, label="Estimate")
        legend = axes[0].legend(loc='upper left', shadow=True, fontsize='x-large')
        
        diff = np.subtract(self.true_lik, self.esti_lik)
        axes[1].plot( 
            diff , 
            '-', 
            alpha=1, 
            label="True - Estimate, mead diff = " + str(np.mean(diff))
        )
        legend = axes[1].legend(loc='upper left', shadow=True, fontsize='x-large')
    
        plt.show()
    
    
    def make_var_path_plots(self):
        plt.figure(figsize=(30,4))
        for i in range(self.beta_i_var.shape[0]):
            plt.plot(range(self.beta_i_var.shape[1]), self.beta_i_var[i,:], '-')
    
    
    def analyze_run_with_comm(f_path, f_beta_file_path, f_step_size, true_cols):
        Betas_in_columns = get_params_from_results_with_comm(f_path)
        beta_i_avg = get_beta_i_avg(Betas_in_columns)
        beta_i_var = get_beta_i_var(Betas_in_columns)
        
        skip_size = f_step_size # SKIP SIZE BASED ON TIME STEPS SKIPPED
        Beta_true_data = pd.read_csv(f_beta_file_path)
        #print("True Beta shape = ", Beta_true_data.shape)
        Beta_com = Beta_true_data[true_cols][(skip_size-1)::skip_size]#Beta_true_data.B_0.index % skip_size == 0]
        Beta_com.reset_index(inplace=True)
        
        true_lik, esti_lik = get_plot_likelihoods(Beta_com, true_cols, beta_i_avg)
        make_like_path_plots(true_lik, esti_lik)
        make_var_path_plots(beta_i_var)
        
        #print("Final Likelihood = ", esti_lik[-1])
    
    
    def analyze_run_no_comm(f_path, f_beta_file_path, f_step_size, true_cols):
        Betas_in_columns = get_params_from_results_no_comm(f_path)
        beta_i_avg = get_beta_i_avg(Betas_in_columns)
        beta_i_var = get_beta_i_var(Betas_in_columns)
        
        skip_size = f_step_size # SKIP SIZE BASED ON TIME STEPS SKIPPED
        Beta_true_data = pd.read_csv(f_beta_file_path)
        #print("True Beta shape = ", Beta_true_data.shape)
        Beta_com = Beta_true_data[true_cols][(skip_size-1)::skip_size]#Beta_true_data.B_0.index % skip_size == 0]
        Beta_com.reset_index(inplace=True)
        
        true_lik, esti_lik = get_plot_likelihoods(Beta_com, true_cols, beta_i_avg)
        make_like_path_plots(true_lik, esti_lik)
        make_var_path_plots(beta_i_var)
        
        #print("Final Likelihood = ", esti_lik[-1])
        
    def within_shard_var_pred_i_shard_s(self, pred_i, shard_s):
        nepoch=self.epoch_number#
        var_out = np.ones(nepoch)*(-1000000)
        for ne in range(nepoch):
            var_out[ne] = np.var(self.params_PREDxPARTxSHARDxEPOCH[pred_i,:,shard_s,ne])
        return var_out
    
    def between_shard_var_pred_i(self, pred_i):
        sshards = self.number_of_shards
        nepoch=self.epoch_number
        var_out = np.ones(nepoch) -1000000
        
        for ne in range(nepoch):
            group_means = np.zeros(sshards)+100000
            for s in range(sshards):
                group_means[s] = np.mean(self.params_PREDxPARTxSHARDxEPOCH[pred_i,:,s,ne])
                
            var_out[ne] = np.var(group_means)
        return var_out

    def likelihood_by_shard(self):
        sshards = self.number_of_shards
        nepoch=self.epoch_number
        var_out = np.ones(nepoch) -1000000
        
        est_shard_likelihoods = list()
        true_shard_likelihoods = list()
        beta_avg = np.zeros((self.number_of_predictors, self.epoch_number))
        for s in range(sshards):
            #like_s = np.zeros(nepoch)
            for ne in range( nepoch):
                f_Betas_in_columns = self.params_PREDxPARTxSHARDxEPOCH[:,:,s,ne]
                beta_avg[:,ne] = f_Betas_in_columns.mean(axis=1)
                
            true_lik, esti_lik = self.get_plot_likelihoods(self.Beta_com, self.true_cols, beta_avg)
            est_shard_likelihoods.append(esti_lik)
            true_shard_likelihoods.append(true_lik)
        self.true_shard_likelihoods = true_shard_likelihoods
        self.est_shard_likelihoods = est_shard_likelihoods    
    
    def avg_likelihood_by_shard(self):
        sshards = self.number_of_shards
        nepoch=self.epoch_number
        #var_out = np.ones(nepoch) -1000000
        
        est_shard_likelihoods = list()
        true_shard_likelihoods = list()
        beta_avg = np.zeros((self.number_of_predictors, self.epoch_number))
        #like_s = np.zeros(nepoch)
        #for s in range(sshards):
        #    like_s[,,s] = f_Betas_in_columns = self.params_PREDxPARTxSHARDxEPOCH[:,:,:,ne]
        #    beta_avg[:,ne] = f_Betas_in_columns.mean(axis=(1,2))
        #for ne in range( nepoch):
        #    f_Betas_in_columns = self.params_PREDxPARTxSHARDxEPOCH[:,:,:,ne]
        #    beta_avg[:,ne] = f_Betas_in_columns.mean(axis=(1,2))
            
        beta_avg = self.params_PREDxPARTxSHARDxEPOCH.mean(axis=(1,2))
        
        true_shard_likelihoods, est_shard_likelihoods = self.get_plot_likelihoods(self.Beta_com, self.true_cols, beta_avg)
        #est_shard_likelihoods.append(esti_lik)
        #true_shard_likelihoods.append(true_lik)
        self.avg_true_shard_likelihoods = true_shard_likelihoods
        self.avg_est_shard_likelihoods = est_shard_likelihoods    
        print("UPDATES ARE BEING ADDED")
    
    def make_like_path_plots_by_shard(self):
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        for s in range(self.number_of_shards):
            
            
                
            axes[0].plot(self.true_shard_likelihoods[s], 'x', alpha=.3, label="Shard "+str(s)+", True", color='red')
            axes[0].plot(self.est_shard_likelihoods[s], 'o', alpha=.3, label="Shard "+str(s)+", Estimate", color='blue')
            #legend = axes[0].legend(loc='upper left', shadow=True, fontsize='x-large')
            
            diff = np.subtract(self.true_shard_likelihoods[s], self.est_shard_likelihoods[s])
            axes[1].plot( 
                diff , 
                '-', 
                alpha=.3, 
                label="Shard "+str(s)+", True - Estimate, mead diff = " + str(np.mean(diff))
            )
            #legend = axes[1].legend(loc='upper left', shadow=True, fontsize='x-large')
        
        axes[0].plot(self.avg_true_shard_likelihoods, 'x', alpha=.6, label="True", color='black')
        axes[0].plot(self.avg_est_shard_likelihoods, 'o', alpha=.6, label="Estimate", color='black')
        diff = np.subtract(self.avg_true_shard_likelihoods, self.avg_est_shard_likelihoods)
        axes[1].plot( 
            diff , 
            '-', 
            alpha=.6, 
            label="True - Estimate, mead diff = " + str(np.mean(diff))
        )
        plt.show()
        #print("********************************")
    
    def get_beta_i_avg_param_Tensor(self):
        #print("len(f_Betas_in_columns)", len(f_Betas_in_columns))
        #print("len(f_Betas_in_columns[0])", len(f_Betas_in_columns[0]))
        #print("len(f_Betas_in_columns[0][0])", len(f_Betas_in_columns[0][0]))
        p_num = len(f_Betas_in_columns[0])
        all_bi_avg = list()
        for i in range(p_num):
            all_bi_avg.append(list())
        
        for tt in range(len(f_Betas_in_columns)):
            for i in range(p_num):
                all_bi_avg[i].append(np.mean(f_Betas_in_columns[tt][i,:]))
                
        return np.array(all_bi_avg)
    
    def id_cleaner(self, path, column_name):
        df = pd.read_csv(path)
        df[column_name][1].split(', ')
        all_integer_ids=list()
        for nr in range(len(df[column_name][:-1])):
            output = list()
            dirty_list = df[column_name][nr].split(',')
            
            number_of_particles = int(df.particle_number[nr])
            number_of_shards = int(df.shards[nr])
            number_of_predictors = int(df['p='][nr])
            for i in range(len(dirty_list)):
                single_particle_params = re.findall(r'-?\d+\.?\d*',dirty_list[i])
                if len(single_particle_params)==0: 
                    continue
                test_list = list(map(float, single_particle_params))[0] 
                output.append(test_list)
            all_integer_ids.append(
                np.array(output)
            )
        return np.array(all_integer_ids)
    
    def get_unique_particle_count_by_epoch(self, pre_post = 'pre'):
        epoch_number = self.epoch_number
        total_particle_count = np.zeros(epoch_number)
        
        for i in range(epoch_number):
            if pre_post=='pre':
                df = pd.DataFrame(
                    {
                        'machine_history_ids':self.machine_history_ids[i],
                        'particle_history_ids':self.particle_history_ids[i],
                    }
                )            #df.groupby(['col1','col2']).size()
                total_particle_count[i] = len(df.groupby(['machine_history_ids','particle_history_ids']).size())
            else:
                df = pd.DataFrame(
                    {
                        'post_machine_history_ids':self.post_machine_history_ids[i],
                        'post_particle_history_ids':self.post_particle_history_ids[i],
                    }
                )            #df.groupby(['col1','col2']).size()
                total_particle_count[i] = len(df.groupby(['post_machine_history_ids','post_particle_history_ids']).size())
        return total_particle_count

    def get_unique_particle_count_by_epoch_and_shard(self, pre_post = 'pre'):
        epoch_number = self.epoch_number
        shard_id = list(range(0,self.number_of_shards))*self.number_of_particles
        shard_id.sort()
        number_of_shards = self.number_of_shards
        shard_temp = list()
        for i in range(epoch_number):
            if pre_post=='pre':
                
                df = pd.DataFrame(
                    {
                        'shard_id'           :shard_id,
                        'machine_history_ids':self.machine_history_ids[i],
                        'particle_history_ids':self.particle_history_ids[i],
                    }
                )            
                shard_temp.append(df)
            else:
                df = pd.DataFrame(
                    {
                        'shard_id'                 :shard_id,
                        'post_machine_history_ids' :self.post_machine_history_ids[i],
                        'post_particle_history_ids':self.post_particle_history_ids[i],
                    }
                )
                shard_temp.append(df)
                
        total_particle_count = np.zeros((epoch_number, number_of_shards))
        for en in range(epoch_number):
            for s in range(number_of_shards):
                index = shard_temp[en].shard_id == s
                temp_machine_stats_df = shard_temp[en][index]
                if pre_post=='pre':
                    total_particle_count[en, s] = len(
                        temp_machine_stats_df.groupby(
                            ['shard_id','machine_history_ids','particle_history_ids']
                        ).size()
                    )
                else:
                    total_particle_count[en, s] = len(
                        temp_machine_stats_df.groupby(
                            ['shard_id','post_machine_history_ids','post_particle_history_ids']
                        ).size()
                    )
                
        return total_particle_count