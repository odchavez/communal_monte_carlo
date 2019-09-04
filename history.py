import numpy as np
import os
import glob



class parameter_history:
    
    def __init__(self):
        self.bo_list_history = list()
        self.bo_machine_list_history = list()
    
    def compile_bo_list_history(self, f_name_stem=''):
        nrun, nshard = self.get_particle_history_dim("particle_hold/file_"+f_name_stem+"_*.npy")
        for nr in range(nrun):
            for ns in range(nshard):
                loop_file_name = "particle_hold/file_" + f_name_stem + "_" + str(nr) + "_" + str(ns) + ".npy"
                loaded_bo_file = np.load(loop_file_name)
                if ns == 0:
                    shard_history_temp = loaded_bo_file.copy()
                else:
                    shard_history_temp = np.append(shard_history_temp, loaded_bo_file, axis = 2)
            if nr == 0:
                self.bo_list_history = shard_history_temp.copy()
            else:
                self.bo_list_history = np.append(self.bo_list_history, shard_history_temp, axis = 0)
                
    def get_particle_history_dim(self, history_file_path_name_extension = "particle_hold/*.npy"):
        #print("history get_particle_history_dim")
        txtfiles = []
        for file in glob.glob(history_file_path_name_extension):
            txtfiles.append(file)
        file_list = sorted(txtfiles)
        nrun = 0
        nshard = 0
        for i in range(len(file_list)):
            newstr = ''.join((ch if ch in '0123456789' else ' ') for ch in file_list[i])
            newstr
            listOfNumbers = [int(i) for i in newstr.split()]
            nrun = max(nrun, listOfNumbers[0]+1)
            nshard = max(nshard, listOfNumbers[1]+1)
            
        return nrun, nshard  
    
    def append_bo_list_history(self, parallel_com_obj ):
        #print("history append_bo_list_history")

        temp_all_parts = self.get_temp_all_particles(parallel_com_obj)
        if len(self.bo_list_history) == 0:
            self.bo_list_history = temp_all_parts
        else:
            self.bo_list_history = np.append(self.bo_list_history, temp_all_parts, axis = 0)
        
    def get_temp_all_particles(self, parallel_com_obj):
        #print("history get_temp_all_particles")

        particle_number = len(parallel_com_obj[0].particle_list)
        number_of_shards = len(parallel_com_obj)
        Z_dim = number_of_shards * particle_number
        bo_shape =  parallel_com_obj[0].particle_list[0].bo_list.shape
        temp_all_parts = np.zeros((bo_shape[0], bo_shape[1], Z_dim))
    
        counter = 0
        for sn in range(len(parallel_com_obj)):
            for pn in range(particle_number):
                particle = parallel_com_obj[sn].get_particle(pn)
                temp_all_parts[:,:,counter] = (
                    parallel_com_obj[sn].particle_list[pn].bo_list
                )
                counter+=1    
        return temp_all_parts
    
    def write_param_results(self, f_experiment_path,  f_params_results_file):
        
        params_path = f_experiment_path + f_params_results_file
        #stats_path  = f_experiment_path + f_other_stats_file
        #print("writing paramater results to ", params_path)
        if os.path.exists(f_experiment_path):
            np.save( 
                params_path,
                self.bo_list_history
            )
            
        else:
            os.mkdir(f_experiment_path)
            np.save( 
                params_path,
                self.bo_list_history
            )
            
        
        
    def write_stats_results(self,  f_stats_df, f_other_stats_file='experiment_results/results.csv'):
        # add to existing csv results of 
        #print("writing experimental run statistics to ", f_other_stats_file)
        if os.path.exists(f_other_stats_file):
            # open file and add row
            f_stats_df.to_csv(
                f_other_stats_file, 
                index = False, 
                mode = 'a', 
                header=False
            )
        else:
            #create file
            f_stats_df.to_csv(
                f_other_stats_file, 
                index = False, 
                mode = 'a'
            )
        