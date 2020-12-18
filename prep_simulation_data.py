import pandas as pd
import numpy as np
import random


class prep_data:
    def __init__(self):
        self.starting_index = 0
        return
    
    def load_new_data(self, params, path, shard_subset=None):
        self.model=params['model']

        if self.model== "probit_sin_wave":
            
            loaded_df = pd.read_csv(path, low_memory=False, index_col=0)
                
            loaded_df = loaded_df.reset_index(drop=True)
            test_cols = loaded_df.columns
            full_de_mat = loaded_df.loc[:,test_cols[-params['p_to_use']:]]
            full_de_mat.time = full_de_mat.time+1

            full_de_mat_X = full_de_mat.copy()
            drop_these=[
                "y", 'time', 'Tau_inv_std', 'Bo_std'
            ]
            self.predictor_names = [x for x in full_de_mat.columns if x not in drop_these]
            full_de_mat_X = full_de_mat[self.predictor_names]
            
            p=len(self.predictor_names)

            N=full_de_mat_X.shape[0]
            temp_params={
                'p': p,
                'b': np.zeros(p),
                'N': N,
                'epoch_at':[N],
            }

            params.update(temp_params)
            self.p=params['p']
            self.N=params['N']
            self.N_batch = params['N_batch']
            self.shards=params['shards']
            self.epoch_at=params['epoch_at']
            self.epoch_number=len(self.epoch_at)
            self.randomize_shards = params['randomize_shards']
            
            self.data_keys=list()

            self.b=np.zeros((self.N,self.p))
            self.b_oos=np.zeros((1,self.p))
            self.B=full_de_mat.Bo_std.iloc[0]
            self.Tau_inv_std = full_de_mat.Tau_inv_std.iloc[0]

            self.shard_output = {}
            self.shard_output['time_value'] = full_de_mat['time'].astype(float)
            self.shard_output['N'] = self.N
            self.shard_output['b'] = self.b
            self.shard_output['B'] = self.B
            self.shard_output['Tau_inv_std'] = self.Tau_inv_std
            self.shard_output['model'] = self.model
            self.shard_output['shards'] = self.shards
            
            self.shard_output['predictors'] = self.predictor_names
            self.shard_output['p'] = self.p
            if shard_subset is not None:
                self.shard_output['X_matrix'] = full_de_mat.loc[shard_subset, self.predictor_names].values
                self.shard_output['Y'] = full_de_mat.loc[shard_subset, "y"].values
                self.shard_output['all_shard_unique_time_values'] = (
                    full_de_mat.loc[shard_subset,'time'].unique())
                self.shard_output['time_value'] = full_de_mat.loc[shard_subset,'time'].astype(float)
            else:
                self.shard_output['X_matrix'] = full_de_mat.loc[:, self.predictor_names].values
                self.shard_output['Y'] = full_de_mat.loc[:, "y"].values
                self.shard_output['all_shard_unique_time_values'] = (
                    full_de_mat.loc[:,'time'].unique())
                self.shard_output['time_value'] = full_de_mat.loc[:,'time'].astype(float)
                
    def get_data(self):
        
        return(self.shard_output)
    
    def format_for_scatter(self, epoch, f_rank):
        return(self.shard_output)

    def determin_shard_subsets(self, data_path, size ):
    
        loaded_df = pd.read_csv(data_path, low_memory=False, index_col=0)
        loaded_df = loaded_df.reset_index(drop=True)
        #print("loaded_df.shape",loaded_df.shape) 
        subsets_list = []
        for i in range(size):
            subsets_list.append(list(range(i, loaded_df.shape[0], size)))
                                
        return(subsets_list)
    
    def partition_index(self):
        """
        get the indecies for each shard to be used in epoch
        """
        shard_indecies = list()
        if self.randomize_shards==0:
            #print("sending every ", self.shards, " observation to each shard")
            #print("data size:", self.shard_output['X_matrix'].shape)
            for rank_i in range(self.shards):
                shard_indecies.append(list(range(rank_i, self.N, self.shards)))
            
        else:
            for rank_i in range(self.shards):
                #print("rank ", str(rank_i))
                shard_indecies.append(list())
            print("randomizing data to each shard...but don't worry shard data will stay in order of time")
            list_in = list(range(self.N))
            random.shuffle(list_in)
            #shards_list = list(range(1,self.shards))
            unordered_shards_rank_list = list(range(0,self.shards))
            #shards_list.insert(0,0)
            random.shuffle(unordered_shards_rank_list)
            
            for i in range(len(unordered_shards_rank_list)):
                index_list = list(range(i, self.N, self.shards))
                subset = [list_in[il] for il in index_list]
                subset.sort()
                shard_indecies[unordered_shards_rank_list[i]] = subset
                print("Rank: ", unordered_shards_rank_list[i], " gets:", subset)
        return shard_indecies
            
            
            
            
def make_epoch_files(files_to_process, data_type, file_stem, Epoch_N, code):

    """
    args:
        files_to_process list of file names where files are data for model
        file stem (str): path stem for writing results
        Epoch_N int : how many observations to process accross all shards before communicating.
        
    returns epoch_files_to_process which is a list of strings where each string is a path to a file containing data for one epoch each containing Epoch_N observations.  It's possible there is "leftover data" at the end in which case there will be another epoach < Epoch_N  
    
    create the files needed from files_to_process that are of size Epoch_N
    """

    epoch_files_to_process = list()
    
    
    epoch_counter = 0
    left_over_data = None
    for ftp in range(len(files_to_process)):
        #load data
        if data_type == "synth_data":
            file_data = pd.read_csv(files_to_process[ftp], index_col=0)
        else:
            file_data = pd.read_csv(files_to_process[ftp])
        if left_over_data is not None:
            file_data = left_over_data.append(file_data)
            left_over_data=None
        file_data = file_data.reset_index(drop=True)
        #check how many epochs will fit in file
        epochs_in_file = int(np.floor(file_data.shape[0]/Epoch_N))
        if epochs_in_file >=1:
            start = 0
            end = Epoch_N
            for eif in range(epochs_in_file):
                #print("################################## LOOP ITERATION BEGIN ##################################")
                #print("iteration:"+str(eif)+"out of"+str(range(epochs_in_file)))
                if 'time' in file_data.columns:
                    #print("file_data.iloc[start:end]=", file_data.iloc[start:end])
                    print("file_data.shape"+str(file_data.shape))
                    print("start:end=", str(start)+":"+str(end))
                    temp_index = np.max(file_data.iloc[start:end].index.values)
                    #print("temp_index = ",str(temp_index))
                    current_time = file_data.time[temp_index]
                    #print("current_time=",str(current_time))
                    df_temp=file_data[file_data.time == current_time]
                    #print(df_temp)
                    max_index = np.max(df_temp.index.values)
                    #print("max_index=",str(max_index))

                    if max_index > end:
                        #print("end set in if max_index > end:")
                        end = max_index+1
    
                data_path = data_type + '/temp/' + file_stem + '_epoch='+ str(epoch_counter)+ '_'+ code + '.csv'
                output = file_data.iloc[start:end, :]
                output = output.reset_index(drop=True)
                if output.shape[0] >= Epoch_N:
                    #print("in if output.shape[0] >= Epoch_N:")
                    #print(output)
                    output.to_csv(data_path)
                    print("writing epoch ", epoch_counter, " with t=", current_time," and shape:", output.shape,
                          " in epochs_in_file >=1:")
                    epoch_files_to_process.append(data_path)
                    epoch_counter+=1
                start=end
                end=end+Epoch_N
                #print("End of Loop end:"+str(end)+"start:"+str(start))
                if (end > file_data.shape[0]) and (start < file_data.shape[0]):
                    #print("################################## there is left over data")
                    left_over_data = file_data.iloc[start:, :]
                    #print(left_over_data)
                elif (start >= file_data.shape[0]):
                    #print("in elif (start >= file_data.shape[0]):")
                    break
                else:
                    left_over_data = None
                    
        elif epochs_in_file == 0:
            left_over_data = file_data

    if left_over_data is not None:
        data_path = data_type + '/temp/' + file_stem + '_epoch='+ str(epoch_counter)+ '_'+ code + '.csv'
        output = left_over_data
        output = output.reset_index(drop=True)
        #print("in if left_over_data is not None:")
        #print(output)
        output.to_csv(data_path)
        print("writing epoch ", epoch_counter, " with t=", current_time," and shape:", output.shape, " in left_over_data")
        epoch_files_to_process.append(data_path)
        
    return epoch_files_to_process
            
            
            
            
            