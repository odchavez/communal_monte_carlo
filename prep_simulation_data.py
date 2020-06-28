import pandas as pd
import numpy as np


class prep_data:
    def __init__(self, params, path):

        self.model=params['model']

        if self.model== "probit_sin_wave":
            loaded_df = pd.read_csv(path, low_memory=False, index_col=0)
            #print("loaded_df.shape = ", loaded_df.shape)
            #print("loaded_df.head() = ", loaded_df.head())
            test_cols = loaded_df.columns
            full_de_mat = loaded_df.loc[:,test_cols[-params['p_to_use']:]]
            
            full_de_mat.time = full_de_mat.time+1
            all_shard_unique_time_values = full_de_mat.time.unique()

            full_de_mat_X = full_de_mat.copy()
            drop_these=[
                "y", 'time', 'Tau_inv_std', 'Bo_std'
            ]
            self.predictor_names = [x for x in full_de_mat.columns if x not in drop_these]
            #print("self.predictor_names = ", self.predictor_names)
            full_de_mat_X = full_de_mat[self.predictor_names]

            p=len(self.predictor_names)

            N=full_de_mat_X.shape[0]
            temp_params={
                'p': p,
                'b': np.zeros(p),
                'N': N,
                'epoch_at':[N]
            }

            params.update(temp_params)
            self.p=params['p']
            self.N=params['N']
            self.N_batch = params['N_batch']
            self.shards=params['shards']
            self.epoch_at=params['epoch_at']
            self.epoch_number=len(self.epoch_at)
            self.randomize_shards = params['randomize_shards']
            self.Y=np.zeros(self.N)
            self.X_matrix=np.zeros(
                (full_de_mat_X.shape[0],
                 len(self.predictor_names)
                )
            )
            self.data_keys=list()

            self.b=np.zeros((self.N,self.p))
            self.b_oos=np.zeros((1,self.p))
            self.B=full_de_mat.Bo_std.iloc[0]
            self.Tau_inv_std = full_de_mat.Tau_inv_std.iloc[0]

            output = {}
            for m in range(self.shards):
                output["shard_"+str(m)]     ={}
                output["shard_"+str(m)]['Y']=np.zeros(self.N)#{}
                output["shard_"+str(m)]['data_keys']=list()
                output["shard_"+str(m)]['time_value']=list()

            epoch_output = {}
            for ep in range(self.epoch_number):
                epoch_output["epoch"+str(ep)]={}
                for m in range(self.shards):
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]     ={}
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['Y']=np.zeros(self.N)
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['data_keys']=list()
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['time_value']=list()
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix']=np.zeros(
                        (
                            full_de_mat_X.shape[0],
                            len(self.predictor_names)
                        )
                    )

            data_index=0
            epoch_counter=0
            for i in range(self.N):
                key=str(i)+":"+str(data_index)
                all_key=str(i)+":"+str(i)
                self.data_keys.append(all_key)
                if self.randomize_shards == 0:
                    s="shard_"+str(i%self.shards)
                else:
                    s = "shard_"+str(np.random.randint(0,self.shards))

                temp_X = np.array(full_de_mat_X.iloc[i]).astype(np.float64)
                self.Y[i]           = np.zeros(self.N_batch)
                self.X_matrix[i,:] = temp_X
                output[s]['Y'][i] = np.zeros(self.N_batch)
                output[s]['data_keys'].append(key)
                output[s]['time_value'].append(float(full_de_mat['time'].iloc[i]))

                output[s]['Y'][i] = self.Y[i] = np.float(full_de_mat['y'].iloc[i])#1

                if i not in self.epoch_at:
                    epoch_output["epoch"+str(epoch_counter)][s]['X_matrix'][i,:] = temp_X
                    epoch_output["epoch"+str(epoch_counter)][s]['Y'][i]=self.Y[i]
                    epoch_output["epoch"+str(epoch_counter)][s]['data_keys'].append(key)
                    epoch_output["epoch"+str(epoch_counter)][s]['time_value'].append(float(full_de_mat['time'].iloc[i]))
                else:
                    for m in range(self.shards):
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['Y'][i] = self.Y[i]
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['data_keys'].append(all_key)
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['time_value'].append(float(full_de_mat['time'].iloc[i]))

                    epoch_counter+=1

                if i%self.shards == self.shards-1:
                    data_index+=1

            tttemp=np.max([100,self.N_batch])
            self.X_oos=np.random.uniform(-1,1,self.p*tttemp).reshape((tttemp,self.p))

            for m in range(self.shards):
                output["shard_"+str(m)]['N'] = self.N
                output["shard_"+str(m)]['b'] = self.b
                output["shard_"+str(m)]['B'] = self.B
                output["shard_"+str(m)]['p'] = self.p
                output["shard_"+str(m)]['Tau_inv_std'] = self.Tau_inv_std
                output["shard_"+str(m)]['model'] = self.model
                output["shard_"+str(m)]['shards']=self.shards

            for ep in range(self.epoch_number):
                epoch_output["epoch"+str(ep)]['parallel_shards']    = self.shards
                epoch_output["epoch"+str(ep)]['b']                  = self.b
                epoch_output["epoch"+str(ep)]['B']                  = self.B
                epoch_output["epoch"+str(ep)]['Tau_inv_std']            = self.Tau_inv_std
                epoch_output["epoch"+str(ep)]['all_shard_unique_time_values'] = all_shard_unique_time_values

                for m in range(self.shards):
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['N']      = self.N
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['b']      = self.b
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['B']      = self.B
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['Tau_inv_std']      = self.Tau_inv_std
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['model']  = self.model
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['shards'] =self.shards
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['all_shard_unique_time_values'] = all_shard_unique_time_values

                    keep_rows_index = np.sum(epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'], axis=1)!=0

                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['predictors'] = self.predictor_names
                    if 'time' in self.predictor_names:

                        epoch_output["epoch"+str(ep)]["shard_"+str(m)]['time'] = (
                            epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'][keep_rows_index,-1]
                        )

                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['p'] = self.p
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'] = (
                        epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'][keep_rows_index,:]
                    )
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['Y'] = (
                        epoch_output["epoch"+str(ep)]["shard_"+str(m)]['Y'][keep_rows_index]
                    )
                    
            #print("self.X_matrix.shape = ", self.X_matrix.shape)
            self.output=output
            self.output['epoch_data']=epoch_output
            self.output['X_matrix']=self.X_matrix
            self.output['Y']=self.Y
            self.output['N']=self.N
            self.output['b']=self.b
            self.output['B']=self.B
            self.output['p']=self.p
            self.output['Tau_inv_std']=self.Tau_inv_std
            self.output['b_oos']=self.b_oos
            self.output['X_oos']=self.X_oos
            self.output['batch_number']=self.N_batch
            self.output['model']=self.model
            self.output['shards']=1
            self.output['parallel_shards']=self.shards
            self.output['data_keys']=self.data_keys
            self.output['predictors']=self.predictor_names
            self.output['all_shard_unique_time_values'] = all_shard_unique_time_values

    def get_data(self):
        return(self.output)

    def format_for_scatter(self, epoch):
        output = []
        for m in range(self.shards):
            output.append(self.output['epoch_data']["epoch"+str(epoch)]["shard_"+str(m)])
        return(output)
