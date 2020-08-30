import pandas as pd
import numpy as np


class prep_data:
    def __init__(self, params, path, f_rank):

        self.model=params['model']

        if self.model== "probit_sin_wave":
            print("working: ", path)
            loaded_df = pd.read_csv(path)[f_rank::params['shards']]
            loaded_df = loaded_df.reset_index(drop=True)
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
            #self.data_keys=list()

            self.b=np.zeros((self.N,self.p))
            self.b_oos=np.zeros((1,self.p))
            self.B=full_de_mat.Bo_std.iloc[0]
            self.Tau_inv_std = full_de_mat.Tau_inv_std.iloc[0]
            
            self.shard_output = {}
            self.shard_output['Y'] = self.Y
            self.shard_output['time_value'] = full_de_mat['time'].astype(float)
            self.shard_output['X_matrix'] = full_de_mat[self.predictor_names].values
            self.shard_output['N'] = self.N
            self.shard_output['b'] = self.b
            self.shard_output['B'] = self.B
            self.shard_output['Tau_inv_std'] = self.Tau_inv_std
            self.shard_output['model'] = self.model
            self.shard_output['shards'] = self.shards
            self.shard_output['all_shard_unique_time_values'] = all_shard_unique_time_values
            self.shard_output['predictors'] = self.predictor_names
            self.shard_output['p'] = self.p

    def format_for_scatter(self, epoch, f_rank):
        return(self.shard_output)