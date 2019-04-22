import pandas as pd
import numpy as np

test_cols = [
    'DayOfWeek_1',
    'DayOfWeek_2',
    'DayOfWeek_3',
    'DayOfWeek_4',
    'DayOfWeek_5',
    'DayOfWeek_6',
    'DayOfWeek_7',
    'UniqueCarrier_9E',
    'UniqueCarrier_AA',
    'UniqueCarrier_AQ',
    'UniqueCarrier_AS',
    'UniqueCarrier_B6',
    'UniqueCarrier_CO',
    'UniqueCarrier_DH',
    'UniqueCarrier_DL',
    'UniqueCarrier_EA',
    'UniqueCarrier_EV',
    'UniqueCarrier_F9',
    'UniqueCarrier_FL',
    'UniqueCarrier_HA',
    'UniqueCarrier_HP',
    'UniqueCarrier_ML (1)',
    'UniqueCarrier_MQ',
    'UniqueCarrier_NW',
    'UniqueCarrier_OH',
    'UniqueCarrier_OO',
    'UniqueCarrier_PA (1)',
    'UniqueCarrier_PI',
    'UniqueCarrier_PS',
    'UniqueCarrier_TW',
    'UniqueCarrier_TZ',
    'UniqueCarrier_UA',
    'UniqueCarrier_US',
    'UniqueCarrier_WN',
    'UniqueCarrier_XE',
    'UniqueCarrier_YV',
    'ArrDelay_00',
    'CRSDepTime'
]
class prep_data:
    def __init__(self, params, path):
        
        self.model=params['model']
        
        if self.model== "probit_sin_wave":
            
            full_de_mat = pd.read_csv(path, low_memory=False, index_col=0).loc[test_cols].iloc[:,:20].T
            full_de_mat['intercept'] = 1
            print(".csv dimentions:", full_de_mat.shape)
            
            
            print('full_de_mat.head()')
            print(full_de_mat.head())
            #print(".csv head:", full_de_mat.head(12))
            #print(full_de_mat.index.values)
            #add intercept
            #row = pd.DataFrame(np.ones(len(full_de_mat.columns.values))).T
            #full_de_mat['intercept'] = 1
            #row.columns=full_de_mat.columns.values
            
            #print(full_de_mat.copy().T.head())
            
            full_de_mat_X = full_de_mat.copy()#full_de_mat.copy()
            drop_these=[
                #'Month_1', 
                'DayOfWeek_1', 
                'UniqueCarrier_9E', 
                #'Origin_ABE', 
                #'Dest_ABE', 
                'CRSDepTime',
                #'CRSArrTime',
                #'FormatTime',
                'ArrDelay_00',
                #'ArrDelay_30',
                #'ArrDelay_60'
            ]
            self.predictor_names = list(set(full_de_mat.columns) - set(drop_these))
            
            full_de_mat_X = full_de_mat[self.predictor_names]#.drop(index=drop_these, inplace=True)
            #self.predictor_names = ['intercept'] + list(full_de_mat_X.index)
            #self.predictor_names = list(full_de_mat_X.index)

            print("self.predictor_names=", self.predictor_names)
            #full_de_mat_X = row.append(full_de_mat_X, ignore_index=True)
            
            #full_de_mat_X = full_de_mat_X.T.head()
            #full_de_mat_X.columns=self.predictor_names
            #print(full_de_mat_X_test.head())
            #full_de_mat_X['CRSDepTime'] = full_de_mat.copy().T
            p=len(self.predictor_names)
            #if 'CRSDepTime' in self.predictor_names:
            #    p=len(self.predictor_names)-1
            #else:
            #    p=len(self.predictor_names)

            N=full_de_mat_X.shape[0]
            temp_params={
                'p': p, 
                'b': np.zeros(p), 
                #'B': np.identity(p),
                'N': N, 
                'epoch_at':[N]#range(1000, N, 1000),
            }

            params.update(temp_params)
            self.p=params['p']
            self.N=params['N']
            self.N_batch = params['N_batch']
            self.shards=params['shards']
            self.epoch_at=params['epoch_at']
            self.epoch_number=len(self.epoch_at)#+1
            #self.X={}
            self.Y=np.zeros(self.N)#{}
            self.X_matrix=np.zeros(
                (full_de_mat_X.shape[1], 
                 len(self.predictor_names)#full_de_mat_X.shape[0]+1
                )
            )
            self.data_keys=list()
            #if 'CRSDepTime' in self.predictor_names:
            #    self.p = self.p-1
                
            self.b=np.zeros((self.N,self.p))
            self.b_oos=np.zeros((1,self.p))
            self.B=np.identity(self.p)


            output = {}
            for m in range(self.shards):
                output["shard_"+str(m)]     ={}
                #output["shard_"+str(m)]['X']={}
                output["shard_"+str(m)]['Y']=np.zeros(self.N)#{}
                output["shard_"+str(m)]['data_keys']=list()
                #output["shard_"+str(m)]['X_matrix']=np.zeros(
                #    (
                #        full_de_mat_X.shape[1], 
                #        full_de_mat_X.shape[0]+1
                #    )
                #)
            
            epoch_output = {}
            for ep in range(self.epoch_number):
                epoch_output["epoch"+str(ep)]={}
                for m in range(self.shards):
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]     ={}
                    #epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X']={}
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['Y']=np.zeros(self.N)#{}
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['data_keys']=list()
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix']=np.zeros(
                        (
                            full_de_mat_X.shape[1], 
                            len(self.predictor_names)#+1#full_de_mat_X.shape[0]+1
                        )
                    )
           
            data_index=0
            epoch_counter=0
            print("self.N=", self.N)
            for i in range(self.N):
                print(i)
                key=str(i)+":"+str(data_index)
                all_key=str(i)+":"+str(i)
                self.data_keys.append(all_key)
                s="shard_"+str(i%self.shards)

                temp_X = np.array(full_de_mat_X.iloc[i]).astype(np.float64)
                #self.X[all_key]     = temp_X
                self.Y[i]           = np.zeros(self.N_batch)
                self.X_matrix[i,:] = temp_X
                #self.X_matrix[i,1:] = 1                
                #self.X_matrix[i,0]  = i
                #output[s]['X'][key] = temp_X
                output[s]['Y'][i] = np.zeros(self.N_batch)
                output[s]['data_keys'].append(key)
                #output[s]['X_matrix'][i,0] = i
                #output[s]['X_matrix'][i,1:] = temp_X
                
                output[s]['Y'][i] = self.Y[i] = np.float(full_de_mat['ArrDelay_00'].iloc[i])#1

                if i not in self.epoch_at:
                    #epoch_output["epoch"+str(epoch_counter)][s]['X'][key] = temp_X
                    epoch_output["epoch"+str(epoch_counter)][s]['X_matrix'][i,:] = temp_X
                    #epoch_output["epoch"+str(epoch_counter)][s]['X_matrix'][i,1] = 1
                    #epoch_output["epoch"+str(epoch_counter)][s]['X_matrix'][i,0] = i
                    epoch_output["epoch"+str(epoch_counter)][s]['Y'][i]=self.Y[i] 
                    epoch_output["epoch"+str(epoch_counter)][s]['data_keys'].append(key)
                else:
                    for m in range(self.shards):
                        #epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['X'][all_key] = self.X[all_key]
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['Y'][i] = self.Y[i]
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['data_keys'].append(all_key)
                    epoch_counter+=1
                
                if i%self.shards == self.shards-1:
                    data_index+=1
                    
            tttemp=np.max([100,self.N_batch])
            self.X_oos=np.random.uniform(-1,1,self.p*tttemp).reshape((tttemp,self.p))
            
            sig=np.max(np.var(self.b[0:(self.N-1),:]-self.b[1:,:], axis=0))
            sig=np.max([sig, 1])
            #print("params['B'=",params['B'])
            #*30*sig
            #print("self.B=", self.B)
            #print("self.shards=", self.shards)
            for m in range(self.shards):
                output["shard_"+str(m)]['N'] = self.N
                output["shard_"+str(m)]['b'] = self.b
                output["shard_"+str(m)]['B'] = self.B#*self.shards
                output["shard_"+str(m)]['p'] = self.p
                output["shard_"+str(m)]['model'] = self.model
                output["shard_"+str(m)]['shards']=self.shards
                
                #L =list(output["shard_"+str(m)]['X'].keys())
                #L1=L[len(L)-1].split(":")
                #max_val=int(L1[len(L1)-1])
                #output["shard_"+str(m)]['batch_number']=max_val+1
            
            for ep in range(self.epoch_number):
                epoch_output["epoch"+str(ep)]['parallel_shards'] = self.shards
                epoch_output["epoch"+str(ep)]['b']               = self.b
                epoch_output["epoch"+str(ep)]['B'] = self.B#*self.shards

                for m in range(self.shards):
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['N'] = self.N
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['b'] = self.b
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['B'] = self.B#*self.shards
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['model'] = self.model
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['shards']=self.shards
                    #keep_rows_index = np.sum(test['epoch_data']['epoch0']['shard_0']['X_matrix'], axis=1)!=0
                    #test['epoch_data']['epoch0']['shard_0']['X_matrix'][keep_rows_index,:].shape
                    keep_rows_index = np.sum(epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'], axis=1)!=0
                    
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['predictors'] = self.predictor_names
                    print("'CRSDepTime' in self.predictor_names = ", 'CRSDepTime' in self.predictor_names)
                    if 'CRSDepTime' in self.predictor_names:
                        
                        #print("cols_to_keep: ", cols_to_keep)
                        epoch_output["epoch"+str(ep)]["shard_"+str(m)]['CRSDepTime'] = (
                            epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'][keep_rows_index,-1]
                        )
                        
                        epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'] = (
                            epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'][keep_rows_index,:-1]
                        )
                        
                        epoch_output["epoch"+str(ep)]["shard_"+str(m)]['p'] = self.p #- 1
                        print("making data.  X has shape: ", epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'].shape)

                    else:
                        epoch_output["epoch"+str(ep)]["shard_"+str(m)]['p'] = self.p
                        epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'] = (
                            epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X_matrix'][keep_rows_index,:]
                        )
            self.output=output
            self.output['epoch_data']=epoch_output
            #self.output['X']=self.X
            self.output['X_matrix']=self.X_matrix
            self.output['Y']=self.Y
            self.output['N']=self.N
            self.output['b']=self.b
            self.output['B']=self.B
            self.output['p']=self.p
            self.output['b_oos']=self.b_oos
            self.output['X_oos']=self.X_oos
            self.output['batch_number']=self.N_batch
            self.output['model']=self.model
            self.output['shards']=1
            self.output['parallel_shards']=self.shards
            self.output['data_keys']=self.data_keys
            self.output['predictors']=self.predictor_names
            
            if 'CRSDepTime' in self.predictor_names:
                print("self.predictor_names=", self.predictor_names)
                self.predictor_names.remove('CRSDepTime')
                self.output['predictors']=self.predictor_names
                print("self.predictor_names=", self.predictor_names)
                #self.output['p']=self.p #- 1
            
            print("self.output['predictors']=", self.output['predictors'])
                
    def get_data(self):
        return(self.output)
    
    def format_for_scatter(self, epoch):
        output = []
        for m in range(self.shards):
            output.append(self.output['epoch_data']["epoch"+str(epoch)]["shard_"+str(m)])
        return(output)