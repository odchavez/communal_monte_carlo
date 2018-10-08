import numpy as np
from matplotlib import pyplot as plt

class simulated_data:
    def __init__(self, params, model, show=True):
        self.model=model
        if model=="probit":
            params={'N': 10000, 'b': [-2,-1,0,1,2], 'B':np.identity(5)}
            self.p = len(params['b'])
            self.N = params['N']
            self.X = np.random.uniform(-1,1,self.p*self.N).reshape((self.N,self.p))
            self.Y = np.zeros(self.N)
            self.b=params['b']
            self.B=params['B']
        
            for i in range(self.N):
                inner=self.X[i].dot(self.b)
                samp=np.random.normal(loc=inner, scale=1.0, size=1)[0]
                if samp>=0:
                    self.Y[i]=1
            self.output={'X':self.X, 'Y':self.Y, 'N': self.N, 'b': self.b, 'B':self.B}
                
        if model== "probit_sin_wave":
            
            omega_shift=params['omega_shift']
            temp_params={'p':len(omega_shift), 'b': omega_shift, 'B':np.identity(len(omega_shift))}
            #print("params=", params)
            #print("temp_params=", temp_params)
            params.update(temp_params)#dict(params.items() + temp_params.items())
            #print("params=", params)
            self.p=params['p']
            self.N=params['N']
            self.N_batch = params['N_batch']
            self.shards=params['shards']
            self.epoch_at=params['epoch_at']
            self.epoch_number=len(self.epoch_at)+1
            self.X={}
            self.Y={}
            self.data_keys=list()
            f=1.5
            x = np.arange(self.N) # the points on the x axis for plotting
            self.b=np.zeros((self.N,self.p))
            self.b_oos=np.zeros((1,self.p))
            
            output = {}
            for m in range(self.shards):
                output["shard_"+str(m)]     ={}
                output["shard_"+str(m)]['X']={}
                output["shard_"+str(m)]['Y']={}
                output["shard_"+str(m)]['data_keys']=list()
            
            epoch_output = {}
            for ep in range(self.epoch_number):
                epoch_output["epoch"+str(ep)]={}
                for m in range(self.shards):
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]     ={}
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X']={}
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['Y']={}
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['data_keys']=list()

            #for ep in range(self.epoch_number):
            #    for m in range(self.shards):
            #        epoch_output["epoch"+str(ep)]["shard_"+str(m)]     ={}
            #        epoch_output["epoch"+str(ep)]["shard_"+str(m)]['X']={}
            #        epoch_output["epoch"+str(ep)]["shard_"+str(m)]['Y']={}
            #print('output=',output)
            #for m in range(self.shards):
            #    self.X["shard_"+str(m)]={}
            #    self.Y["shard_"+str(m)]={}
            
            for os in range(self.p):
                self.b[:,os] = [ 1.0*np.sin(np.pi*f * (i/(self.N*1.0)) + omega_shift[os]) for i in x]
                #% matplotlib inline
                # showing the exact location of the smaples
                if show:
                    plt.stem(x,self.b[:,os], 'r', )
                    plt.plot(x,self.b[:,os])
                    plt.yticks(np.linspace(start=-2, stop=2, num=9))
                    plt.grid(True)
                    plt.show()
            
            for os in range(self.p):
                self.b_oos[:,os] =  [ 1.0*np.sin(np.pi*f * (i/(self.N*1.0)) + omega_shift[os]) for i in [max(x)+1]]
                
            data_index=0
            epoch_counter=0
            for i in range(self.N):
                #key=str(i)+":"+str(i%self.shards)+":"+str(data_index)
                key=str(i)+":"+str(data_index)
                all_key=str(i)+":"+str(i)
                #print('key=',key)
                self.data_keys.append(key)
                s="shard_"+str(i%self.shards)
                #print('s=',s)
                #print('output[s]=', output[s])
                temp_X = np.random.uniform(-1,1,self.p*self.N_batch).reshape((self.N_batch,self.p))
                self.X[all_key]     = temp_X
                self.Y[all_key]     = np.zeros(self.N_batch)
                
                output[s]['X'][key] = temp_X#.copy()
                output[s]['Y'][key] = np.zeros(self.N_batch)
                output[s]['data_keys'].append(key)
                

                inner=temp_X.dot(self.b[i,])
                #print('innter=',inner)
                samp=np.random.normal(loc=inner, scale=1.0)
                #print("samp=",samp)
                for j in range(self.N_batch):
                    if samp[j]>=0:
                        output[s]['Y'][key][j]=self.Y[all_key][j]=1
                        #if i not in self.epoch_at:
                        #    #epoch_output["epoch"+str(epoch_counter)]["shard_"+str(s)]['Y'][key][j]=1
                        #    output[s]['Y'][key][j]=1
                        #    self.Y[all_key][j]=1
                        #else:
                        #    output[s]['Y'][key][j]=1
                        #    self.Y[all_key][j]=1
                
                if i not in self.epoch_at:
                    #output[s]['X'][key] = temp_X.copy()
                    #output[s]['Y'][key] = np.zeros(self.N_batch)
                    #for m in range(self.shards):
                    epoch_output["epoch"+str(epoch_counter)][s]['X'][key]=temp_X
                    epoch_output["epoch"+str(epoch_counter)][s]['Y'][key]=self.Y[all_key] #np.zeros(self.N_batch)
                    epoch_output["epoch"+str(epoch_counter)][s]['data_keys'].append(key)
                else:# i in self.epoch_at:
                    
                    #print("xkind=",xkind)
                    #print("current_n=", current_n)
                    for m in range(self.shards):
                        #print("x_keys[xkind]=",x_keys[xkind])
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['X'][all_key] = self.X[all_key]
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['Y'][all_key] = self.Y[all_key]
                        epoch_output["epoch"+str(epoch_counter)]["shard_"+str(m)]['data_keys'].append(all_key)
                    epoch_counter+=1
                
                if i%self.shards == self.shards-1:
                    data_index+=1
            tttemp=np.max([100,self.N_batch])
            self.X_oos=np.random.uniform(-1,1,self.p*tttemp).reshape((tttemp,self.p))
            
            sig=np.max(np.var(self.b[0:(self.N-1),:]-self.b[1:,:], axis=0))
            print("sig=", sig)
            self.B=30*sig*params['B']#self.shards
            
            for m in range(self.shards):
                output["shard_"+str(m)]['N'] = self.N
                output["shard_"+str(m)]['b'] = self.b
                output["shard_"+str(m)]['B'] = self.B*self.shards
                output["shard_"+str(m)]['p'] = self.p
                output["shard_"+str(m)]['model'] = self.model
                output["shard_"+str(m)]['shards']=self.shards
                
                L =list(output["shard_"+str(m)]['X'].keys())
                L1=L[len(L)-1].split(":")
                max_val=int(L1[len(L1)-1])
                output["shard_"+str(m)]['batch_number']=max_val+1
            
            for ep in range(self.epoch_number):
                epoch_output["epoch"+str(ep)]['parallel_shards'] = self.shards
                epoch_output["epoch"+str(ep)]['b']               = self.b
                epoch_output["epoch"+str(ep)]['B'] = self.B*self.shards

                for m in range(self.shards):
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['N'] = self.N
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['b'] = self.b
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['B'] = self.B*self.shards
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['p'] = self.p
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['model'] = self.model
                    epoch_output["epoch"+str(ep)]["shard_"+str(m)]['shards']=self.shards

            
            self.output=output#{, , , , , }
            self.output['epoch_data']=epoch_output
            self.output['X']=self.X
            self.output['Y']=self.Y
            self.output['N']=self.N
            self.output['b']=self.b
            self.output['B']=self.B
            self.output['p']=self.p
            self.output['b_oos']=self.b_oos
            self.output['X_oos']=self.X_oos
            self.output['batch_number']=self.N_batch
            self.output['model']=self.model
            self.output['shards']=1#self.shards
            self.output['parallel_shards']=self.shards
            self.output['data_keys']=self.data_keys
    def get_data(self):
        return(self.output)
     
def temp_make_data_function(params, model, show):
    test = simulated_data(params, model, show).get_data()
    return test, params