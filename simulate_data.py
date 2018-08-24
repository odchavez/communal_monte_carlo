import numpy as np
from matplotlib import pyplot as plt

class simulated_data:
    def __init__(self, params, model):
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
            self.X={}
            self.Y={}
            f=1.0
            x = np.arange(self.N) # the points on the x axis for plotting
            self.b=np.zeros((self.N,self.p))
            
            for os in range(self.p):
                self.b[:,os] = [ 1.0*np.sin(np.pi*f * (i/(self.N*1.0)) + omega_shift[os]) for i in x]
                #% matplotlib inline
                # showing the exact location of the smaples
                plt.stem(x,self.b[:,os], 'r', )
                plt.plot(x,self.b[:,os])
                plt.yticks(np.linspace(start=-2, stop=2, num=9))
                plt.grid(True)
                plt.show()
                
            for i in range(self.N):
                self.X[str(i)] = np.random.uniform(-1,1,self.p*self.N_batch).reshape((self.N_batch,self.p))
                self.Y[str(i)] = np.zeros(self.N_batch)
                inner=self.X[str(i)].dot(self.b[i,])
                #print('innter=',inner)
                samp=np.random.normal(loc=inner, scale=1.0)
                #print("samp=",samp)
                for j in range(self.N_batch):
                    if samp[j]>=0:
                        self.Y[str(i)][j]=1
                    
            sig=np.max(np.var(self.b[0:(self.N-1),:]-self.b[1:,:], axis=0))
            print("sig=", sig)
            self.B=sig*params['B']*2
            self.output={'X':self.X, 'Y':self.Y, 'N': self.N, 'b': self.b, 'B':self.B, 'model':self.model}
            
    def get_data(self):
        return(self.output)
     