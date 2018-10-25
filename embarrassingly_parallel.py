from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import particle_filter
import pf_models as pfm
from random import randint
from scipy.optimize import linprog

class embarrassingly_parallel:
    def __init__(self, data, params):
        print("parallel class...")
        
        self.PART_NUM=params['particles_per_shard']
        self.model=params['model']
        self.sample_method= params['sample_method']
        self.number_of_shards=params['shards']
        self.pf_obj=list()
        self.params=params
        self.data=data
        for m in range(self.number_of_shards):
            print("fitting shard ", m)
            pfo = particle_filter.particle_filter(self.data['shard_'+str(m)], self.PART_NUM, self.model,self.sample_method)
            self.pf_obj.append(pfo)
            self.pf_obj[m].run_particle_filter()
    
    def run_batch(self, data):
        for m in range(self.number_of_shards):
            self.pf_obj[m].update_data(data['shard_'+str(m)])
            self.pf_obj[m].run_particle_filter()
    
    #def update_parallel_pf_data(self, data):
    #    for m in range(self.number_of_shards):
    #        self.pf_obj[m].update_data(self.data['shard_'+str(m)])
            
    def plot_parameter_path(self, particle_prop=0.01):
        print("plot_parameter_path...")
        
        param_num=self.pf_obj[0].get_particle(0).bo_list.shape[1]
        print("param_num=",param_num)
        total_time_steps = len(self.pf_obj[0].get_particle(0).bo_list[:,0])
        print("total_time_steps=",total_time_steps)
        params=list()
        particle_indices = np.random.choice(self.PART_NUM, max(int(self.PART_NUM*particle_prop), 1))
        print("particle_indices=",particle_indices)
        
        for os in range(param_num):
          temp_all_parts = np.zeros((len(particle_indices), total_time_steps))
          #print("data keys = ", self.data.keys())
          for sn in range(self.data['parallel_shards']):
            for pn in range(len(particle_indices)):
              particle=self.pf_obj[sn].get_particle(particle_indices[pn])
              p_temp = particle.bo_list[:,os].copy()
              p_temp[np.isnan(p_temp)]=0
              temp_all_parts[pn,:]=np.add(temp_all_parts[pn,:],p_temp)
                
          #for ts in range(len(self.params['epoch_at'])):
          #  ts_values=self.params['epoch_at'][ts]
          #  temp_all_parts[:,ts_values]=temp_all_parts[:,ts_values]/self.data['parallel_shards']      
          params.append(temp_all_parts)
        
        for par_n in range(param_num):
            avg_param_0=np.mean(params[par_n], axis=0)
            std_parma_0=np.std(params[par_n], axis=0)
            above=np.add(avg_param_0,std_parma_0*2)
            below=np.add(avg_param_0,-std_parma_0*2)
            
            truth=self.data['b'][:,par_n]#data['shard_0']['b'][:,par_n]
            
            x = np.arange(len(avg_param_0)) # the points on the x axis for plotting
            
            fig, ax1 = plt.subplots()
            plt.plot(x,truth,'black')
            ax1.fill_between(x, below, above, facecolor='green',  alpha=0.3)
            plt.plot(x,avg_param_0, 'b', alpha=.8)
            for line_tick in self.params['epoch_at']:
                plt.axvline(x=line_tick, color='r', alpha=0.25)
            min_tic=np.min([np.min(below),np.min(truth)])
            max_tic=np.max([np.max(above),np.max(truth)])
            plt.yticks(np.linspace(start=min_tic, stop=max_tic, num=12))
            plt.grid(True)
            plt.show()

    def plot_CMC_parameter_path(self, particle_prop=0.01):
        print("plot_parameter_path...")
        
        param_num=self.pf_obj[0].get_particle(0).bo_list.shape[1]
        total_time_steps = len(self.pf_obj[0].get_particle(0).bo_list[:,0])
        params=list()
        particle_indices = np.random.choice(self.PART_NUM, max(int(self.PART_NUM*particle_prop), 1))
        
        #for os in range(param_num):
        #    temp_all_parts = np.zeros((len(particle_indices), total_time_steps))
        #    #print("data keys = ", self.data['data_keys'])
        #    for sn in range(self.data['parallel_shards']):
        #        for pn in range(len(particle_indices)):
        #            particle=self.pf_obj[sn].get_particle(particle_indices[pn])
        #            p_temp = particle.bo_list[:,os].copy()
        #            p_temp[np.isnan(p_temp)]=0
        #            temp_all_parts[pn,:]=np.add(temp_all_parts[pn,:],p_temp)
        #    
        #    #print("temp_all_parts=",temp_all_parts)
        #    for ts in range(len(self.params['epoch_at'])):
        #        ts_values=self.params['epoch_at'][ts]
        #        temp_all_parts[:,ts_values] = temp_all_parts[:,ts_values]/self.data['parallel_shards']      
        #    params.append(temp_all_parts)
        Z_dim=self.number_of_shards*self.PART_NUM
        temp_all_parts = np.zeros((total_time_steps, param_num, Z_dim))
        temp_all_parts[temp_all_parts==0]=np.NaN

        counter=0
        for sn in range(self.data['parallel_shards']):
            for pn in range(len(particle_indices)):
                particle=self.pf_obj[sn].get_particle(particle_indices[pn])
                #print("particle.bo_list.copy().shape=", particle.bo_list.copy().shape)
                temp_all_parts[:,:,counter] = particle.bo_list.copy()
                counter+=1
                #p_temp[np.isnan(p_temp)]=0
                #temp_all_parts[pn,:]=np.add(temp_all_parts[pn,:],p_temp)
        #print("temp_all_parts.shape=",temp_all_parts.shape)
        #print("temp_all_parts.head()=",temp_all_parts.head())
        params=np.nanmean(temp_all_parts, axis=2)
        params_std=np.nanstd(temp_all_parts, axis=2)
        #print("params.shape=",params.shape)
        #print('parmas=', params)
        for par_n in range(param_num):
            avg_param_0=params[:,par_n]#np.mean(params[par_n], axis=0)
            std_parma_0=params_std[:,par_n]#np.std(params[par_n], axis=0)
            above=np.add(avg_param_0,std_parma_0*2)
            below=np.add(avg_param_0,-std_parma_0*2)
            
            truth=self.data['b'][:,par_n]#data['shard_0']['b'][:,par_n]
            
            x = np.arange(len(avg_param_0)) # the points on the x axis for plotting
            
            fig, ax1 = plt.subplots()
            plt.plot(x,truth,'black')
            ax1.fill_between(x, below, above, facecolor='green',  alpha=0.3)
            plt.plot(x,avg_param_0, 'b', alpha=.8)
            for line_tick in self.params['epoch_at']:
                plt.axvline(x=line_tick, color='r', alpha=0.25)
            min_tic=np.min([np.min(below),np.min(truth)])
            max_tic=np.max([np.max(above),np.max(truth)])
            plt.yticks(np.linspace(start=min_tic, stop=max_tic, num=12))
            plt.grid(True)
            plt.show()

    def plot_CMC_parameter_path_by_shard(self, particle_prop=0.01):
        print("plot_parameter_path...")
        color_list=['r','g','b','k']
        param_num=self.pf_obj[0].get_particle(0).bo_machine_list.shape[1]
        total_time_steps = len(self.pf_obj[0].get_particle(0).bo_machine_list[:,0])
        particle_indices = np.random.choice(self.PART_NUM, max(int(self.PART_NUM*particle_prop), 1))
        for par_n in range(param_num):
            truth=self.data['b'][:,par_n]#data['shard_0']['b'][:,par_n]
            x = np.arange(len(self.data['b'][:,par_n])) # the points on the x axis for plotting
            plt.plot(x,truth,'black')
            fig, ax = plt.subplots(1,2,3,4)

            for sn in range(self.data['parallel_shards']):
                params=list()
                Z_dim=self.number_of_shards*self.PART_NUM
                temp_all_parts = np.zeros((total_time_steps, param_num, Z_dim))
                temp_all_parts[temp_all_parts==0]=np.NaN
                counter=0
                for pn in range(len(particle_indices)):
                    particle=self.pf_obj[sn].get_particle(particle_indices[pn])
                    #print("particle.bo_list.copy().shape=", particle.bo_list.copy().shape)
                    temp_all_parts[:,:,counter] = particle.bo_machine_list.copy()
                    counter+=1

                params=np.nanmean(temp_all_parts, axis=2)
                params_std=np.nanstd(temp_all_parts, axis=2)
                #print("particle.bo_list.copy()=",pd.DataFrame(particle.bo_list.copy()).head(20))
                #print(pd.DataFrame(params))
                #for par_n in range(param_num):
                avg_param_0=avg_param_0=params[:,par_n]#np.mean(params[par_n], axis=0)
                avg_param_0=pd.Series(avg_param_0).fillna(method='ffill')
                avg_param_0=pd.Series(avg_param_0).fillna(method='bfill')

                std_parma_0=params_std[:,par_n]#np.std(params[par_n], axis=0)
                above=np.add(avg_param_0,std_parma_0*2)
                below=np.add(avg_param_0,-std_parma_0*2)
                #
                #
                x = np.arange(len(avg_param_0)) # the points on the x axis for plotting
                #
                #fig, ax1 = plt.subplots()
                ax[sn].fill_between(x, below, above, facecolor=color_list[sn],  alpha=0.3)
                plt.plot(x,avg_param_0, color_list[sn], '.', alpha=.8)
            #for line_tick in self.params['epoch_at']:
            #    plt.axvline(x=line_tick, color='r', alpha=0.25)
            #min_tic=np.min([np.min(below),np.min(truth)])
            #max_tic=np.max([np.max(above),np.max(truth)])
            #plt.yticks(np.linspace(start=min_tic, stop=max_tic, num=12))
            #plt.grid(True)
                    
            plt.show()
                
            
    def shuffel_embarrassingly_parallel_particles(self, machine_list=None, method=None, wass_n=None):
                
        self.all_particles=list()
        #print("self.data.keys():",self.data.keys())
        for m in range(self.data['parallel_shards']):
            for p in range(self.PART_NUM):
                #all_particles.append()
                #print(self.data.keys())
                temp_particle = pfm.probit_sin_wave_particle( np.array(self.data['b'][0]), self.data['B'], -1)
            #pfo = particle_filter.particle_filter(self.data['shard_'+str(m)], self.PART_NUM, self.model,self.sample_method)
            #self.pf_obj.append(pfo)
                temp_particle.copy_particle_values(self.pf_obj[m].particle_list[p])
                self.all_particles.append(temp_particle)
                #self.all_particles.append(self.pf_obj[m].particle_list[p]) #run_particle_filter()
        if method == "wasserstein":
            print("computing waserstein barrycenter...")
            print("collecting parameter info from shards...")
            f_wts = 0.01+self.get_approx_shard_wasserstein_barycenter(wass_n)[0]
            print("data successfully prepared...")  
            print('machine weights: ', f_wts)
            temp_all_wts=np.repeat(f_wts,self.PART_NUM)
            all_wts = temp_all_wts/np.sum(temp_all_wts)
            index_vals = np.random.choice(range(len(all_wts)), len(all_wts), p=all_wts)
            count=0
            for m in range(self.data['parallel_shards']):
                for p in range(self.PART_NUM):
                    self.pf_obj[m].particle_list[p].copy_particle_values( self.all_particles[index_vals[count]] )
                    count+=1
        else:        
            for m in range(self.data['parallel_shards']):
                for p in range(self.PART_NUM):
                    index=randint(0, len(self.all_particles)-1)
                    #print(self.all_particles[index].print_vals())
                    self.pf_obj[m].particle_list[p].copy_particle_values( self.all_particles[index] )
                                    
    def update_data(self, data_from_outside):
        self.data = data_from_outside
        
    def prep_particles_for_wassber(self, n):
    #if True:
        #print("running the code to make theta")
        #n=PART_NUM
        K=self.number_of_shards
        
        myTheta=list()
        theta = np.zeros((n*K,len(self.params['omega_shift'])))
        row=0
        for m in range(self.number_of_shards):
            myTheta.append(np.zeros((n,len(self.params['omega_shift']))))
            for p in range(n):
                particle_index = randint(0, self.PART_NUM-1)
                theta[row,:] = myTheta[m][p,:] = self.pf_obj[m].particle_list[particle_index].bo
                row+=1
        return theta, myTheta
    
    def get_wasserstein_barycenter(self, n, K, d, theta, myTheta):
        
        Nk1 = np.ones((1,n))    # one vector of length n
        N1  = np.ones((K*n,1))  # one vector of length K*n
        IN  = np.identity(K*n)  # Diagonal matrix of size K*n
        INk = np.identity(n)    # Diagonal matrix of size n
        
        # cost vector
        # each thetak is the matrix of samples from subset posterior k=1,...,K
        # theta is the overall sample matrix, formed by stacking the thetak
        #cost = c()
        cost = np.array([])
        #for (i in 1:K){
        for i in range(K):
            thetak = myTheta[i]
            theta_theta_transpose = np.matmul(theta, np.transpose(theta))
            diag_of_theta_theta_transpose=np.diag(theta_theta_transpose)
            diag_of_theta_theta_transpose=diag_of_theta_theta_transpose.reshape((len(diag_of_theta_theta_transpose),1))
            sec1 = np.matmul(diag_of_theta_theta_transpose, Nk1)
            
            thetak_thetak_transpose=np.matmul(thetak, np.transpose(thetak))
            tdiag_thetak_thetak_transpose=np.transpose(np.diag(thetak_thetak_transpose))
            tdiag_thetak_thetak_transpose=tdiag_thetak_thetak_transpose.reshape((1,len(tdiag_thetak_thetak_transpose)))
            sec2 = np.matmul(N1, tdiag_thetak_thetak_transpose)
            
            theta_thetak_transpose = np.matmul(theta, np.transpose(thetak))
            sec3=-2.0*theta_thetak_transpose
            Mk = sec1 + sec2 + sec3
    
            cost = np.concatenate((cost, np.transpose(Mk).reshape(1,Mk.size)[0]))
    
        cost = np.concatenate((cost, np.zeros(K*n)))
        
        # constraint matrix 
        # A1-A6 are the 6 components of the A constraint matrix in the first Srivastava paper.
        A1 = np.zeros((1,(K*n)**2)) 
        A2 = np.ones((1, K*n))
        a3=np.kron(Nk1, IN)
        A3 = np.kron(np.identity(K), a3)
        A4 = np.kron(np.ones((K,1)), IN)
        a5 = np.transpose(np.kron(INk, N1))
        A5 = np.transpose(np.kron(np.identity(K), np.transpose(a5)))
        A6 = np.zeros((K*n, K*n))
        cbA1A2  = np.concatenate((A1, A2), axis=1)
        cbA3mA4 = np.concatenate((A3, -A4), axis=1)
        cbA5A6  = np.concatenate((A5, A6), axis=1)
        A = np.concatenate((cbA1A2, cbA3mA4, cbA5A6)) 
    
        # the right hand side of constraints for the matrix A
        consRHS = np.concatenate((np.ones(1), 
                                  np.zeros((K**2)*n), 
                                  1.0/n*np.ones(K*n)))
        
        # direction of the constraints.
        
        # note the constraint that the output vector 'a' of probabilities is >=0 is not explicitly put into the solver
        # since this solver already inputs this constraint, other solver packages MAY REQUIRE THIS EXPLICITLY
        
        out = linprog(cost, A_eq=A, b_eq=consRHS, options={"disp": False})
        # this just extracts the relevant output containing the posterior vector of probabilities 'a' for the consensus posterior
        end = len(out['x'])
        start = len(out['x'])-K*n
        sol = out['x'][start:end]
        
        return sol
    
    def get_approx_shard_wasserstein_barycenter(self, f_n):
        wsbc_num       = 20
        wssber_wts     = {}
        theta_list     = list()
        myTheta_list   = list()
        apx_bc_mac_wts = list()
        f_K = self.number_of_shards
        f_d = len(self.params['omega_shift'])
        for i in range(wsbc_num):
            theta, myTheta = self.prep_particles_for_wassber(f_n)
            wssber_wts[i] = self.get_wasserstein_barycenter(f_n, f_K, f_d, theta, myTheta)
            temp_wts=np.sum(np.array_split(wssber_wts[i], f_K), axis=1)
            apx_bc_mac_wts.append(temp_wts)
            theta_list.append(theta)
            myTheta_list.append(myTheta)
        
        f_shard_wts_0 = np.array(apx_bc_mac_wts).reshape((wsbc_num,f_K))
        f_shard_wts   = np.mean(f_shard_wts_0, axis=0)
        return f_shard_wts, apx_bc_mac_wts, theta_list, myTheta_list