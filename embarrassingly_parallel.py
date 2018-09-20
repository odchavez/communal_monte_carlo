from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import particle_filter

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
        total_time_steps = len(self.pf_obj[0].get_particle(0).bo_list[:,0])
        params=list()
        particle_indices = np.random.choice(self.PART_NUM, max(int(self.PART_NUM*particle_prop), 1))
        
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

    def plot_CMC_parameter_path(self, particle_prop=0.01):
        print("plot_parameter_path...")
        
        param_num=self.pf_obj[0].get_particle(0).bo_list.shape[1]
        total_time_steps = len(self.pf_obj[0].get_particle(0).bo_list[:,0])
        params=list()
        particle_indices = np.random.choice(self.PART_NUM, max(int(self.PART_NUM*particle_prop), 1))
        
        for os in range(param_num):
          temp_all_parts = np.zeros((len(particle_indices), total_time_steps))
          print("data keys = ", self.data.keys())
          for sn in range(self.data['parallel_shards']):
            for pn in range(len(particle_indices)):
              particle=self.pf_obj[sn].get_particle(particle_indices[pn])
              p_temp = particle.bo_list[:,os].copy()
              p_temp[np.isnan(p_temp)]=0
              temp_all_parts[pn,:]=np.add(temp_all_parts[pn,:],p_temp)
            
            #print("temp_all_parts=",temp_all_parts)
          for ts in range(len(self.params['epoch_at'])):
            ts_values=self.params['epoch_at'][ts]
            temp_all_parts[:,ts_values]=temp_all_parts[:,ts_values]/self.data['parallel_shards']      
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
    
    #def plot_CMC_parameter_path_by_machine(self):
        #   
        #for sn in range(len(pf_obj)):
        #    pf_obj[sn].plot_particle_path(1)
        #plt.show()
            
    def shuffel_embarrassingly_parallel_particles(self, machine_list=None, method='uniform'):
        
        self.all_particles=list()
        #print("self.data.keys():",self.data.keys())
        for m in range(self.data['parallel_shards']):
            for p in range(self.PART_NUM):
                #all_particles.append()
            #pfo = particle_filter.particle_filter(self.data['shard_'+str(m)], self.PART_NUM, self.model,self.sample_method)
            #self.pf_obj.append(pfo)
                self.all_particles.append(self.pf_obj[m].particle_list[p]) #run_particle_filter()
                
        index=0
        for m in range(self.data['parallel_shards']):
            for p in range(self.PART_NUM):
                self.pf_obj[m].particle_list[p].copy_particle_values( self.all_particles[index] )
                index+=1
                
    def update_data(self, data_from_outside):
        self.data = data_from_outside