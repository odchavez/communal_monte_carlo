import embarrassingly_parallel as ep


#import numpy as np
import particle_filter
#import simulate_data
import communal_monte_carlo as cmc
import embarrassingly_parallel
import pf_plots 

class communal_monte_carlo:
        
        def __init__(self, 
                     data_arg, 
                     params_arg, 
                     shuffle_method_arg=None,
                     wass_n=None
                    ):
            self.params         = params_arg
            self.data           = data_arg
            self.shuffle_method = shuffle_method_arg
            self.shuffle_size   = wass_n
            
        def run_embarassingly_parallel(self):
            cmcobj = ep.embarrassingly_parallel(self.data['epoch_data']['epoch0'], self.params)
            print('in communal_monte_carlo.py, self.cmcobj = ', cmcobj)
            cmcobj.shuffel_embarrassingly_parallel_particles(self.shuffle_method, self.shuffle_size)
            
            for ea in range(1, len(self.params['epoch_at'])):
                print(ea)
                cmcobj.run_batch(self.data['epoch_data']['epoch'+str(ea)])
                cmcobj.shuffel_embarrassingly_parallel_particles(self.shuffle_method, self.shuffle_size)
        
            return(cmcobj)