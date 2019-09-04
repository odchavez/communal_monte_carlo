from tqdm import tqdm
import particle_filter
import communal_monte_carlo as cmc
import embarrassingly_parallel as ep
import pf_plots 

class communal_monte_carlo:
        
        def __init__(self, 
                     data_arg, 
                     load_data_from_file,
                     params_arg, 
                     shuffle_method_arg=None,
                     wass_n=None
                    ):
            self.params         = params_arg
            self.shuffle_method = shuffle_method_arg
            self.shuffle_size   = wass_n
            self.load_data_from_file = load_data_from_file
            if ~load_data_from_file:
                self.data = data_arg
            else:
                self.data_paths = data_arg
                
        def run_embarassingly_parallel(self):
            
            pbar = tqdm(range(len(self.params['epoch_at'])))
            pbar.set_description("Building Model")
            
            for ea in pbar:

                if ea == 0:
                    if self.load_data_from_file: 
                        print("implement load data particle filter")
                    else:
                        cmcobj = ep.embarrassingly_parallel(self.data['epoch_data']['epoch0'], 
                                                            self.params
                                                           )
                    cmcobj.shuffel_embarrassingly_parallel_particles(self.shuffle_method, 
                                                                     self.shuffle_size
                                                                    )
                
                if ea > 0:
                    if self.load_data_from_file: 
                        print("implement load data particle filter")
                    else:
                        cmcobj.run_batch(self.data['epoch_data']['epoch'+str(ea)])
                    cmcobj.shuffel_embarrassingly_parallel_particles(self.shuffle_method, 
                                                                     self.shuffle_size
                                                                    )
                
            return cmcobj