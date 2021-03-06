class pf_params():

    def __init__(self, shards_number):
        M=shards_number
        PART_NUM = 20

        self.params={
            'N_batch':1, 
            'shards': M,
            'particles_per_shard':PART_NUM,
            'model':'probit_sin_wave',
            'sample_method':"importance",
            'data_type': 'airline'
        }
        
    def get_params(self):
        return(self.params)
    
    def get_N_batch(self):
        return
    
    def get_shards(self):
        return self.params['shards']
    
    def get_particles_per_shard(self):
        return self.params['particles_per_shard']
    
    def get_model(self):
        return self.params['model']
    
    def get_sample_method(self):
        return self.params['sample_method']
    
    def get_data_type(self):
        return self.params['data_type']
    
class pf_params_synth_data():

    def __init__(self, shards_number, args):
        #args.particles_per_shard, args.p_to_use, args.randomize_shards)
        M=shards_number
        PART_NUM = 1000
        COMM_PER_SHARD_INTERVAL = 999999999#number of observations to process per shard before communication step
        self.params={
            'N_batch'            : 1, 
            'shards'             : M,
            'p_to_use'           : args.p_to_use,
            'particles_per_shard': args.particles_per_shard,
            's_num_before_comm'  : COMM_PER_SHARD_INTERVAL,
            'randomize_shards'   : args.randomize_shards,
            'model'              : 'probit_sin_wave',
            'sample_method'      : "importance",
            'data_type'          : 'simulated',
            'source_folder'      : args.source_folder
            
        }
        
    def get_params(self):
        return(self.params)
    
    def get_N_batch(self):
        return
    
    def get_shards(self):
        return self.params['shards']
    
    def get_particles_per_shard(self):
        return self.params['particles_per_shard']
    
    def get_model(self):
        return self.params['model']
    
    def get_sample_method(self):
        return self.params['sample_method']
    
    def get_data_type(self):
        return self.params['data_type']