from enum import Enum

class BaseEnum(Enum):

    def __str__(self):
        return str(self.value)

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


class AssignmentAgentStatus(BaseEnum):
    OPEN = 'open'
    NEW = 'new'
    SPOKE = 'spoke-to-client'
    MET_OR_SHOWING = 'met-showing'
    OFFERS_MADE = 'offers-made'
    CONTRACT = 'contract'
    CLOSED = 'closed'
    CONFIRMED_CLOSED = 'confirmed-closed'
    RELEASE = 'release'
    
class pf_params():

    def __init__(self, shards_number):
        M=shards_number
        PART_NUM = 10
        #len_mult = 200
        #epoch_num=3
        #step = (len_mult*M)/epoch_num
        #epoch_at=np.append(np.arange(step, len_mult*M-step+1, step), len_mult*M-1)

        self.params={
            #'N': full_de_mat.shape[0], 
            'N_batch':1, 
            #'omega_shift' : np.zeros(full_de_mat.shape[1]),
            #'p'=full_de_mat.shape[1],
            'shards': M,
            #'epoch_at':[full_de_mat.shape[0]],
            'particles_per_shard':PART_NUM,
            'model':'probit_sin_wave',
            'sample_method':"importance"
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