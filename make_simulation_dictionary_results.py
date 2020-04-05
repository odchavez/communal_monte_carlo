import analysis_functions as af
import pandas as pd
import numpy as np
import pickle



shard_number = [4,10]#[4,10, 20]  #it may be that Nt must equal shard number to avoid bug
Xy_N=2000
N_Epoch = [100,200]#,500]
Nt = [10]
p = [2, 8, 16]
GP_version = list(range(10))
part_num = [20,100,1000]#,10000]
predictors = ['B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9','B_10','B_11','B_12','B_13','B_14','B_15']
                        
                        
results = af.prep_big_results_dict(
    f_shard_number = shard_number,
    f_Xy_N = Xy_N,
    f_N_Epoch = N_Epoch,
    f_Nt = Nt,
    f_p = p ,
    f_GP_version = GP_version,
    f_part_num = part_num,
    f_predictors = predictors
)

pickle_out = open("first_set_of_simulation_results_test.pk","wb")
pickle.dump(results, pickle_out)
pickle_out.close()