import analysis_functions as af
import pandas as pd
import numpy as np
import re
import pickle

###################################################################################################################################
#
#                                                 shard_num = [X] 
#
##################################################################################################################################

experiment_number = 999 # when debugging this will likely be a number different from 0
shard_number = [3]
Xy_N=6000
N_Epoch = [300,6000]
Nt = [30]
p = [2]
GP_version = list(range(10))
part_num = [20]
predictors = [
    'B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9',
    'B_10','B_11','B_12','B_13','B_14','B_15','B_16','B_17','B_18','B_19',
    'B_20','B_21','B_22','B_23','B_24','B_25','B_26','B_27','B_28','B_29',
    'B_30','B_31'
]

big_results_dict = af.prep_big_results_dict_v2(
    f_shard_number = shard_number,
    f_Xy_N = Xy_N,
    f_N_Epoch = N_Epoch,
    f_Nt = Nt,
    f_p = p ,
    f_GP_version = GP_version,
    f_part_num = part_num,
    f_predictors = predictors,  
    f_exp_num=experiment_number, 
    f_comm=True
)
######

Xy_N_6000_hm_plot_data = af.heat_map_data_prep_mean_estimate(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)
print("Likelihood")
print(Xy_N_6000_hm_plot_data)
#Xy_N_6000_hm_plot_data.to_csv('experiment_results/heat_map_data/Xy_N_6000_hm_plot_data_debug.csv',index = False,) 
#####

Xy_N_6000_hm_plot_data = af.heat_map_data_prep_std_estimate(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)
print("Likelihood std error")
print(Xy_N_6000_hm_plot_data)
#####

Xy_N_6000_hm_plot_data = af.heat_map_data_prep_particle_comm_time_mean(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)
print("Comm Time")
print(Xy_N_6000_hm_plot_data)
#####

Xy_N_6000_hm_plot_data = af.heat_map_data_prep_particle_comm_time_std(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)
print("Comm Time std")
print(Xy_N_6000_hm_plot_data)
#####

Xy_N_6000_hm_plot_data = af.heat_map_data_prep_pf_run_time_mean(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)
print("Particle Filter Run Time")
print(Xy_N_6000_hm_plot_data)
#####

Xy_N_6000_hm_plot_data = af.heat_map_data_prep_pf_run_time_std(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)
print("Particle Filter Run Time")
print(Xy_N_6000_hm_plot_data)
#####