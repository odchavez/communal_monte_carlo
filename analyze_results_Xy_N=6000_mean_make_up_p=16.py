import analysis_functions as af
import pandas as pd
import numpy as np
import re
import pickle

# format = sn_en_nt_p_gpv_
shard_number = [10,15,30]
Xy_N=6000
N_Epoch = [30]
Nt = [30]
p = [16]
GP_version = list(range(10))
part_num = [1000]
predictors = ['B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9','B_10','B_11','B_12','B_13','B_14','B_15']

big_results_dict = af.prep_big_results_dict(
    f_shard_number = shard_number,
    f_Xy_N = Xy_N,
    f_N_Epoch = N_Epoch,
    f_Nt = Nt,
    f_p = p ,
    f_GP_version = GP_version,
    f_part_num = part_num,
    f_predictors = predictors
)


###########################
#          p=16
###########################


Xy_N_6000_hm_mean_plot_data_16_x_1000_x_30_x_10 = af.heat_map_data_prep_mean(
    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=10, big_results_dict=big_results_dict
)
Xy_N_6000_hm_mean_plot_data_16_x_1000_x_30_x_10.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_16_x_1000_x_30_x_10.csv',
    index = False,
)

######

Xy_N_6000_hm_mean_plot_data_16_x_1000_x_30_x_15 = af.heat_map_data_prep_mean(
    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=15, big_results_dict=big_results_dict
)
Xy_N_6000_hm_mean_plot_data_16_x_1000_x_30_x_15.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_16_x_1000_x_30_x_15.csv',
    index = False,
)

######

Xy_N_6000_hm_mean_plot_data_16_x_1000_x_30_x_30 = af.heat_map_data_prep_mean(
    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=30, big_results_dict=big_results_dict
)
Xy_N_6000_hm_mean_plot_data_16_x_1000_x_30_x_30.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_16_x_1000_x_30_x_30.csv',
    index = False,
)