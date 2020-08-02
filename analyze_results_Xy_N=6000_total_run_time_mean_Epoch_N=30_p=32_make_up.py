
import analysis_functions as af
import pandas as pd
import numpy as np
import re
import pickle

###########################
#          p=32
###########################
print(" WORKING p=32.....")

shard_number = [3]
Xy_N=6000
N_Epoch = [30]
Nt = [30]
p = [32]
GP_version = list(range(10))
part_num = [20, 100]
predictors = [
    'B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9',
    'B_10','B_11','B_12','B_13','B_14','B_15','B_16','B_17','B_18','B_19',
    'B_20','B_21','B_22','B_23','B_24','B_25','B_26','B_27','B_28','B_29',
    'B_30','B_31',
]

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

Xy_N_6000_hm_plot_data_32_x_20_x_30_x_3 = af.heat_map_data_prep_total_run_time_mean(
    pred_num=32, part_num=[20], N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)

Xy_N_6000_hm_plot_data_32_x_20_x_30_x_3.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_run_time_plot_data_32_x_20_x_30_x_3.csv',
    index = False,
)

Xy_N_6000_hm_plot_data_32_x_100_x_30_x_3 = af.heat_map_data_prep_total_run_time_mean(
    pred_num=32, part_num=[100], N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)

Xy_N_6000_hm_plot_data_32_x_100_x_30_x_3.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_run_time_plot_data_32_x_100_x_30_x_3.csv',
    index = False,
)