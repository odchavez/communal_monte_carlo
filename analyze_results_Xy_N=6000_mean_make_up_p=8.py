#file name: analyze_results_Xy_N=6000_mean_p=32_or_shard=60.py
import analysis_functions as af
import pandas as pd
import numpy as np
import re
import pickle

shard_number = [60]
Xy_N=6000
N_Epoch = [30,150,300,600,1200]
Nt = [30]
p = [8] #p=2 is complete
GP_version = list(range(10))
part_num = [20, 100, 1000]
predictors = [
    'B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9',
    'B_10','B_11','B_12','B_13','B_14','B_15'
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

###########################
#          p=8
###########################
print(" WORKING p=8.....")
Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_60 = af.heat_map_data_prep_mean(
    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=60, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_60.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_8_x_20_1000_x_30_1200_x_60.csv',
    index = False,
)
