import analysis_functions as af
import pandas as pd
import numpy as np
import re
import pickle

# format = sn_en_nt_p_gpv_
shard_number = [3,6, 10,15,30]
Xy_N=6000
N_Epoch = [30,150,300,600,1200]
Nt = [30]
p = [2, 8, 16, 32]
GP_version = list(range(10))
part_num = [20, 100, 1000]
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

print("number of entries in prep_big_results_dict = ", len(big_results_dict))
keys = big_results_dict.keys()
for k in keys:
    print(k)
    if len(big_results_dict[k].wi_comm_list)==0:continue
    print(big_results_dict[k].last_avg_lik_diff +  big_results_dict[k].last_std_err_lik_diff)
    print(big_results_dict[k].last_avg_lik_diff )
    print(big_results_dict[k].last_avg_lik_diff -  big_results_dict[k].last_std_err_lik_diff)
    print("")

###########################
#          p=2
###########################

Xy_N_6000_hm_plot_data_2_x_20_1000_x_150_1200_x_3 = af.heat_map_data_prep_mean(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_2_x_20_1000_x_150_1200_x_3.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_2_x_20_1000_x_150_1200_x_3.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_2_x_20_1000_x_150_1200_x_6 = af.heat_map_data_prep_mean(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=6, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_2_x_20_1000_x_150_1200_x_6.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_2_x_20_1000_x_150_1200_x_6.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_2_x_20_1000_x_150_1200_x_10 = af.heat_map_data_prep_mean(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=10, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_2_x_20_1000_x_150_1200_x_10.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_2_x_20_1000_x_150_1200_x_10.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_2_x_20_1000_x_150_1200_x_15 = af.heat_map_data_prep_mean(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=15, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_2_x_20_1000_x_150_1200_x_15.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_2_x_20_1000_x_150_1200_x_15.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_2_x_20_1000_x_150_1200_x_30 = af.heat_map_data_prep_mean(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=30, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_2_x_20_1000_x_150_1200_x_30.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_2_x_20_1000_x_150_1200_x_30.csv',
    index = False,
)

###########################
#          p=8
###########################

Xy_N_6000_hm_plot_data_8_x_20_1000_x_150_1200_x_3 = af.heat_map_data_prep_mean(
    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_8_x_20_1000_x_150_1200_x_3.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_8_x_20_1000_x_150_1200_x_3.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_8_x_20_1000_x_150_1200_x_6 = af.heat_map_data_prep_mean(
    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=6, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_8_x_20_1000_x_150_1200_x_6.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_8_x_20_1000_x_150_1200_x_6.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_8_x_20_1000_x_150_1200_x_10 = af.heat_map_data_prep_mean(
    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=10, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_8_x_20_1000_x_150_1200_x_10.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_8_x_20_1000_x_150_1200_x_10.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_8_x_20_1000_x_150_1200_x_15 = af.heat_map_data_prep_mean(
    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=15, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_8_x_20_1000_x_150_1200_x_15.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_8_x_20_1000_x_150_1200_x_15.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_8_x_20_1000_x_150_1200_x_30 = af.heat_map_data_prep_mean(
    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=30, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_8_x_20_1000_x_150_1200_x_30.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_8_x_20_1000_x_150_1200_x_30.csv',
    index = False,
)

###########################
#          p=16
###########################

Xy_N_6000_hm_plot_data_16_x_20_1000_x_150_1200_x_3 = af.heat_map_data_prep_mean(
    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_16_x_20_1000_x_150_1200_x_3.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_16_x_20_1000_x_150_1200_x_3.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_16_x_20_1000_x_150_1200_x_6 = af.heat_map_data_prep_mean(
    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=6, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_16_x_20_1000_x_150_1200_x_6.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_16_x_20_1000_x_150_1200_x_6.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_16_x_20_1000_x_150_1200_x_10 = af.heat_map_data_prep_mean(
    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=10, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_16_x_20_1000_x_150_1200_x_10.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_16_x_20_1000_x_150_1200_x_10.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_16_x_20_1000_x_150_1200_x_15 = af.heat_map_data_prep_mean(
    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=15, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_16_x_20_1000_x_150_1200_x_15.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_16_x_20_1000_x_150_1200_x_15.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_16_x_20_1000_x_150_1200_x_30 = af.heat_map_data_prep_mean(
    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=30, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_16_x_20_1000_x_150_1200_x_30.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_16_x_20_1000_x_150_1200_x_30.csv',
    index = False,
)

###########################
#          p=32
###########################

Xy_N_6000_hm_plot_data_32_x_20_1000_x_150_1200_x_3 = af.heat_map_data_prep_mean(
    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_32_x_20_1000_x_150_1200_x_3.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_32_x_20_1000_x_150_1200_x_3.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_32_x_20_1000_x_150_1200_x_6 = af.heat_map_data_prep_mean(
    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=6, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_32_x_20_1000_x_150_1200_x_6.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_32_x_20_1000_x_150_1200_x_6.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_32_x_20_1000_x_150_1200_x_10 = af.heat_map_data_prep_mean(
    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=10, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_32_x_20_1000_x_150_1200_x_10.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_32_x_20_1000_x_150_1200_x_10.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_32_x_20_1000_x_150_1200_x_15 = af.heat_map_data_prep_mean(
    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=15, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_32_x_20_1000_x_150_1200_x_15.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_32_x_20_1000_x_150_1200_x_15.csv',
    index = False,
)

######

Xy_N_6000_hm_plot_data_32_x_20_1000_x_150_1200_x_30 = af.heat_map_data_prep_mean(
    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=30, big_results_dict=big_results_dict
)
Xy_N_6000_hm_plot_data_32_x_20_1000_x_150_1200_x_30.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_mean_plot_data_32_x_20_1000_x_150_1200_x_30.csv',
    index = False,
)

######
