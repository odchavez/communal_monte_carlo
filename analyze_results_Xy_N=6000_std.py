import analysis_functions as af
import pandas as pd
import numpy as np
import re
import pickle


####################################################################################################################################
##
##                                                 shard_num = [3] 
##
###################################################################################################################################
#
#shard_number = [3]
#Xy_N=6000
#N_Epoch = [30,150,300,600,1200]
#Nt = [30]
#p = [2, 8, 16, 32]
#GP_version = list(range(10))
#part_num = [20, 100, 1000]
#predictors = [
#    'B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9',
#    'B_10','B_11','B_12','B_13','B_14','B_15','B_16','B_17','B_18','B_19',
#    'B_20','B_21','B_22','B_23','B_24','B_25','B_26','B_27','B_28','B_29',
#    'B_30','B_31'
#]
#
#big_results_dict = af.prep_big_results_dict_v2(
#    f_shard_number = shard_number,
#    f_Xy_N = Xy_N,
#    f_N_Epoch = N_Epoch,
#    f_Nt = Nt,
#    f_p = p ,
#    f_GP_version = GP_version,
#    f_part_num = part_num,
#    f_predictors = predictors,  
#    f_exp_num=0, 
#    f_comm=True
#)
#
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_3 = af.heat_map_data_prep_std(
#    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_3)
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_3.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_2_x_20_1000_x_30_1200_x_3.csv',
#    index = False,
#) 
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_3 = af.heat_map_data_prep_std(
#    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_3)
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_3.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_8_x_20_1000_x_30_1200_x_3.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_3 = af.heat_map_data_prep_std(
#    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_3)
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_3.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_16_x_20_1000_x_30_1200_x_3.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_3 = af.heat_map_data_prep_std(
#    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=3, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_3)
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_3.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_32_x_20_1000_x_30_1200_x_3.csv',
#    index = False,
#)
#
#
#
####################################################################################################################################
##
##                                                 shard_num = [6] 
##
###################################################################################################################################
#
#shard_number = [6]
#Xy_N=6000
#N_Epoch = [30,150,300,600,1200]
#Nt = [30]
#p = [2, 8, 16, 32]
#GP_version = list(range(10))
#part_num = [20, 100, 1000]
#predictors = [
#    'B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9',
#    'B_10','B_11','B_12','B_13','B_14','B_15','B_16','B_17','B_18','B_19',
#    'B_20','B_21','B_22','B_23','B_24','B_25','B_26','B_27','B_28','B_29',
#    'B_30','B_31'
#]
#
#big_results_dict = af.prep_big_results_dict_v2(
#    f_shard_number = shard_number,
#    f_Xy_N = Xy_N,
#    f_N_Epoch = N_Epoch,
#    f_Nt = Nt,
#    f_p = p ,
#    f_GP_version = GP_version,
#    f_part_num = part_num,
#    f_predictors = predictors,  
#    f_exp_num=0, 
#    f_comm=True
#)
#
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_6 = af.heat_map_data_prep_std(
#    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=6, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_6)
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_6.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_2_x_20_1000_x_30_1200_x_6.csv',
#    index = False,
#) 
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_6 = af.heat_map_data_prep_std(
#    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=6, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_6)
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_6.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_8_x_20_1000_x_30_1200_x_6.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_6 = af.heat_map_data_prep_std(
#    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=6, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_6)
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_6.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_16_x_20_1000_x_30_1200_x_6.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_6 = af.heat_map_data_prep_std(
#    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=6, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_6)
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_6.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_32_x_20_1000_x_30_1200_x_6.csv',
#    index = False,
#)



####################################################################################################################################
##
##                                                 shard_num = [10] 
##
###################################################################################################################################
#
#shard_number = [10]
#Xy_N=6000
#N_Epoch = [30,150,300,600,1200]
#Nt = [30]
#p = [2, 8, 16, 32]
#GP_version = list(range(10))
#part_num = [20, 100, 1000]
#predictors = [
#    'B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9',
#    'B_10','B_11','B_12','B_13','B_14','B_15','B_16','B_17','B_18','B_19',
#    'B_20','B_21','B_22','B_23','B_24','B_25','B_26','B_27','B_28','B_29',
#    'B_30','B_31'
#]
#
#big_results_dict = af.prep_big_results_dict_v2(
#    f_shard_number = shard_number,
#    f_Xy_N = Xy_N,
#    f_N_Epoch = N_Epoch,
#    f_Nt = Nt,
#    f_p = p ,
#    f_GP_version = GP_version,
#    f_part_num = part_num,
#    f_predictors = predictors,  
#    f_exp_num=0, 
#    f_comm=True
#)
#
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_10 = af.heat_map_data_prep_std(
#    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=10, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_10)
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_10.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_2_x_20_1000_x_30_1200_x_10.csv',
#    index = False,
#) 
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_10 = af.heat_map_data_prep_std(
#    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=10, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_10)
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_10.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_8_x_20_1000_x_30_1200_x_10.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_10 = af.heat_map_data_prep_std(
#    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=10, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_10)
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_10.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_16_x_20_1000_x_30_1200_x_10.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_10 = af.heat_map_data_prep_std(
#    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=10, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_10)
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_10.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_32_x_20_1000_x_30_1200_x_10.csv',
#    index = False,
#)
#
###################################################################################################################################
#
#                                                 shard_num = [15] 
#
##################################################################################################################################

#shard_number = [15]
#Xy_N=6000
#N_Epoch = [30,150,300,600,1200]
#Nt = [30]
#p = [2, 8, 16, 32]
#GP_version = list(range(10))
#part_num = [20, 100, 1000]
#predictors = [
#    'B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9',
#    'B_10','B_11','B_12','B_13','B_14','B_15','B_16','B_17','B_18','B_19',
#    'B_20','B_21','B_22','B_23','B_24','B_25','B_26','B_27','B_28','B_29',
#    'B_30','B_31'
#]
#
#big_results_dict = af.prep_big_results_dict_v2(
#    f_shard_number = shard_number,
#    f_Xy_N = Xy_N,
#    f_N_Epoch = N_Epoch,
#    f_Nt = Nt,
#    f_p = p ,
#    f_GP_version = GP_version,
#    f_part_num = part_num,
#    f_predictors = predictors,  
#    f_exp_num=0, 
#    f_comm=True
#)
#
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_15 = af.heat_map_data_prep_std(
#    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=15, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_15)
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_15.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_2_x_20_1000_x_30_1200_x_15.csv',
#    index = False,
#) 
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_15 = af.heat_map_data_prep_std(
#    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=15, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_15)
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_15.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_8_x_20_1000_x_30_1200_x_15.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_15 = af.heat_map_data_prep_std(
#    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=15, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_15)
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_15.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_16_x_20_1000_x_30_1200_x_15.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_15 = af.heat_map_data_prep_std(
#    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=15, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_15)
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_15.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_32_x_20_1000_x_30_1200_x_15.csv',
#    index = False,
#)



###################################################################################################################################
#
#                                                 shard_num = [30] 
#
##################################################################################################################################
#
#shard_number = [30]
#Xy_N=6000
#N_Epoch = [30,150,300,600,1200]
#Nt = [30]
#p = [2, 8, 16, 32]
#GP_version = list(range(10))
#part_num = [20, 100, 1000]
#predictors = [
#    'B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9',
#    'B_10','B_11','B_12','B_13','B_14','B_15','B_16','B_17','B_18','B_19',
#    'B_20','B_21','B_22','B_23','B_24','B_25','B_26','B_27','B_28','B_29',
#    'B_30','B_31'
#]
#
#big_results_dict = af.prep_big_results_dict_v2(
#    f_shard_number = shard_number,
#    f_Xy_N = Xy_N,
#    f_N_Epoch = N_Epoch,
#    f_Nt = Nt,
#    f_p = p ,
#    f_GP_version = GP_version,
#    f_part_num = part_num,
#    f_predictors = predictors,  
#    f_exp_num=0, 
#    f_comm=True
#)
#
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_30 = af.heat_map_data_prep_std(
#    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=30, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_30)
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_30.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_2_x_20_1000_x_30_1200_x_30.csv',
#    index = False,
#) 
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_30 = af.heat_map_data_prep_std(
#    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=30, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_30)
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_30.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_8_x_20_1000_x_30_1200_x_30.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_30 = af.heat_map_data_prep_std(
#    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=30, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_30)
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_30.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_16_x_20_1000_x_30_1200_x_30.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_30 = af.heat_map_data_prep_std(
#    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=30, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_30)
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_30.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_32_x_20_1000_x_30_1200_x_30.csv',
#    index = False,
#)
#

###################################################################################################################################
##
##                                                 shard_num = [45] 
##
###################################################################################################################################
#
#shard_number = [45]
#Xy_N=6000
#N_Epoch = [30,150,300,600,1200]
#Nt = [30]
#p = [2, 8, 16, 32]
#GP_version = list(range(10))
#part_num = [20, 100, 1000]
#predictors = [
#    'B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9',
#    'B_10','B_11','B_12','B_13','B_14','B_15','B_16','B_17','B_18','B_19',
#    'B_20','B_21','B_22','B_23','B_24','B_25','B_26','B_27','B_28','B_29',
#    'B_30','B_31'
#]
#
#big_results_dict = af.prep_big_results_dict_v2(
#    f_shard_number = shard_number,
#    f_Xy_N = Xy_N,
#    f_N_Epoch = N_Epoch,
#    f_Nt = Nt,
#    f_p = p ,
#    f_GP_version = GP_version,
#    f_part_num = part_num,
#    f_predictors = predictors,  
#    f_exp_num=0, 
#    f_comm=True
#)
#
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_45 = af.heat_map_data_prep_std(
#    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=45, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_45)
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_45.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_2_x_20_1000_x_30_1200_x_45.csv',
#    index = False,
#) 
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_45 = af.heat_map_data_prep_std(
#    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=45, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_45)
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_45.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_8_x_20_1000_x_30_1200_x_45.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_45 = af.heat_map_data_prep_std(
#    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=45, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_45)
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_45.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_16_x_20_1000_x_30_1200_x_45.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_45 = af.heat_map_data_prep_std(
#    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=45, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_45)
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_45.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_32_x_20_1000_x_30_1200_x_45.csv',
#    index = False,
#)
#
#

###################################################################################################################################
#
#                                                 shard_num = [60] 
#
##################################################################################################################################

#shard_number = [60]
#Xy_N=6000
#N_Epoch = [30,150,300,600,1200]
#Nt = [30]
#p = [2, 8, 16, 32]
#GP_version = list(range(10))
#part_num = [20, 100, 1000]
#predictors = [
#    'B_0','B_1','B_2','B_3','B_4','B_5','B_6','B_7','B_8','B_9',
#    'B_10','B_11','B_12','B_13','B_14','B_15','B_16','B_17','B_18','B_19',
#    'B_20','B_21','B_22','B_23','B_24','B_25','B_26','B_27','B_28','B_29',
#    'B_30','B_31'
#]
#
#big_results_dict = af.prep_big_results_dict_v2(
#    f_shard_number = shard_number,
#    f_Xy_N = Xy_N,
#    f_N_Epoch = N_Epoch,
#    f_Nt = Nt,
#    f_p = p ,
#    f_GP_version = GP_version,
#    f_part_num = part_num,
#    f_predictors = predictors,  
#    f_exp_num=0, 
#    f_comm=True
#)
#
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_60 = af.heat_map_data_prep_std(
#    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=60, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_60)
#Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_60.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_2_x_20_1000_x_30_1200_x_60.csv',
#    index = False,
#) 
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_60 = af.heat_map_data_prep_std(
#    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=60, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_60)
#Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_60.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_8_x_20_1000_x_30_1200_x_60.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_60 = af.heat_map_data_prep_std(
#    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=60, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_60)
#Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_60.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_16_x_20_1000_x_30_1200_x_60.csv',
#    index = False,
#)    
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_60 = af.heat_map_data_prep_std(
#    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=60, big_results_dict=big_results_dict
#)
#print(Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_60)
#Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_60.to_csv(
#    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_32_x_20_1000_x_30_1200_x_60.csv',
#    index = False,
#)


###################################################################################################################################
#
#                                                 shard_num = [120] 
#
##################################################################################################################################
shard_number = [120]
Xy_N=6000
N_Epoch = [30,150,300,600,1200]
Nt = [30]
p = [2, 8, 16, 32]
GP_version = list(range(10))
part_num = [20, 100, 1000]
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
    f_exp_num=0, 
    f_comm=True
)
Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_120 = af.heat_map_data_prep_std(
    pred_num=2, part_num=part_num, N_Epoch = N_Epoch, shard_num=120, big_results_dict=big_results_dict
)
print(Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_120)
Xy_N_6000_hm_plot_data_2_x_20_1000_x_30_1200_x_120.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_2_x_20_1000_x_30_1200_x_120.csv',
    index = False,
)

      
Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_120 = af.heat_map_data_prep_std(
    pred_num=8, part_num=part_num, N_Epoch = N_Epoch, shard_num=120, big_results_dict=big_results_dict
)
print(Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_120)
Xy_N_6000_hm_plot_data_8_x_20_1000_x_30_1200_x_120.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_8_x_20_1000_x_30_1200_x_120.csv',
    index = False,
)

      
Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_120 = af.heat_map_data_prep_std(
    pred_num=16, part_num=part_num, N_Epoch = N_Epoch, shard_num=120, big_results_dict=big_results_dict
)
print(Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_120)
Xy_N_6000_hm_plot_data_16_x_20_1000_x_30_1200_x_120.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_16_x_20_1000_x_30_1200_x_120.csv',
    index = False,
)

      
Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_120 = af.heat_map_data_prep_std(
    pred_num=32, part_num=part_num, N_Epoch = N_Epoch, shard_num=120, big_results_dict=big_results_dict
)
print(Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_120)
Xy_N_6000_hm_plot_data_32_x_20_1000_x_30_1200_x_120.to_csv(
    'experiment_results/heat_map_data/Xy_N_6000_hm_std_plot_data_32_x_20_1000_x_30_1200_x_120.csv',
    index = False,
)
