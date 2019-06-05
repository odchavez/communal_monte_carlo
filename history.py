#from matplotlib import pyplot as plt
#import pandas as pd
import numpy as np
import os
#import particle_filter
#import pf_models as pfm
#import math as m
#from random import randint
#from scipy.optimize import linprog
#import seaborn as sns
import glob



class parameter_history:
    
    def __init__(self):
        
        self.bo_list_history = list()
        self.bo_machine_list_history = list()
    
    def compile_bo_list_history(self, f_name_stem=''):
        
        nrun, nshard = self.get_particle_history_dim("particle_hold/file_"+f_name_stem+"_*.npy")
        for nr in range(nrun):
             
            for ns in range(nshard):
                loop_file_name = "particle_hold/file_" + f_name_stem + "_" + str(nr) + "_" + str(ns) + ".npy"
                loaded_bo_file = np.load(loop_file_name)
                print("line 30 loaded_bo_file.shape = " + str(loaded_bo_file.shape))
                if ns == 0:
                    shard_history_temp = loaded_bo_file.copy()
                else:
                    shard_history_temp = np.append(shard_history_temp, loaded_bo_file, axis = 2)
            
            
            print("shard_history_temp.shape = " + str(shard_history_temp.shape))
            if nr == 0:
                self.bo_list_history = shard_history_temp.copy()
                print("self.bo_list_history.shape = " + str(self.bo_list_history.shape))
            else:
                self.bo_list_history = np.append(self.bo_list_history, shard_history_temp, axis = 0)
                #print(loaded_bo_file.shape)
            
            
        #particle_file_names = []
        #for file in glob.glob("particle_hold/*.npy"):
        #    particle_file_names.append(file)
        #print("##########################################################")
        #print("type(particle_file_names) = " + str(type(particle_file_names)))
        #for i in range(len(particle_file_names)):
        #    print("attempting to open " + particle_file_names[i])
        #    loaded_bo_file = np.load(particle_file_names[i])
        #    if len(self.bo_list_history) == 0:
        #        self.bo_list_history = loaded_bo_file
        #    else:
        #        self.bo_list_history = np.append(self.bo_list_history, loaded_bo_file, axis = 0)
        #        print("self.bo_list_history.shape = ", self.bo_list_history.shape)
                
    def get_particle_history_dim(self, history_file_path_name_extension = "particle_hold/*.npy"):
        import glob
        #import re
        
        txtfiles = []
        for file in glob.glob(history_file_path_name_extension):
            txtfiles.append(file)
            
        file_list = sorted(txtfiles)
        
        nrun = 0
        nshard = 0
        for i in range(len(file_list)):
            newstr = ''.join((ch if ch in '0123456789' else ' ') for ch in file_list[i])
            newstr
            listOfNumbers = [int(i) for i in newstr.split()]
            nrun = max(nrun, listOfNumbers[0]+1)
            nshard = max(nshard, listOfNumbers[1]+1)
            
        return nrun, nshard  
    
    def append_bo_list_history(self, parallel_com_obj ):
        print("type(parallel_com_obj)=", type(parallel_com_obj))
        temp_all_parts = self.get_temp_all_particles(parallel_com_obj)
        print("shape of temp_all_parts = ", temp_all_parts.shape)
        if len(self.bo_list_history) == 0:
            self.bo_list_history = temp_all_parts
        else:
            self.bo_list_history = np.append(self.bo_list_history, temp_all_parts, axis = 0)
            print("self.bo_list_history.shape = ", self.bo_list_history.shape)
            
    #def append_bo_machine_list_history(self, parallel_com_obj):
        
        #if len(self.bo_machine_list_history) == 0:
        #    self.bo_machione_list_history = parallel_com_obj
        #else:
        #    # appending the bo_list makes it so that each index is an epoch
        #    self.bo_machine_list_history.append(parallel_com_obj)
            
        #return
        
    def get_temp_all_particles(self, parallel_com_obj):
        particle_number = len(parallel_com_obj[0].particle_list)
        number_of_shards = len(parallel_com_obj)
        Z_dim = number_of_shards * particle_number
        #total_time_steps = len( parallel_com_obj[0].get_particle(0).bo_machine_list[:,0] )
        bo_shape =  parallel_com_obj[0].particle_list[0].bo_list.shape
        print("base bo list shape:", bo_shape )
        temp_all_parts = np.zeros((bo_shape[0], bo_shape[1], Z_dim))
    
        counter = 0
        for sn in range(len(parallel_com_obj)):
            #print(sn)
            #print(
            #    "(parallel_com_obj[i].particle_list[0].shape) = " ,
            #    parallel_com_obj[sn].particle_list[0].bo_list.shape
            #)
            for pn in range(particle_number):
                particle = parallel_com_obj[sn].get_particle(pn)
                temp_all_parts[:,:,counter] = (
                    parallel_com_obj[sn].particle_list[pn].bo_list #particle.bo_machine_list.copy()
                )
                counter+=1
                
        return temp_all_parts