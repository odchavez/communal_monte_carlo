import numpy as np
import os
import glob



class parameter_history:
    
    def __init__(self):
        pass

    def write_stats_results(self,  f_stats_df, f_other_stats_file='experiment_results/results.csv'):
        print("writing experimental run statistics to ", f_other_stats_file)
        f_stats_df.to_csv(
            f_other_stats_file, 
            index = False, 
        )
        