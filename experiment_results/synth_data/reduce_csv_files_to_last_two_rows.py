import pandas as pd
import numpy as np
from all_results_file_names import all_results_file_names

for i in range(len(all_results_file_names)):
    
    path = all_results_file_names[i]
    
    try:
        data = pd.read_csv(path).tail(2)
        data.to_csv(path, index=False)
        print("SUCCESS WITH: ", path)
    except:
        print("FAIL WITH: ", path)