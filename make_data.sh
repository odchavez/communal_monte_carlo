

for i in {1,2,3,4,5,6,7,8,9}; #on mac
    do 
        echo "dataset GP_version:" $i
        python run_simulate_data.py --N_total 1000000 --Epoch_N 100000 --predictor_N 32 --N_per_tic 1 --GP_version $i --model_type regression --regression_error 1.0 --data_seed $i
    done
