

for i in {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}; #on mac
    do 
        echo "dataset GP_version:" $i
        python run_simulate_data.py --N_total 1000000 --Epoch_N 10000 --predictor_N 32 --N_per_tic 10 --GP_version $i --model_type classification --data_seed $i
    done
