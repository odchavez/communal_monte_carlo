"""
classification run with: python run_simulate_data.py --N_total 400000 --Epoch_N 2000 --predictor_N 32 --N_per_tic 100 --GP 1 --GP_version 0 

regression run with: python run_simulate_data.py --N_total 20000 --Epoch_N 10000 --predictor_N 32 --N_per_tic 1 --GP_version 1 --model_type regression --regression_error 1.0 --covariance_cutoff .999
"""

import argparse
import simulate_data as sd

def get_args():
    parser = argparse.ArgumentParser(
        description='Runs the synthetic data generator.'
    )
    parser.add_argument(
        '--N_total', type=int,
        help='The number of observations in total accross all epochs and shards.',
        required=True
    )
    parser.add_argument(
        '--Epoch_N', type=int,
        help='The number of observations to have per epoch, ie. before a communication step.',
        required=True
    )
    parser.add_argument(
        '--predictor_N', type=int,
        help='The number of observations to have per epoch, ie. before a communication step.',
        required=True
    )
    parser.add_argument(
        '--N_per_tic', type=int,
        help='The number of observations to have per time tic, ie observations with the same time stamp.',
        required=True
    )
    parser.add_argument(
        '--sin_or_cos', type=str,
        help='use sine or cos path for Beta.',
        required=False, default='cos'
    )
    parser.add_argument(
        '--line_1', type=int,
        help='intercept for line 1.',
        required=False, default=1
    )
    parser.add_argument(
        '--line_2', type=int,
        help='intercept for line 1.',
        required=False, default=-1
    )
    parser.add_argument(
        '--line_3', type=int,
        help='intercept for line 1.',
        required=False, default=-2
    )
    parser.add_argument(
        '--data_seed', type=int,
        help='randomization seed - defauts to 0.',
        required=False, default= 0
    )
    parser.add_argument(
        '--GP', type=int,
        help='use Gaussian Process generated Betas - defauts to 0.  Select GP=1 to generate GP Betas',
        required=False, default= 0
    )
    parser.add_argument(
        '--GP_version', type=int,
        help='use Gaussian Process generated Betas - defauts to 0.  Select GP=1 to generate GP Betas',
        required=False, default= 0
    )
    parser.add_argument(
        '--model_type', type=str,
        help='create classification or regression data',
        required=True
    )
    parser.add_argument(
        '--regression_error', type=float,
        help='Amount of noise to add to Y = X*Beta + error',
        required=False, default= 1.
    )
    parser.add_argument(
        '--covariance_cutoff', type=float,
        help='set to zero all values of covariance below cutoff',
        required=False, default= 0.
    )
    parser.add_argument(
        '--h', type=float,
        help='cov param',
        required=False, default= 0.
    )
    parser.add_argument(
        '--lam', type=float,
        help='cov param',
        required=False, default= 0.
    )
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    if args.model_type == "classification":
        sim_data_obj = sd.simulated_data2(
            n_per_file = args.Epoch_N, 
            N_total = args.N_total, 
            n_per_tic = args.N_per_tic, 
            pred_number=args.predictor_N,
            seed = args.data_seed,
            GP=args.GP,
            GB_version=args.GP_version
        )
        sim_data_obj.generate_Betas(
            sin_or_cos = args.sin_or_cos, 
            intercepts_1=args.line_1, 
            intercepts_2=args.line_2, 
            intercepts_3=args.line_3,   
        )
        sim_data_obj.generate_data()
    elif args.model_type == "regression": 
        sim_data_obj = sd.simulated_data_regression(
            n_per_file = args.Epoch_N, 
            N_total = args.N_total, 
            n_per_tic = args.N_per_tic, 
            pred_number=args.predictor_N,
            seed = args.data_seed,
            GP_version=args.GP_version,
            err_std = args.regression_error
        )
        sim_data_obj.generate_Betas()
        sim_data_obj.generate_data()
    else:
        raise ValueError('{} modle is not implemented'.format(method))