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
        help='use Gaussian Process generated Betas - defauts to 0.  Select GP=1 to generate GB Betas',
        required=False, default= 0
    )
    parser.add_argument(
        '--GP_version', type=int,
        help='use Gaussian Process generated Betas - defauts to 0.  Select GP=1 to generate GB Betas',
        required=False, default= 0
    )
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    
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
