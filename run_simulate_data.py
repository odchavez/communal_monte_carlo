import argparse
import simulate_data as sd

def get_args():
    parser = argparse.ArgumentParser(
        description='Runs the synthetic data generator.'
    )
    parser.add_argument(
        '--obs_per_epoch', type=int,
        help='The number of observations to have per epoch, ie. before a communication step.',
        required=True
    )
    return parser.parse_args()

if __name__ == "__main__":
    
    args = get_args()
    
    sim_data_obj = sd.simulated_data2(n_per_file = args.obs_per_epoch)
    sim_data_obj.generate_Betas()
    sim_data_obj.generate_data()
