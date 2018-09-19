import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import itertools

def probit_success_probability(X, b):
    x_j_tB=X.dot(b)
    PHI=norm.cdf(x_j_tB)
    return(PHI)

class pf_plots:
    
    def __init__(self):
        print("let's plot some particle filters!")
        self.distance_from_truth=list()
        self.distance_from_truth_mu=list()
        self.distance_from_truth_par=list()
        self.distance_from_truth_par_mu=list()
        self.distance_from_truth_cmc=list()
        self.distance_from_truth_cmc_mu=list()
        
    def plot_probit_predictive_lik_diff(self):
        plot()
        plt.show()
        
    def prep_one_machine_pf(self, pfo, test):
        for k in range(test['batch_number']):
            d = pfo.get_predictive_distribution(test['X_oos'][k])
            d_mu=np.mean(d)
            truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
            self.distance_from_truth_mu.append(d_mu-truth)
            self.distance_from_truth.append(d-truth)
        self.flattened_list  = list(itertools.chain(*self.distance_from_truth))
        
    def prep_parallel_machine_pf(self, parcobj, test):
        for pf in range(len(parcobj.pf_obj)):
            for k in range(test['batch_number']):
                d = parcobj.pf_obj[pf].get_predictive_distribution(test['X_oos'][k])
                truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
                d_mu=np.mean(d)
                self.distance_from_truth_par_mu.append(d_mu-truth)
                self.distance_from_truth_par.append(d-truth)

        self.flattened_list_par  = list(itertools.chain(*self.distance_from_truth_par))

    def prep_cmc_machine_pf(self, cmcobj, test):
        for pf in range(len(cmcobj.pf_obj)):
            for k in range(test['batch_number']):
                d = cmcobj.pf_obj[pf].get_predictive_distribution(test['X_oos'][k])
                truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
                d_mu=np.mean(d)
                self.distance_from_truth_cmc_mu.append(d_mu-truth)
                self.distance_from_truth_cmc.append(d-truth)

        self.flattened_list_cmc  = list(itertools.chain(*self.distance_from_truth_cmc))
        
    def plot_pred_lik_diff(self, pfo, parcobj, test):
        
        self.prep_one_machine_pf(pfo, test)
        self.prep_parallel_machine_pf(parcobj, test)
        
        plt.hist(x=self.distance_from_truth_mu, bins='auto',  density=True, color='b', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.distance_from_truth_mu))

        plt.hist(x=self.distance_from_truth_par_mu, bins='auto',  density=True, color='green', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.distance_from_truth_par_mu))
        #plt.xticks([0,0.5])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Likelihood Difference')
        plt.axvline(x=0, color='r')
        plt.show()

        plt.hist(x=self.flattened_list, bins='auto',  density=True, color='b', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list))

        plt.hist(x=self.flattened_list_par, bins='auto',  density=True, color='green', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list_par))
        #plt.xticks([0,0.5])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Distribution Difference')
        plt.axvline(x=0, color='r')
        plt.show()
        
    def plot_pred_lik_diff_3_way(self, pfo, parcobj, cmcobj, test):
        
        self.prep_one_machine_pf(pfo, test)
        self.prep_parallel_machine_pf(parcobj, test)
        self.prep_cmc_machine_pf(cmcobj, test)
        
        plt.hist(x=self.distance_from_truth_mu, bins='auto',  density=True, color='b', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.distance_from_truth_mu), color='b')

        plt.hist(x=self.distance_from_truth_par_mu, bins='auto',  density=True, color='green', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.distance_from_truth_par_mu), color='green')
        
        plt.hist(x=self.distance_from_truth_cmc_mu, bins='auto',  density=True, color='r', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.distance_from_truth_cmc_mu), color='r')
        
        #plt.xticks([0,0.5])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Likelihood Difference')
        plt.axvline(x=0, color='r')
        plt.show()

        plt.hist(x=self.flattened_list, bins='auto',  density=True, color='b', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list))

        plt.hist(x=self.flattened_list_par, bins='auto',  density=True, color='green', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list_par))
        
        plt.hist(x=self.flattened_list_cmc, bins='auto',  density=True, color='r', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list_cmc))
        
        #plt.xticks([0,0.5])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Distribution Difference')
        plt.axvline(x=0, color='r')
        plt.show()