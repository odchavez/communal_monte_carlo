import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import itertools
#import seaborn as sns

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
        self.distance_from_truth_par_grand_mu=list()
        
        self.distance_from_truth_cmc=list()
        self.distance_from_truth_cmc_mu=list()
        self.distance_from_truth_cmc_grand_mu=list()
        
        self.distance_from_truth_cmc_wass=list()
        self.distance_from_truth_cmc_mu_wass=list()
        self.distance_from_truth_cmc_grand_mu_wass=list()
        
    def plot_probit_predictive_lik_diff(self):
        plot()
        plt.show()
        
    def prep_one_machine_pf(self, pfo, test):
        big_K=test['X_oos'].shape[0]
        for k in range(big_K):
            d = pfo.get_predictive_distribution(test['X_oos'][k])
            d_mu=np.mean(d)
            truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
            self.distance_from_truth_mu.append(np.abs(d_mu-truth))
            self.distance_from_truth.append(np.abs(d-truth))
        self.flattened_list  = list(itertools.chain(*self.distance_from_truth))
        
    def prep_parallel_machine_pf(self, parcobj, test):
        for pf in range(len(parcobj.pf_obj)):
            big_K=test['X_oos'].shape[0]
            for k in range(big_K):
                d = parcobj.pf_obj[pf].get_predictive_distribution(test['X_oos'][k])
                truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
                d_mu=np.mean(d)
                self.distance_from_truth_par_mu.append(np.abs(d_mu-truth))
                self.distance_from_truth_par.append(np.abs(d-truth))

        self.flattened_list_par  = list(itertools.chain(*self.distance_from_truth_par))

    def prep_cmc_machine_pf(self, cmcobj, test):
        for pf in range(len(cmcobj.pf_obj)):
            big_K=test['X_oos'].shape[0]
            for k in range(big_K):                
                d = cmcobj.pf_obj[pf].get_predictive_distribution(test['X_oos'][k])
                truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
                d_mu=np.mean(d)
                self.distance_from_truth_cmc_mu.append(np.abs(d_mu-truth))
                self.distance_from_truth_cmc.append(np.abs(d-truth))
        self.flattened_list_cmc  = list(itertools.chain(*self.distance_from_truth_cmc))

    def prep_cmc_machine_pf_wass(self, cmcobj, test):
        for pf in range(len(cmcobj.pf_obj)):
            big_K=test['X_oos'].shape[0]
            for k in range(big_K):                
                d = cmcobj.pf_obj[pf].get_predictive_distribution(test['X_oos'][k])
                truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
                d_mu=np.mean(d)
                self.distance_from_truth_cmc_mu_wass.append(np.abs(d_mu-truth))
                self.distance_from_truth_cmc_wass.append(np.abs(d-truth))
        self.flattened_list_cmc_wass  = list(itertools.chain(*self.distance_from_truth_cmc_wass))
        
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
        mu       = np.mean(self.distance_from_truth_par_mu)
        muplus   = mu + np.std(self.distance_from_truth_par_mu)/np.sqrt(len(self.distance_from_truth_par_mu))
        muminus  = mu - np.std(self.distance_from_truth_par_mu)/np.sqrt(len(self.distance_from_truth_par_mu))
        plt.axvline(x=muplus, color='r')
        plt.axvline(x=muminus, color='r')
        
        plt.hist(x=self.distance_from_truth_cmc_mu, bins='auto',  density=True, color='r', alpha=0.5)#, rwidth=0.85)
        mu       = np.mean(self.distance_from_truth_cmc_mu)
        muplus   = mu + np.std(self.distance_from_truth_cmc_mu)/np.sqrt(len(self.distance_from_truth_cmc_mu))
        muminus  = mu - np.std(self.distance_from_truth_cmc_mu)/np.sqrt(len(self.distance_from_truth_cmc_mu))
        plt.axvline(x=muplus, color='r')
        plt.axvline(x=muminus, color='r')
        
        #plt.xticks([0,0.5])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Likelihood Difference')
        plt.axvline(x=0, color='r')
        plt.show()

        plt.hist(x=self.flattened_list, bins='auto',  density=True, color='b', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list), color='b')

        plt.hist(x=self.flattened_list_par, bins='auto',  density=True, color='green', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list_par), color='g')
        
        plt.hist(x=self.flattened_list_cmc, bins='auto',  density=True, color='r', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list_cmc), color='r')
        
        #plt.xticks([0,0.5])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Distribution Difference')
        plt.axvline(x=0, color='r')
        plt.show()
        
    def plot_pred_lik_diff_3_way_experiment_run(self, pfo_list, parcobj_list, cmcobj_list, test):
        
        self.prep_one_machine_pf_experiment_run(pfo_list, test)
        self.prep_parallel_machine_pf_experiment_run(parcobj_list, test)
        self.prep_cmc_machine_pf_experiment_run(cmcobj_list, test)
        
        sns.distplot(self.distance_from_truth_mu, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='b')
        #plt.hist(x=self.distance_from_truth_mu, bins='auto',  density=True, color='b', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.distance_from_truth_mu), color='b')

        sns.distplot(self.distance_from_truth_par_mu, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='g')
        #plt.hist(x=self.distance_from_truth_par_mu, bins='auto',  density=True, color='green', alpha=0.5)#, rwidth=0.85)
        mu       = np.mean(self.distance_from_truth_par_mu)
        muplus   = mu + 2*np.std(self.distance_from_truth_par_grand_mu)/np.sqrt(len(self.distance_from_truth_par_grand_mu))
        muminus  = mu - 2*np.std(self.distance_from_truth_par_grand_mu)/np.sqrt(len(self.distance_from_truth_par_grand_mu))
        plt.axvline(x=muplus, color='g')
        plt.axvline(x=muminus, color='g')
        plt.axvline(x=np.mean(self.distance_from_truth_par_mu), color='green')
        
        sns.distplot(self.distance_from_truth_cmc_mu, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='r')
        #plt.hist(x=self.distance_from_truth_cmc_mu, bins='auto',  density=True, color='r', alpha=0.5)#, rwidth=0.85)
        mu       = np.mean(self.distance_from_truth_cmc_mu)
        muplus   = mu + 2*np.std(self.distance_from_truth_cmc_grand_mu)/np.sqrt(len(self.distance_from_truth_cmc_grand_mu))
        muminus  = mu - 2*np.std(self.distance_from_truth_cmc_grand_mu)/np.sqrt(len(self.distance_from_truth_cmc_grand_mu))
        plt.axvline(x=muplus, color='r')
        plt.axvline(x=muminus, color='r')
        plt.axvline(x=np.mean(self.distance_from_truth_cmc_mu), color='r')
        
        #plt.xticks([0,0.5])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Likelihood Difference')
        plt.axvline(x=0, color='r')
        plt.show()

        sns.distplot(self.flattened_list, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='b')
        #plt.hist(x=self.flattened_list, bins='auto',  density=True, color='b', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list), color='b')

        sns.distplot(self.flattened_list_par, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='g')
        #plt.hist(x=self.flattened_list_par, bins='auto',  density=True, color='green', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list_par), color='g')

        sns.distplot(self.flattened_list_cmc, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='r')
        #plt.hist(x=self.flattened_list_cmc, bins='auto',  density=True, color='r', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list_cmc), color='r')
        
        #plt.xticks([0,0.5])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Distribution Difference')
        plt.axvline(x=0, color='r')
        plt.show()
        
    def prep_one_machine_pf_experiment_run(self, pfo_list, test):
        big_K=test['X_oos'].shape[0]
        for er in range(len(pfo_list)):
            pfo = pfo_list[er]
            for k in range(big_K):
                d = pfo.get_predictive_distribution(test['X_oos'][k])
                d_mu=np.mean(d)
                truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
                self.distance_from_truth_mu.append(np.abs(d_mu-truth))
                self.distance_from_truth.append(np.abs(d-truth))
        self.flattened_list  = list(itertools.chain(*self.distance_from_truth))
        
    def prep_parallel_machine_pf_experiment_run(self, parcobj_list, test):
        big_K=test['X_oos'].shape[0]
        for er in range(len(parcobj_list)):
            parcobj = parcobj_list[er]
            temp=list()
            for pf in range(len(parcobj.pf_obj)):
                for k in range(big_K):
                    d = parcobj.pf_obj[pf].get_predictive_distribution(test['X_oos'][k])
                    truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
                    d_mu=np.mean(d)
                    self.distance_from_truth_par_mu.append(np.abs(d_mu-truth))
                    self.distance_from_truth_par.append(np.abs(d-truth))
                    temp.append(np.abs(d_mu-truth))
            
            self.distance_from_truth_par_grand_mu.append(np.mean(temp))
        self.flattened_list_par  = list(itertools.chain(*self.distance_from_truth_par))

    def prep_cmc_machine_pf_experiment_run(self, cmcobj_list, test):
        big_K=test['X_oos'].shape[0]
        for er in range(len(cmcobj_list)):
            temp=list()
            cmcobj = cmcobj_list[er]
            for pf in range(len(cmcobj.pf_obj)):
                for k in range(big_K):                
                    d = cmcobj.pf_obj[pf].get_predictive_distribution(test['X_oos'][k])
                    truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
                    d_mu=np.mean(d)
                    self.distance_from_truth_cmc_mu.append(np.abs(d_mu-truth))
                    self.distance_from_truth_cmc.append(np.abs(d-truth))
                    temp.append(np.abs(d_mu-truth))
            self.distance_from_truth_cmc_grand_mu.append(np.mean(temp))
        self.flattened_list_cmc  = list(itertools.chain(*self.distance_from_truth_cmc))
        
    def prep_cmc_machine_pf_experiment_run_wass(self, cmcobj_list, test):
        big_K=test['X_oos'].shape[0]
        for er in range(len(cmcobj_list)):
            temp=list()
            cmcobj = cmcobj_list[er]
            for pf in range(len(cmcobj.pf_obj)):
                for k in range(big_K):                
                    d = cmcobj.pf_obj[pf].get_predictive_distribution(test['X_oos'][k])
                    truth=probit_success_probability(test['X_oos'][k], test['b_oos'][0])
                    d_mu=np.mean(d)
                    self.distance_from_truth_cmc_mu_wass.append(np.abs(d_mu-truth))
                    self.distance_from_truth_cmc_wass.append(np.abs(d-truth))
                    temp.append(np.abs(d_mu-truth))
            self.distance_from_truth_cmc_grand_mu_wass.append(np.mean(temp))
        self.flattened_list_cmc_wass  = list(itertools.chain(*self.distance_from_truth_cmc_wass))
        
    def plot_pred_lik_diff_4_way(self, pfo, parcobj, cmcobj, cmcwassobj, test):
        
        self.prep_one_machine_pf(pfo, test)
        self.prep_parallel_machine_pf(parcobj, test)
        self.prep_cmc_machine_pf(cmcobj, test)
        self.prep_cmc_machine_pf_wass(cmcwassobj, test)
        
        plt.hist(x=self.distance_from_truth_mu, bins='auto',  density=True, color='b', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.distance_from_truth_mu), color='b')

        plt.hist(x=self.distance_from_truth_par_mu, bins='auto',  density=True, color='green', alpha=0.5)#, rwidth=0.85)
        mu       = np.mean(self.distance_from_truth_par_mu)
        muplus   = mu + np.std(self.distance_from_truth_par_mu)/np.sqrt(len(self.distance_from_truth_par_mu))
        muminus  = mu - np.std(self.distance_from_truth_par_mu)/np.sqrt(len(self.distance_from_truth_par_mu))
        plt.axvline(x=muplus, color='r')
        plt.axvline(x=muminus, color='r')
        
        plt.hist(x=self.distance_from_truth_cmc_mu, bins='auto',  density=True, color='r', alpha=0.5)#, rwidth=0.85)
        mu       = np.mean(self.distance_from_truth_cmc_mu)
        muplus   = mu + np.std(self.distance_from_truth_cmc_mu)/np.sqrt(len(self.distance_from_truth_cmc_mu))
        muminus  = mu - np.std(self.distance_from_truth_cmc_mu)/np.sqrt(len(self.distance_from_truth_cmc_mu))
        plt.axvline(x=muplus, color='r')
        plt.axvline(x=muminus, color='r')
        
        plt.hist(x=self.distance_from_truth_cmc_mu_wass, bins='auto',  density=True, color='pink', alpha=0.5)#, rwidth=0.85)
        mu       = np.mean(self.distance_from_truth_cmc_mu_wass)
        muplus   = mu + np.std(self.distance_from_truth_cmc_mu_wass)/np.sqrt(len(self.distance_from_truth_cmc_mu_wass))
        muminus  = mu - np.std(self.distance_from_truth_cmc_mu_wass)/np.sqrt(len(self.distance_from_truth_cmc_mu_wass))
        plt.axvline(x=muplus, color='pink')
        plt.axvline(x=muminus, color='pink')
        
        #plt.xticks([0,0.5])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Likelihood Difference')
        plt.axvline(x=0, color='r')
        plt.show()

        plt.hist(x=self.flattened_list, bins='auto',  density=True, color='b', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list), color='b')

        plt.hist(x=self.flattened_list_par, bins='auto',  density=True, color='green', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list_par), color='g')
        
        plt.hist(x=self.flattened_list_cmc, bins='auto',  density=True, color='r', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list_cmc), color='r')
        
        plt.hist(x=self.flattened_list_cmc_wass, bins='auto',  density=True, color='pink', alpha=0.5)#, rwidth=0.85)
        plt.axvline(x=np.mean(self.flattened_list_cmc), color='pink')
        
        #plt.xticks([0,0.5])
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Distribution Difference')
        plt.axvline(x=0, color='r')
        plt.show()
        
    def plot_pred_lik_diff_4_way_experiment_run(self, pfo_list, parcobj_list, cmcobj_list, cmcobjwass_list, test):
        
        self.prep_one_machine_pf_experiment_run(pfo_list, test)
        self.prep_parallel_machine_pf_experiment_run(parcobj_list, test)
        self.prep_cmc_machine_pf_experiment_run(cmcobj_list, test)
        self.prep_cmc_machine_pf_experiment_run_wass(cmcobjwass_list, test)
        
        sns.distplot(self.distance_from_truth_mu, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='b')
        plt.axvline(x=np.mean(self.distance_from_truth_mu), color='b')

        sns.distplot(self.distance_from_truth_par_mu, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='g')
        mu       = np.mean(self.distance_from_truth_par_mu)
        muplus   = mu + 2*np.std(self.distance_from_truth_par_grand_mu)/np.sqrt(len(self.distance_from_truth_par_grand_mu))
        muminus  = mu - 2*np.std(self.distance_from_truth_par_grand_mu)/np.sqrt(len(self.distance_from_truth_par_grand_mu))
        plt.axvline(x=muplus, color='g')
        plt.axvline(x=muminus, color='g')
        plt.axvline(x=np.mean(self.distance_from_truth_par_mu), color='green')
        
        sns.distplot(self.distance_from_truth_cmc_mu, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='r')
        mu       = np.mean(self.distance_from_truth_cmc_mu)
        muplus   = mu + 2*np.std(self.distance_from_truth_cmc_grand_mu)/np.sqrt(len(self.distance_from_truth_cmc_grand_mu))
        muminus  = mu - 2*np.std(self.distance_from_truth_cmc_grand_mu)/np.sqrt(len(self.distance_from_truth_cmc_grand_mu))
        plt.axvline(x=muplus, color='r')
        plt.axvline(x=muminus, color='r')
        plt.axvline(x=np.mean(self.distance_from_truth_cmc_mu), color='r')
        
        sns.distplot(self.distance_from_truth_cmc_mu_wass, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='pink')
        mu       = np.mean(self.distance_from_truth_cmc_mu_wass)
        muplus   = mu + 2*np.std(self.distance_from_truth_cmc_grand_mu_wass)/np.sqrt(len(self.distance_from_truth_cmc_grand_mu_wass))
        muminus  = mu - 2*np.std(self.distance_from_truth_cmc_grand_mu_wass)/np.sqrt(len(self.distance_from_truth_cmc_grand_mu_wass))
        plt.axvline(x=muplus, color='pink')
        plt.axvline(x=muminus, color='pink')
        plt.axvline(x=np.mean(self.distance_from_truth_cmc_mu_wass), color='r')
        
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Likelihood Difference')
        plt.axvline(x=0, color='r')
        plt.show()

        sns.distplot(self.flattened_list, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='b')
        plt.axvline(x=np.mean(self.flattened_list), color='b')

        sns.distplot(self.flattened_list_par, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='g')
        plt.axvline(x=np.mean(self.flattened_list_par), color='g')

        sns.distplot(self.flattened_list_cmc, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='r')
        plt.axvline(x=np.mean(self.flattened_list_cmc), color='r')
        
        sns.distplot(self.flattened_list_cmc_wass, hist = False, kde = True, kde_kws = {'linewidth': 3}, label = None, color='pink')
        plt.axvline(x=np.mean(self.flattened_list_cmc), color='pink')        
        
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('P(Y_est=1) - P(Y_true=1)')
        plt.ylabel('Density')
        plt.title('Predictive Distribution Difference')
        plt.axvline(x=0, color='r')
        plt.show()