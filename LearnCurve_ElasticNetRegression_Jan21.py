import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import glob
import random

from sklearn.linear_model import ElasticNet

#Load SLATM representations of 10,000 geometries
load_SLATM_name = 'Results/SLATM_rep_full.npz'
dict_SLATM_rep_full = np.load(load_SLATM_name)
SLATM_rep_full = dict_SLATM_rep_full['SLATM_rep_full_svd']
print("SLATM_rep_full.shape = ", SLATM_rep_full.shape)
input_dim = SLATM_rep_full.shape[1]
print('input_dim = ', input_dim)
energy_DFTB_full = np.genfromtxt('raw_data/E-DFTB.dat', skip_header=0, skip_footer=0, names=None, dtype=None, delimiter=' ')
print("energy_DFTB_full.shape = ", energy_DFTB_full.shape)

energy_ZINDO_full = np.genfromtxt('raw_data/E-ZINDO.dat', skip_header=0, skip_footer=0, names=None, dtype=None, delimiter=' ')
print("energy_ZINDO_full.shape = ", energy_ZINDO_full.shape)

#Take a small portion of data for parameter optimization
subset_size_for_params_optimize = 516
SLATM_rep_sub = SLATM_rep_full[0:subset_size_for_params_optimize, :]
energy_DFTB_sub = energy_DFTB_full[0:subset_size_for_params_optimize]
energy_ZINDO_sub = energy_ZINDO_full[0:subset_size_for_params_optimize]
print('SLATM_rep_sub.shape = ', SLATM_rep_sub.shape)
print('energy_DFTB_sub = ', energy_DFTB_sub.shape)
print('energy_ZINDO_sub = ', energy_ZINDO_sub.shape)

'''
#Optimize alpha, l1_ratio - Start here

def ParamOpt_alpha(energy_method, input, output):
    alpha_try_ENR = np.logspace(-10, 10, num=21, base=2) #try eight values from roughly 2^-10 (10^-3) to 2^10 (10^3)
    #Initialize the arrays of hyperparameter values to try out and their corresponding mean absolute error (MAE)
    alpha_try_implemented_ENR = []
    alphaOpt_MAE_average_ENR_array = []
    #Loop over potential hyperparameter values and build a corresponding model
    for alpha_ENR in alpha_try_ENR:
        #Initialize the Elastic Net Regression model
        EN_Regressor = ElasticNet(alpha=alpha_ENR)
        #Apply 5-fold cross-validation on the RFR model with negative MAE as the score, then get the MAE
        neg_MAE_crossval_kfold_ENR = cross_val_score(EN_Regressor, input, output, cv=5, scoring='neg_mean_absolute_error')
        MAE_average_ENR = np.mean(neg_MAE_crossval_kfold_ENR)*(-1)
        print('MAE_average_ENR = ', MAE_average_ENR)
        #Append the successfully implemented hyperparameter and corresponding MAE to their arrays for saving
        alpha_try_implemented_ENR.append(alpha_ENR)
        alphaOpt_MAE_average_ENR_array.append(MAE_average_ENR)
        # Save the latest arrays of hyperparameter values and their corresponding MAE
        np.savetxt('Results/' + energy_method + '_alpha_try_implemented_ENR.csv', alpha_try_implemented_ENR, delimiter=',')
        np.savetxt('Results/' + energy_method + '_alphaOpt_MAE_average_ENR_array.csv', alphaOpt_MAE_average_ENR_array, delimiter=',')

    #Find the index of the hyperparameter value which returns the lowest MAE
    index_alpha_crossval_minMAE_ENR = np.where(alphaOpt_MAE_average_ENR_array == np.min(alphaOpt_MAE_average_ENR_array))[0][0] #can take twice component with index 0 because the index returned as a list of a 1-member array
    print('index_alpha_crossval_minMAE_ENR = ', index_alpha_crossval_minMAE_ENR)
    #Find the corresponding optimized hyperparameter
    alpha_crossval_optimized_ENR = alpha_try_ENR[index_alpha_crossval_minMAE_ENR]
    print('alpha_crossval_optimized_ENR =', alpha_crossval_optimized_ENR)
    #Save the optimized hyperparameter
    np.savetxt('Results/' + energy_method + '_alpha_crossval_optimized_ENR.csv', [alpha_crossval_optimized_ENR], delimiter=',')

    #Plot optimization curve - Start here
    alpha_try_implemented_ENR = np.loadtxt('Results/' + energy_method + '_alpha_try_implemented_ENR.csv',  delimiter=',')
    alphaOpt_MAE_average_ENR_array = np.loadtxt('Results/' + energy_method + '_alphaOpt_MAE_average_ENR_array.csv', delimiter=',')

    plt.figure()
    plt.loglog(alpha_try_implemented_ENR, alphaOpt_MAE_average_ENR_array)
    plt.title('ENR cross-validation MAE for alpha with ' + energy_method)
    plt.xlabel('log of alpha')
    plt.ylabel('log of MAE')
    plt.savefig(str('Results/Jan21_crossval_MAE_alpha_ENR_' + energy_method + '.png'), bbox_inches='tight')
    # Plot optimization curve - End here

ParamOpt_alpha('DFTB', SLATM_rep_sub, energy_DFTB_sub)
ParamOpt_alpha('ZINDO', SLATM_rep_sub, energy_ZINDO_sub)



def ParamOpt_l1_ratio(energy_method, input, output):
    alpha_crossval_optimized_ENR = np.loadtxt('Results/' + energy_method + '_alpha_crossval_optimized_ENR.csv', delimiter=',')
    l1_ratio_try_ENR = np.linspace(start=0, stop=1, num=21) #try 21 values evenly spaced from 0 to 1 for the relative weight of l1 loss
    #Initialize the arrays of hyperparameter values to try out and their corresponding mean absolute error (MAE)
    l1_ratio_try_implemented_ENR = []
    l1_ratioOpt_MAE_average_ENR_array = []
    #Loop over potential hyperparameter values and build a corresponding model
    for l1_ratio_ENR in l1_ratio_try_ENR:
        #Initialize the Elastic Net Regression model
        EN_Regressor = ElasticNet(alpha=alpha_crossval_optimized_ENR, l1_ratio=l1_ratio_ENR)
        #Apply 5-fold cross-validation on the RFR model with negative MAE as the score, then get the MAE
        neg_MAE_crossval_kfold_ENR = cross_val_score(EN_Regressor, input, output, cv=5, scoring='neg_mean_absolute_error')
        MAE_average_ENR = np.mean(neg_MAE_crossval_kfold_ENR)*(-1)
        print('MAE_average_ENR = ', MAE_average_ENR)
        #Append the successfully implemented hyperparameter and corresponding MAE to their arrays for saving
        l1_ratio_try_implemented_ENR.append(l1_ratio_ENR)
        l1_ratioOpt_MAE_average_ENR_array.append(MAE_average_ENR)
        # Save the latest arrays of hyperparameter values and their corresponding MAE
        np.savetxt('Results/' + energy_method + '_l1_ratio_try_implemented_ENR.csv', l1_ratio_try_implemented_ENR, delimiter=',')
        np.savetxt('Results/' + energy_method + '_l1_ratioOpt_MAE_average_ENR_array.csv', l1_ratioOpt_MAE_average_ENR_array, delimiter=',')

    #Find the index of the hyperparameter value which returns the lowest MAE
    index_l1_ratio_crossval_minMAE_ENR = np.where(l1_ratioOpt_MAE_average_ENR_array == np.min(l1_ratioOpt_MAE_average_ENR_array))[0][0] #can take twice component with index 0 because the index returned as a list of a 1-member array
    print('index_l1_ratio_crossval_minMAE_ENR = ', index_l1_ratio_crossval_minMAE_ENR)
    #Find the corresponding optimized hyperparameter
    l1_ratio_crossval_optimized_ENR = l1_ratio_try_ENR[index_l1_ratio_crossval_minMAE_ENR]
    print('l1_ratio_crossval_optimized_ENR =', l1_ratio_crossval_optimized_ENR)
    #Save the optimized hyperparameter
    np.savetxt('Results/' + energy_method + '_l1_ratio_crossval_optimized_ENR.csv', [l1_ratio_crossval_optimized_ENR], delimiter=',')

    #Plot optimization curve - Start here
    l1_ratio_try_implemented_ENR = np.loadtxt('Results/' + energy_method + '_l1_ratio_try_implemented_ENR.csv',  delimiter=',')
    l1_ratioOpt_MAE_average_ENR_array = np.loadtxt('Results/' + energy_method + '_l1_ratioOpt_MAE_average_ENR_array.csv', delimiter=',')

    plt.figure()
    plt.loglog(l1_ratio_try_implemented_ENR, l1_ratioOpt_MAE_average_ENR_array)
    plt.title('ENR cross-validation MAE for l1_ratio with ' + energy_method)
    plt.xlabel('log of l1_ratio')
    plt.ylabel('log of MAE')
    plt.savefig(str('Results/Jan21_crossval_MAE_l1_ratio_ENR_' + energy_method + '.png'), bbox_inches='tight')
    # Plot optimization curve - End here

ParamOpt_l1_ratio('DFTB', SLATM_rep_sub, energy_DFTB_sub)
ParamOpt_l1_ratio('ZINDO', SLATM_rep_sub, energy_ZINDO_sub)



#Optimize alpha, l1_ratio - End here
'''



# Define and call a function to plot the learning curve for increasing number of samples - Start here
import time
def LearnCurve(energy_method, input_full, output_full):
    #Define an array of potential sample sizes for the learning curves
    sample_size_ENR_array = np.logspace(5, 13, num=9, base=2)
    print('sample_size_ENR_array = ', sample_size_ENR_array)
    #Train and cross-validate with various sample size - Start here

    #Load the optimized hyperparameter to feed the learning-curve-construction model
    alpha_crossval_optimized_ENR = int(np.loadtxt('Results/' + energy_method + '_alpha_crossval_optimized_ENR.csv', delimiter=','))
    l1_ratio_crossval_optimized_ENR = int(np.loadtxt('Results/' + energy_method + '_l1_ratio_crossval_optimized_ENR.csv', delimiter=','))

    #ENRmodel_LearnCurve = ElasticNet(alpha=alpha_crossval_optimized_ENR, l1_ratio=l1_ratio_crossval_optimized_ENR)
    ENRmodel_LearnCurve = ElasticNet() #Do not use the empirically optimized hyperparameter l1_ratio = 0 because it led to explosion of error in the training curve
    #Initialize the arrays to collect each implemented sample size and its corresponding positive MAE, run time
    LearnCurve_ENR_avg_score_array = []
    runtime_ENR_array = []
    sample_size_ENR_implemented_array = []
    #Apply the model on different sample sizes
    for sample_size_ENR in sample_size_ENR_array:
         #Use try-except with break to stop the loop when the training/cross-validation size becomes too large
        try:
            sample_size_ENR = int(sample_size_ENR)
            #Cut the full input and output arrays into smaller arrays of the sample size
            input = input_full[0:sample_size_ENR, :]
            output = output_full[0:sample_size_ENR]
            #Record the time before and after running the model 5-fold cross validation to calculate their difference for the time complexity curve
            start_ENR = time.time()
            score_crossval_ENR_array = cross_val_score(ENRmodel_LearnCurve, input, output, cv=5, scoring='neg_mean_absolute_error')
            end_ENR = time.time()
            runtime_ENR = end_ENR - start_ENR

            print('score_crossval_ENR_array = ', score_crossval_ENR_array)
            #Get the averaged positive MAE from the array of 5 MAE values returned by 5-fold cross-validation of the model
            LearnCurve_ENR_avg_score = -np.mean(score_crossval_ENR_array)
            print('LearnCurve_ENR_avg_score = ', LearnCurve_ENR_avg_score)
            #Append the latest averaged MAE and implemented sample size to their respective arrays for saving
            LearnCurve_ENR_avg_score_array.append(LearnCurve_ENR_avg_score)
            print('LearnCurve_ENR_avg_score_array = ', LearnCurve_ENR_avg_score_array)
            runtime_ENR_array.append(runtime_ENR)
            print('runtime_ENR = ', runtime_ENR)
            sample_size_ENR_implemented_array.append(sample_size_ENR)
            np.savetxt('Results/' + energy_method + '_sample_size_ENR_implemented_array.csv', sample_size_ENR_implemented_array, delimiter=',')
            np.savetxt('Results/' + energy_method + '_LearnCurve_ENR_avg_score_array.csv', LearnCurve_ENR_avg_score_array, delimiter=',')
            np.savetxt('Results/' + energy_method + '_runtime_ENR_array.csv', runtime_ENR_array, delimiter=',')
        except:
         break

    # Train and cross-validate with various sample sizes - End here

    #Load the arrays of implemented sample sizes, MAE and model cross-validation run time values for plotting
    sample_size_ENR_implemented_array = np.loadtxt('Results/' + energy_method + '_sample_size_ENR_implemented_array.csv', delimiter=',')
    LearnCurve_ENR_avg_score_array = np.loadtxt('Results/' + energy_method + '_LearnCurve_ENR_avg_score_array.csv', delimiter=',')
    runtime_ENR_array = np.loadtxt('Results/' + energy_method + '_runtime_ENR_array.csv', delimiter=',')

    #Plot the learning curve
    plt.figure()
    plt.loglog(sample_size_ENR_implemented_array, LearnCurve_ENR_avg_score_array)
    plt.title('ENR learning Curve with MAE for ' + energy_method + ' energy')
    plt.xlabel('log of sample size')
    plt.ylabel('log of mean absolute error')
    plt.savefig(str('Results/Jan21_MAE_LearnCurve_ENR_' + energy_method + '.png'), bbox_inches='tight')

    #Plot the time complexity curve
    plt.figure()
    plt.plot(sample_size_ENR_implemented_array, runtime_ENR_array)
    plt.title('ENR time complexity for ' + energy_method + ' energy')
    plt.xlabel('sample size')
    plt.ylabel('time taken to run ENR')
    plt.savefig(str('Results/Jan21_runtime_ENR_' + energy_method + '.png'), bbox_inches='tight')

LearnCurve('DFTB', SLATM_rep_full, energy_DFTB_full)
LearnCurve('ZINDO', SLATM_rep_full, energy_ZINDO_full)

# Define and call a function to plot the learning curve for increasing number of samples - End here
