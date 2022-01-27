import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import glob
import random

from sklearn.kernel_ridge import KernelRidge

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


#Optimize alpha (kernel scaling hyperparameter) - Start here

def ParamOpt_alpha(energy_method, input, output):
    alpha_try_KRR = np.logspace(-25, -10, num=16, base=2) #try eight values from roughly 2^-26 (10^-8) to 2^-19 (10^-6)
    #Initialize the arrays of hyperparameter values to try out and their corresponding mean absolute error (MAE)
    alpha_try_implemented_KRR = []
    alphaOpt_MAE_average_KRR_array = []
    #Loop over potential hyperparameter values and build a corresponding model
    for alpha_KRR in alpha_try_KRR:
        #Initialize the Kernel Ridge Regression model
        KR_Regressor = KernelRidge(gamma=10**(-7), alpha=alpha_KRR, kernel='laplacian')
        #Apply 5-fold cross-validation on the RFR model with negative MAE as the score, then get the MAE
        neg_MAE_crossval_kfold_KRR = cross_val_score(KR_Regressor, input, output, cv=5, scoring='neg_mean_absolute_error')
        MAE_average_KRR = np.mean(neg_MAE_crossval_kfold_KRR)*(-1)
        print('MAE_average_KRR = ', MAE_average_KRR)
        #Append the successfully implemented hyperparameter and corresponding MAE to their arrays for saving
        alpha_try_implemented_KRR.append(alpha_KRR)
        alphaOpt_MAE_average_KRR_array.append(MAE_average_KRR)
        # Save the latest arrays of hyperparameter values and their corresponding MAE
        np.savetxt('Results/' + energy_method + '_alpha_try_implemented_KRR.csv', alpha_try_implemented_KRR, delimiter=',')
        np.savetxt('Results/' + energy_method + '_alphaOpt_MAE_average_KRR_array.csv', alphaOpt_MAE_average_KRR_array, delimiter=',')

    #Find the index of the hyperparameter value which returns the lowest MAE
    index_alpha_crossval_minMAE_KRR = np.where(alphaOpt_MAE_average_KRR_array == np.min(alphaOpt_MAE_average_KRR_array))[0][0] #can take twice component with index 0 because the index returned as a list of a 1-member array
    print('index_alpha_crossval_minMAE_KRR = ', index_alpha_crossval_minMAE_KRR)
    #Find the corresponding optimized hyperparameter
    alpha_crossval_optimized_KRR = alpha_try_KRR[index_alpha_crossval_minMAE_KRR]
    print('alpha_crossval_optimized_KRR =', alpha_crossval_optimized_KRR)
    #Save the optimized hyperparameter
    np.savetxt('Results/' + energy_method + '_alpha_crossval_optimized_KRR.csv', [alpha_crossval_optimized_KRR], delimiter=',')

    #Plot optimization curve - Start here
    alpha_try_implemented_KRR = np.loadtxt('Results/' + energy_method + '_alpha_try_implemented_KRR.csv',  delimiter=',')
    alphaOpt_MAE_average_KRR_array = np.loadtxt('Results/' + energy_method + '_alphaOpt_MAE_average_KRR_array.csv', delimiter=',')

    plt.figure()
    plt.loglog(alpha_try_implemented_KRR, alphaOpt_MAE_average_KRR_array)
    plt.title('KRR cross-validation MAE for alpha with ' + energy_method)
    plt.xlabel('log of alpha')
    plt.ylabel('log of MAE')
    plt.savefig(str('Results/Jan21_crossval_MAE_alpha_KRR_' + energy_method + '.png'), bbox_inches='tight')
    # Plot optimization curve - End here

ParamOpt_alpha('DFTB', SLATM_rep_sub, energy_DFTB_sub)
ParamOpt_alpha('ZINDO', SLATM_rep_sub, energy_ZINDO_sub)

#Optimize alpha (kernel scaling hyperparameter) - End here



# Define and call a function to plot the learning curve for increasing number of samples - Start here
import time
def LearnCurve(energy_method, input_full, output_full):
    '''
    #Define an array of potential sample sizes for the learning curves
    sample_size_KRR_array = np.logspace(5, 13, num=9, base=2)
    print('sample_size_KRR_array = ', sample_size_KRR_array)
    #Train and cross-validate with various sample size - Start here

    #Load the optimized hyperparameter to feed the learning-curve-construction model
    alpha_crossval_optimized_KRR = int(np.loadtxt('Results/' + energy_method + '_alpha_crossval_optimized_KRR.csv', delimiter=','))

    KRRmodel_LearnCurve = KernelRidge(gamma = 10**(-7), alpha=alpha_crossval_optimized_KRR, kernel='laplacian')
    #Initialize the arrays to collect each implemented sample size and its corresponding positive MAE, run time
    LearnCurve_KRR_avg_score_array = []
    runtime_KRR_array = []
    sample_size_KRR_implemented_array = []
    #Apply the model on different sample sizes
    for sample_size_KRR in sample_size_KRR_array:
         #Use try-except with break to stop the loop when the training/cross-validation size becomes too large
        try:
            sample_size_KRR = int(sample_size_KRR)
            #Cut the full input and output arrays into smaller arrays of the sample size
            input = input_full[0:sample_size_KRR, :]
            output = output_full[0:sample_size_KRR]
            #Record the time before and after running the model 5-fold cross validation to calculate their difference for the time complexity curve
            start_KRR = time.time()
            score_crossval_KRR_array = cross_val_score(KRRmodel_LearnCurve, input, output, cv=5, scoring='neg_mean_absolute_error')
            end_KRR = time.time()
            runtime_KRR = end_KRR - start_KRR

            print('score_crossval_KRR_array = ', score_crossval_KRR_array)
            #Get the averaged positive MAE from the array of 5 MAE values returned by 5-fold cross-validation of the model
            LearnCurve_KRR_avg_score = -np.mean(score_crossval_KRR_array)
            print('LearnCurve_KRR_avg_score = ', LearnCurve_KRR_avg_score)
            #Append the latest averaged MAE and implemented sample size to their respective arrays for saving
            LearnCurve_KRR_avg_score_array.append(LearnCurve_KRR_avg_score)
            print('LearnCurve_KRR_avg_score_array = ', LearnCurve_KRR_avg_score_array)
            runtime_KRR_array.append(runtime_KRR)
            print('runtime_KRR = ', runtime_KRR)
            sample_size_KRR_implemented_array.append(sample_size_KRR)
            np.savetxt('Results/' + energy_method + '_sample_size_KRR_implemented_array.csv', sample_size_KRR_implemented_array, delimiter=',')
            np.savetxt('Results/' + energy_method + '_LearnCurve_KRR_avg_score_array.csv', LearnCurve_KRR_avg_score_array, delimiter=',')
            np.savetxt('Results/' + energy_method + '_runtime_KRR_array.csv', runtime_KRR_array, delimiter=',')
        except:
         break

    # Train and cross-validate with various sample sizes - End here
    '''
    #Load the arrays of implemented sample sizes, MAE and model cross-validation run time values for plotting
    sample_size_KRR_implemented_array = np.loadtxt('Results/' + energy_method + '_sample_size_KRR_implemented_array.csv', delimiter=',')
    LearnCurve_KRR_avg_score_array = np.loadtxt('Results/' + energy_method + '_LearnCurve_KRR_avg_score_array.csv', delimiter=',')
    runtime_KRR_array = np.loadtxt('Results/' + energy_method + '_runtime_KRR_array.csv', delimiter=',')

    #Plot the learning curve
    plt.figure()
    plt.loglog(sample_size_KRR_implemented_array, LearnCurve_KRR_avg_score_array)
    plt.title('KRR learning Curve with MAE for ' + energy_method + ' energy')
    plt.xlabel('log of sample size')
    plt.ylabel('log of mean absolute error')
    plt.savefig(str('Results/Jan21_MAE_LearnCurve_KRR_' + energy_method + '.png'), bbox_inches='tight')

    #Plot the time complexity curve
    plt.figure()
    plt.plot(sample_size_KRR_implemented_array, runtime_KRR_array)
    plt.title('KRR time complexity for ' + energy_method + ' energy')
    plt.xlabel('sample size')
    plt.ylabel('time taken to run KRR')
    plt.savefig(str('Results/Jan21_runtime_KRR_' + energy_method + '.png'), bbox_inches='tight')

#LearnCurve('DFTB', SLATM_rep_full, energy_DFTB_full)
LearnCurve('ZINDO', SLATM_rep_full, energy_ZINDO_full)

# Define and call a function to plot the learning curve for increasing number of samples - End here
