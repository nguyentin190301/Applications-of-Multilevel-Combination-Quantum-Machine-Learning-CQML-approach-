import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import glob
import random

from sklearn.linear_model import BayesianRidge

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


#Optimize alpha_1, alpha_2, lambda_1, lambda_2 (hyperparameters) - Start here

def ParamOpt_alpha_1(energy_method, input, output):
    alpha_1_try_BRR = np.logspace(-26, -19, num=8, base=2) #try eight values from roughly 2^-26 (10^-8) to 2^-19 (10^-6)
    #Initialize the arrays of hyperparameter values to try out and their corresponding mean absolute error (MAE)
    alpha_1_try_implemented_BRR = []
    alpha_1Opt_MAE_average_BRR_array = []
    #Loop over potential hyperparameter values and build a corresponding model
    for alpha_1_BRR in alpha_1_try_BRR:
        #Initialize the Bayesian Ridge Regression model
        BR_Regressor = BayesianRidge(alpha_1=alpha_1_BRR)
        #Apply 5-fold cross-validation on the RFR model with negative MAE as the score, then get the MAE
        neg_MAE_crossval_kfold_BRR = cross_val_score(BR_Regressor, input, output, cv=5, scoring='neg_mean_absolute_error')
        MAE_average_BRR = np.mean(neg_MAE_crossval_kfold_BRR)*(-1)
        print('MAE_average_BRR = ', MAE_average_BRR)
        #Append the successfully implemented hyperparameter and corresponding MAE to their arrays for saving
        alpha_1_try_implemented_BRR.append(alpha_1_BRR)
        alpha_1Opt_MAE_average_BRR_array.append(MAE_average_BRR)
        # Save the latest arrays of hyperparameter values and their corresponding MAE
        np.savetxt('Results/' + energy_method + '_alpha_1_try_implemented_BRR.csv', alpha_1_try_implemented_BRR, delimiter=',')
        np.savetxt('Results/' + energy_method + '_alpha_1Opt_MAE_average_BRR_array.csv', alpha_1Opt_MAE_average_BRR_array, delimiter=',')

    #Find the index of the hyperparameter value which returns the lowest MAE
    index_alpha_1_crossval_minMAE_BRR = np.where(alpha_1Opt_MAE_average_BRR_array == np.min(alpha_1Opt_MAE_average_BRR_array))[0][0] #can take twice component with index 0 because the index returned as a list of a 1-member array
    print('index_alpha_1_crossval_minMAE_BRR = ', index_alpha_1_crossval_minMAE_BRR)
    #Find the corresponding optimized hyperparameter
    alpha_1_crossval_optimized_BRR = alpha_1_try_BRR[index_alpha_1_crossval_minMAE_BRR]
    print('alpha_1_crossval_optimized_BRR =', alpha_1_crossval_optimized_BRR)
    #Save the optimized hyperparameter
    np.savetxt('Results/' + energy_method + '_alpha_1_crossval_optimized_BRR.csv', [alpha_1_crossval_optimized_BRR], delimiter=',')

    #Plot optimization curve - Start here
    alpha_1_try_implemented_BRR = np.loadtxt('Results/' + energy_method + '_alpha_1_try_implemented_BRR.csv',  delimiter=',')
    alpha_1Opt_MAE_average_BRR_array = np.loadtxt('Results/' + energy_method + '_alpha_1Opt_MAE_average_BRR_array.csv', delimiter=',')

    plt.figure()
    plt.loglog(alpha_1_try_implemented_BRR, alpha_1Opt_MAE_average_BRR_array)
    plt.title('BRR cross-validation MAE for alpha_1 with ' + energy_method)
    plt.xlabel('log of alpha_1')
    plt.ylabel('log of MAE')
    plt.savefig(str('Results/Jan21_crossval_MAE_alpha_1_BRR_' + energy_method + '.png'), bbox_inches='tight')
    # Plot optimization curve - End here

ParamOpt_alpha_1('DFTB', SLATM_rep_sub, energy_DFTB_sub)
ParamOpt_alpha_1('ZINDO', SLATM_rep_sub, energy_ZINDO_sub)



def ParamOpt_alpha_2(energy_method, input, output):
    alpha_2_try_BRR = np.logspace(-26, -19, num=8, base=2) #try eight values from roughly 2^-26 (10^-8) to 2^-19 (10^-6)
    #Initialize the arrays of hyperparameter values to try out and their corresponding mean absolute error (MAE)
    alpha_2_try_implemented_BRR = []
    alpha_2Opt_MAE_average_BRR_array = []
    #Loop over potential hyperparameter values and build a corresponding model
    for alpha_2_BRR in alpha_2_try_BRR:
        #Initialize the Bayesian Ridge Regression model
        BR_Regressor = BayesianRidge(alpha_2=alpha_2_BRR)
        #Apply 5-fold cross-validation on the RFR model with negative MAE as the score, then get the MAE
        neg_MAE_crossval_kfold_BRR = cross_val_score(BR_Regressor, input, output, cv=5, scoring='neg_mean_absolute_error')
        MAE_average_BRR = np.mean(neg_MAE_crossval_kfold_BRR)*(-1)
        print('MAE_average_BRR = ', MAE_average_BRR)
        #Append the successfully implemented hyperparameter and corresponding MAE to their arrays for saving
        alpha_2_try_implemented_BRR.append(alpha_2_BRR)
        alpha_2Opt_MAE_average_BRR_array.append(MAE_average_BRR)
        # Save the latest arrays of hyperparameter values and their corresponding MAE
        np.savetxt('Results/' + energy_method + '_alpha_2_try_implemented_BRR.csv', alpha_2_try_implemented_BRR, delimiter=',')
        np.savetxt('Results/' + energy_method + '_alpha_2Opt_MAE_average_BRR_array.csv', alpha_2Opt_MAE_average_BRR_array, delimiter=',')

    #Find the index of the hyperparameter value which returns the lowest MAE
    index_alpha_2_crossval_minMAE_BRR = np.where(alpha_2Opt_MAE_average_BRR_array == np.min(alpha_2Opt_MAE_average_BRR_array))[0][0] #can take twice component with index 0 because the index returned as a list of a 1-member array
    print('index_alpha_2_crossval_minMAE_BRR = ', index_alpha_2_crossval_minMAE_BRR)
    #Find the corresponding optimized hyperparameter
    alpha_2_crossval_optimized_BRR = alpha_2_try_BRR[index_alpha_2_crossval_minMAE_BRR]
    print('alpha_2_crossval_optimized_BRR =', alpha_2_crossval_optimized_BRR)
    #Save the optimized hyperparameter
    np.savetxt('Results/' + energy_method + '_alpha_2_crossval_optimized_BRR.csv', [alpha_2_crossval_optimized_BRR], delimiter=',')

    #Plot optimization curve - Start here
    alpha_2_try_implemented_BRR = np.loadtxt('Results/' + energy_method + '_alpha_2_try_implemented_BRR.csv',  delimiter=',')
    alpha_2Opt_MAE_average_BRR_array = np.loadtxt('Results/' + energy_method + '_alpha_2Opt_MAE_average_BRR_array.csv', delimiter=',')

    plt.figure()
    plt.loglog(alpha_2_try_implemented_BRR, alpha_2Opt_MAE_average_BRR_array)
    plt.title('BRR cross-validation MAE for alpha_2 with ' + energy_method)
    plt.xlabel('log of alpha_2')
    plt.ylabel('log of MAE')
    plt.savefig(str('Results/Jan21_crossval_MAE_alpha_2_BRR_' + energy_method + '.png'), bbox_inches='tight')
    # Plot optimization curve - End here

ParamOpt_alpha_2('DFTB', SLATM_rep_sub, energy_DFTB_sub)
ParamOpt_alpha_2('ZINDO', SLATM_rep_sub, energy_ZINDO_sub)


def ParamOpt_lambda_1(energy_method, input, output):
    lambda_1_try_BRR = np.logspace(-26, -19, num=8, base=2) #try eight values from roughly 2^-26 (10^-8) to 2^-19 (10^-6)
    #Initialize the arrays of hyperparameter values to try out and their corresponding mean absolute error (MAE)
    lambda_1_try_implemented_BRR = []
    lambda_1Opt_MAE_average_BRR_array = []
    #Loop over potential hyperparameter values and build a corresponding model
    for lambda_1_BRR in lambda_1_try_BRR:
        #Initialize the Bayesian Ridge Regression model
        BR_Regressor = BayesianRidge(lambda_1=lambda_1_BRR)
        #Apply 5-fold cross-validation on the RFR model with negative MAE as the score, then get the MAE
        neg_MAE_crossval_kfold_BRR = cross_val_score(BR_Regressor, input, output, cv=5, scoring='neg_mean_absolute_error')
        MAE_average_BRR = np.mean(neg_MAE_crossval_kfold_BRR)*(-1)
        print('MAE_average_BRR = ', MAE_average_BRR)
        #Append the successfully implemented hyperparameter and corresponding MAE to their arrays for saving
        lambda_1_try_implemented_BRR.append(lambda_1_BRR)
        lambda_1Opt_MAE_average_BRR_array.append(MAE_average_BRR)
        # Save the latest arrays of hyperparameter values and their corresponding MAE
        np.savetxt('Results/' + energy_method + '_lambda_1_try_implemented_BRR.csv', lambda_1_try_implemented_BRR, delimiter=',')
        np.savetxt('Results/' + energy_method + '_lambda_1Opt_MAE_average_BRR_array.csv', lambda_1Opt_MAE_average_BRR_array, delimiter=',')

    #Find the index of the hyperparameter value which returns the lowest MAE
    index_lambda_1_crossval_minMAE_BRR = np.where(lambda_1Opt_MAE_average_BRR_array == np.min(lambda_1Opt_MAE_average_BRR_array))[0][0] #can take twice component with index 0 because the index returned as a list of a 1-member array
    print('index_lambda_1_crossval_minMAE_BRR = ', index_lambda_1_crossval_minMAE_BRR)
    #Find the corresponding optimized hyperparameter
    lambda_1_crossval_optimized_BRR = lambda_1_try_BRR[index_lambda_1_crossval_minMAE_BRR]
    print('lambda_1_crossval_optimized_BRR =', lambda_1_crossval_optimized_BRR)
    #Save the optimized hyperparameter
    np.savetxt('Results/' + energy_method + '_lambda_1_crossval_optimized_BRR.csv', [lambda_1_crossval_optimized_BRR], delimiter=',')

    #Plot optimization curve - Start here
    lambda_1_try_implemented_BRR = np.loadtxt('Results/' + energy_method + '_lambda_1_try_implemented_BRR.csv',  delimiter=',')
    lambda_1Opt_MAE_average_BRR_array = np.loadtxt('Results/' + energy_method + '_lambda_1Opt_MAE_average_BRR_array.csv', delimiter=',')

    plt.figure()
    plt.loglog(lambda_1_try_implemented_BRR, lambda_1Opt_MAE_average_BRR_array)
    plt.title('BRR cross-validation MAE for lambda_1 with ' + energy_method)
    plt.xlabel('log of lambda_1')
    plt.ylabel('log of MAE')
    plt.savefig(str('Results/Jan21_crossval_MAE_lambda_1_BRR_' + energy_method + '.png'), bbox_inches='tight')
    # Plot optimization curve - End here

ParamOpt_lambda_1('DFTB', SLATM_rep_sub, energy_DFTB_sub)
ParamOpt_lambda_1('ZINDO', SLATM_rep_sub, energy_ZINDO_sub)


def ParamOpt_lambda_2(energy_method, input, output):
    lambda_2_try_BRR = np.logspace(-26, -19, num=8, base=2) #try eight values from roughly 2^-26 (10^-8) to 2^-19 (10^-6)
    #Initialize the arrays of hyperparameter values to try out and their corresponding mean absolute error (MAE)
    lambda_2_try_implemented_BRR = []
    lambda_2Opt_MAE_average_BRR_array = []
    #Loop over potential hyperparameter values and build a corresponding model
    for lambda_2_BRR in lambda_2_try_BRR:
        #Initialize the Bayesian Ridge Regression model
        BR_Regressor = BayesianRidge(lambda_2=lambda_2_BRR)
        #Apply 5-fold cross-validation on the RFR model with negative MAE as the score, then get the MAE
        neg_MAE_crossval_kfold_BRR = cross_val_score(BR_Regressor, input, output, cv=5, scoring='neg_mean_absolute_error')
        MAE_average_BRR = np.mean(neg_MAE_crossval_kfold_BRR)*(-1)
        print('MAE_average_BRR = ', MAE_average_BRR)
        #Append the successfully implemented hyperparameter and corresponding MAE to their arrays for saving
        lambda_2_try_implemented_BRR.append(lambda_2_BRR)
        lambda_2Opt_MAE_average_BRR_array.append(MAE_average_BRR)
        # Save the latest arrays of hyperparameter values and their corresponding MAE
        np.savetxt('Results/' + energy_method + '_lambda_2_try_implemented_BRR.csv', lambda_2_try_implemented_BRR, delimiter=',')
        np.savetxt('Results/' + energy_method + '_lambda_2Opt_MAE_average_BRR_array.csv', lambda_2Opt_MAE_average_BRR_array, delimiter=',')

    #Find the index of the hyperparameter value which returns the lowest MAE
    index_lambda_2_crossval_minMAE_BRR = np.where(lambda_2Opt_MAE_average_BRR_array == np.min(lambda_2Opt_MAE_average_BRR_array))[0][0] #can take twice component with index 0 because the index returned as a list of a 1-member array
    print('index_lambda_2_crossval_minMAE_BRR = ', index_lambda_2_crossval_minMAE_BRR)
    #Find the corresponding optimized hyperparameter
    lambda_2_crossval_optimized_BRR = lambda_2_try_BRR[index_lambda_2_crossval_minMAE_BRR]
    print('lambda_2_crossval_optimized_BRR =', lambda_2_crossval_optimized_BRR)
    #Save the optimized hyperparameter
    np.savetxt('Results/' + energy_method + '_lambda_2_crossval_optimized_BRR.csv', [lambda_2_crossval_optimized_BRR], delimiter=',')

    #Plot optimization curve - Start here
    lambda_2_try_implemented_BRR = np.loadtxt('Results/' + energy_method + '_lambda_2_try_implemented_BRR.csv',  delimiter=',')
    lambda_2Opt_MAE_average_BRR_array = np.loadtxt('Results/' + energy_method + '_lambda_2Opt_MAE_average_BRR_array.csv', delimiter=',')

    plt.figure()
    plt.loglog(lambda_2_try_implemented_BRR, lambda_2Opt_MAE_average_BRR_array)
    plt.title('BRR cross-validation MAE for lambda_2 with ' + energy_method)
    plt.xlabel('log of lambda_2')
    plt.ylabel('log of MAE')
    plt.savefig(str('Results/Jan21_crossval_MAE_lambda_2_BRR_' + energy_method + '.png'), bbox_inches='tight')
    # Plot optimization curve - End here

ParamOpt_lambda_2('DFTB', SLATM_rep_sub, energy_DFTB_sub)
ParamOpt_lambda_2('ZINDO', SLATM_rep_sub, energy_ZINDO_sub)

#Optimize alpha_1, alpha_2, lambda_1, lambda_2 (hyperparameters) - End here




# Define and call a function to plot the learning curve for increasing number of samples - Start here
import time
def LearnCurve(energy_method, input_full, output_full):
    #Define an array of potential sample sizes for the learning curves
    sample_size_BRR_array = np.logspace(5, 13, num=9, base=2)
    print('sample_size_BRR_array = ', sample_size_BRR_array)
    #Train and cross-validate with various sample size - Start here

    #Load the optimized hyperparameter to feed the learning-curve-construction model
    alpha_1_crossval_optimized_BRR = int(np.loadtxt('Results/' + energy_method + '_alpha_1_crossval_optimized_BRR.csv', delimiter=','))
    alpha_2_crossval_optimized_BRR = int(np.loadtxt('Results/' + energy_method + '_alpha_2_crossval_optimized_BRR.csv', delimiter=','))
    lambda_1_crossval_optimized_BRR = int(np.loadtxt('Results/' + energy_method + '_lambda_1_crossval_optimized_BRR.csv', delimiter=','))
    lambda_2_crossval_optimized_BRR = int(np.loadtxt('Results/' + energy_method + '_lambda_2_crossval_optimized_BRR.csv', delimiter=','))
    BRRmodel_LearnCurve = BayesianRidge(alpha_1=alpha_1_crossval_optimized_BRR, alpha_2=alpha_2_crossval_optimized_BRR, lambda_1=lambda_1_crossval_optimized_BRR, lambda_2=lambda_2_crossval_optimized_BRR)
    #Initialize the arrays to collect each implemented sample size and its corresponding positive MAE, run time
    LearnCurve_BRR_avg_score_array = []
    runtime_BRR_array = []
    sample_size_BRR_implemented_array = []
    #Apply the model on different sample sizes
    for sample_size_BRR in sample_size_BRR_array:
         #Use try-except with break to stop the loop when the training/cross-validation size becomes too large
        try:
            sample_size_BRR = int(sample_size_BRR)
            #Cut the full input and output arrays into smaller arrays of the sample size
            input = input_full[0:sample_size_BRR, :]
            output = output_full[0:sample_size_BRR]
            #Record the time before and after running the model 5-fold cross validation to calculate their difference for the time complexity curve
            start_BRR = time.time()
            score_crossval_BRR_array = cross_val_score(BRRmodel_LearnCurve, input, output, cv=5, scoring='neg_mean_absolute_error')
            end_BRR = time.time()
            runtime_BRR = end_BRR - start_BRR

            print('score_crossval_BRR_array = ', score_crossval_BRR_array)
            #Get the averaged positive MAE from the array of 5 MAE values returned by 5-fold cross-validation of the model
            LearnCurve_BRR_avg_score = -np.mean(score_crossval_BRR_array)
            print('LearnCurve_BRR_avg_score = ', LearnCurve_BRR_avg_score)
            #Append the latest averaged MAE and implemented sample size to their respective arrays for saving
            LearnCurve_BRR_avg_score_array.append(LearnCurve_BRR_avg_score)
            print('LearnCurve_BRR_avg_score_array = ', LearnCurve_BRR_avg_score_array)
            runtime_BRR_array.append(runtime_BRR)
            print('runtime_BRR = ', runtime_BRR)
            sample_size_BRR_implemented_array.append(sample_size_BRR)
            np.savetxt('Results/' + energy_method + '_sample_size_BRR_implemented_array.csv', sample_size_BRR_implemented_array, delimiter=',')
            np.savetxt('Results/' + energy_method + '_LearnCurve_BRR_avg_score_array.csv', LearnCurve_BRR_avg_score_array, delimiter=',')
            np.savetxt('Results/' + energy_method + '_runtime_BRR_array.csv', runtime_BRR_array, delimiter=',')
        except:
         break

    # Train and cross-validate with various sample sizes - End here

    #Load the arrays of implemented sample sizes, MAE and model cross-validation run time values for plotting
    sample_size_BRR_implemented_array = np.loadtxt('Results/' + energy_method + '_sample_size_BRR_implemented_array.csv', delimiter=',')
    LearnCurve_BRR_avg_score_array = np.loadtxt('Results/' + energy_method + '_LearnCurve_BRR_avg_score_array.csv', delimiter=',')
    runtime_BRR_array = np.loadtxt('Results/' + energy_method + '_runtime_BRR_array.csv', delimiter=',')

    #Plot the learning curve
    plt.figure()
    plt.loglog(sample_size_BRR_implemented_array, LearnCurve_BRR_avg_score_array)
    plt.title('BRR learning Curve with MAE for ' + energy_method + ' energy')
    plt.xlabel('log of sample size')
    plt.ylabel('log of mean absolute error')
    plt.savefig(str('Results/Jan21_MAE_LearnCurve_BRR_' + energy_method + '.png'), bbox_inches='tight')

    #Plot the time complexity curve
    plt.figure()
    plt.plot(sample_size_BRR_implemented_array, runtime_BRR_array)
    plt.title('BRR time complexity for ' + energy_method + ' energy')
    plt.xlabel('sample size')
    plt.ylabel('time taken to run BRR')
    plt.savefig(str('Results/Jan21_runtime_BRR_' + energy_method + '.png'), bbox_inches='tight')

LearnCurve('DFTB', SLATM_rep_full, energy_DFTB_full)
LearnCurve('ZINDO', SLATM_rep_full, energy_ZINDO_full)

# Define and call a function to plot the learning curve for increasing number of samples - End here
