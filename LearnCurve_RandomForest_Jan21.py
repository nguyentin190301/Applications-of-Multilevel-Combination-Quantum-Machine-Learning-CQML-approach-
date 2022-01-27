import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import glob
import random

from sklearn.ensemble import RandomForestRegressor

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
#Optimize the number of random trees (hyperparameter) - Start here

def ParamOpt_trees(energy_method, input, output):
    trees_try_RFR = [10, 30, 90, 270, 810] # minimum MAE at trees = 270 for this array of trials
    #Initialize the arrays of hyperparameter values to try out and their corresponding mean absolute error (MAE)
    trees_try_implemented_RFR = []
    treesOpt_MAE_average_RFR_array = []
    #Loop over potential hyperparameter values and build a corresponding model
    for trees_RFR in trees_try_RFR:
        #Use try, except, break to avoid error when there is some hyperparameter the model cannot handle
        try:
            #Initialize the Random Forest Regression (RFR) model
            RF_Regressor = RandomForestRegressor(n_estimators = trees_RFR, random_state = 1)
            #Apply 5-fold cross-validation on the RFR model with negative MAE as the score, then get the MAE
            neg_MAE_crossval_kfold_RFR = cross_val_score(RF_Regressor, input, output, cv=5, scoring='neg_mean_absolute_error')
            MAE_average_RFR = np.mean(neg_MAE_crossval_kfold_RFR)*(-1)
            print('MAE_average = ', MAE_average_RFR)
            #Append the successfully implemented hyperparameter and corresponding MAE to their arrays for saving
            trees_try_implemented_RFR.append(trees_RFR)
            treesOpt_MAE_average_array_RFR.append(MAE_average_RFR)
            # Save the latest arrays of hyperparameter values and their corresponding MAE
            np.savetxt('Results/' + energy_method + '_trees_try_implemented_RFR.csv', trees_try_implemented_RFR, delimiter=',')
            np.savetxt('Results/' + energy_method + '_treesOpt_MAE_average_RFR_array.csv', treesOpt_MAE_average_RFR_array, delimiter=',')
        except:
            break

    #Find the index of the hyperparameter value which returns the lowest MAE
    index_trees_crossval_minMAE_RFR = np.where(treesOpt_MAE_average_RFR_array == np.min(treesOpt_MAE_average_RFR_array))[0][0] #can take twice component with index 0 because the index returned as a list of a 1-member array
    print('index_trees_crossval_minMAE_RFR = ', index_trees_crossval_minMAE_RFR)
    #Find the corresponding optimized hyperparameter
    trees_crossval_optimized_RFR = trees_try_RFR[index_trees_crossval_minMAE_RFR]
    print('trees_crossval_optimized_RFR =', trees_crossval_optimized_RFR)
    #Save the optimized hyperparameter
    np.savetxt('Results/' + energy_method + '_trees_crossval_optimized_RFR.csv', [trees_crossval_optimized_RFR], delimiter=',')

    #Plot optimization curve for trees - Start here
    trees_try_implemented_RFR = np.loadtxt('Results/' + energy_method + '_trees_try_implemented_RFR.csv',  delimiter=',')
    treesOpt_MAE_average_RFR_array = np.loadtxt('Results/' + energy_method + '_treesOpt_MAE_average_RFR_array.csv', delimiter=',')

    plt.figure()
    plt.loglog(trees_try_implemented_RFR, treesOpt_MAE_average_RFR_array)
    plt.title('RFR cross-validation MAE for trees with ' + energy_method)
    plt.xlabel('log of trees')
    plt.ylabel('log of MAE')
    plt.savefig(str('Results/Jan21_crossval_MAE_treesOpt_RFR_' + energy_method + '.png'), bbox_inches='tight')
    # Plot optimization curve for trees - End here

ParamOpt_trees('DFTB', SLATM_rep_sub, energy_DFTB_sub)
ParamOpt_trees('ZINDO', SLATM_rep_sub, energy_ZINDO_sub)

#Optimize the number of random trees (hyperparameter) - End here
'''

# Define and call a function to plot the learning curve for increasing number of samples - Start here
import time
def LearnCurve(energy_method, input_full, output_full):
    #Define an array of potential sample sizes for the learning curves
    sample_size_RFR_array = np.logspace(5, 13, num=9, base=2)
    print('sample_size_RFR_array = ', sample_size_RFR_array)
    #Train and cross-validate with various sample size - Start here

    #Load the optimized hyperparameter to feed the learning-curve-construction model
    trees_crossval_optimized_RFR = int(np.loadtxt('Results/' + energy_method + '_trees_crossval_optimized_RFR.csv', delimiter=','))
    RFRmodel_LearnCurve = RandomForestRegressor(n_estimators=trees_crossval_optimized)
    #Initialize the arrays to collect each implemented sample size and its corresponding positive MAE, run time
    LearnCurve_RFR_avg_score_array = []
    runtime_RFR_array = []
    sample_size_RFR_implemented_array = []
    #Apply the model on different sample sizes
    for sample_size_RFR in sample_size_RFR_array:
         #Use try-except with break to stop the loop when the training/cross-validation size becomes too large
        try:
            sample_size_RFR = int(sample_size_RFR)
            #Cut the full input and output arrays into smaller arrays of the sample size
            input = input_full[0:sample_size_RFR, :]
            output = output_full[0:sample_size_RFR]
            #Record the time before and after running the model 5-fold cross validation to calculate their difference for the time complexity curve
            start_RFR = time.time()
            score_crossval_RFR_array = cross_val_score(RFRmodel_LearnCurve, input, output, cv=5, scoring='neg_mean_absolute_error')
            end_RFR = time.time()
            runtime_RFR = end_RFR - start_RFR

            print('score_crossval_RFR_array = ', score_crossval_RFR_array)
            #Get the averaged positive MAE from the array of 5 MAE values returned by 5-fold cross-validation of the model
            LearnCurve_RFR_avg_score = -np.mean(score_crossval_RFR_array)
            print('LearnCurve_RFR_avg_score = ', LearnCurve_RFR_avg_score)
            #Append the latest averaged MAE and implemented sample size to their respective arrays for saving
            LearnCurve_RFR_avg_score_array.append(LearnCurve_RFR_avg_score)
            print('LearnCurve_RFR_avg_score_array = ', LearnCurve_RFR_avg_score_array)
            runtime_RFR_array.append(runtime_RFR)
            print('runtime_RFR = ', runtime_RFR)
            sample_size_RFR_implemented_array.append(sample_size_RFR)
            np.savetxt('Results/' + energy_method + '_sample_size_RFR_implemented_array.csv', sample_size_RFR_implemented_array, delimiter=',')
            np.savetxt('Results/' + energy_method + '_LearnCurve_RFR_avg_score_array.csv', LearnCurve_RFR_avg_score_array, delimiter=',')
            np.savetxt('Results/' + energy_method + '_runtime_RFR_array.csv', runtime_RFR_array, delimiter=',')
        except:
         break

    # Train and cross-validate with various sample sizes - End here

    #Load the arrays of implemented sample sizes, MAE and model cross-validation run time values for plotting
    sample_size_RFR_implemented_array = np.loadtxt('Results/' + energy_method + '_sample_size_RFR_implemented_array.csv', delimiter=',')
    LearnCurve_RFR_avg_score_array = np.loadtxt('Results/' + energy_method + '_LearnCurve_RFR_avg_score_array.csv', delimiter=',')
    runtime_RFR_array = np.loadtxt('Results/' + energy_method + '_runtime_RFR_array.csv', delimiter=',')

    #Plot the learning curve
    plt.figure()
    plt.loglog(sample_size_RFR_implemented_array, LearnCurve_RFR_avg_score_array)
    plt.title('RFR learning Curve with MAE for ' + energy_method + ' energy')
    plt.xlabel('log of sample size')
    plt.ylabel('log of mean absolute error')
    plt.savefig(str('Results/Jan21_MAE_LearnCurve_RFR_' + energy_method + '.png'), bbox_inches='tight')

    #Plot the time complexity curve
    plt.figure()
    plt.plot(sample_size_RFR_implemented_array, runtime_RFR_array)
    plt.title('RFR time complexity for ' + energy_method + ' energy')
    plt.xlabel('sample size')
    plt.ylabel('time taken to run RFR')
    plt.savefig(str('Results/Jan21_runtime_RFR_' + energy_method + '.png'), bbox_inches='tight')

LearnCurve('DFTB', SLATM_rep_full, energy_DFTB_full)
LearnCurve('ZINDO', SLATM_rep_full, energy_ZINDO_full)

# Define and call a function to plot the learning curve for increasing number of samples - End here