import numpy as np
import matplotlib.pyplot as plt

def Comparative_Plot_LearnCurve(energy_method):
    #Load the arrays of implemented sample sizes, MAE and model cross-validation run time values for plotting
    sample_size_KRR_implemented_array = np.loadtxt('Results/' + energy_method + '_sample_size_KRR_implemented_array.csv', delimiter=',')
    LearnCurve_KRR_avg_score_array = np.loadtxt('Results/' + energy_method + '_LearnCurve_KRR_avg_score_array.csv', delimiter=',')
    runtime_KRR_array = np.loadtxt('Results/' + energy_method + '_runtime_KRR_array.csv', delimiter=',')

    sample_size_BRR_implemented_array = np.loadtxt('Results/' + energy_method + '_sample_size_BRR_implemented_array.csv', delimiter=',')
    LearnCurve_BRR_avg_score_array = np.loadtxt('Results/' + energy_method + '_LearnCurve_BRR_avg_score_array.csv', delimiter=',')
    runtime_BRR_array = np.loadtxt('Results/' + energy_method + '_runtime_BRR_array.csv', delimiter=',')

    sample_size_ENR_implemented_array = np.loadtxt('Results/' + energy_method + '_sample_size_ENR_implemented_array.csv', delimiter=',')
    LearnCurve_ENR_avg_score_array = np.loadtxt('Results/' + energy_method + '_LearnCurve_ENR_avg_score_array.csv', delimiter=',')
    runtime_ENR_array = np.loadtxt('Results/' + energy_method + '_runtime_ENR_array.csv', delimiter=',')

    sample_size_RFR_implemented_array = np.loadtxt('Results/' + energy_method + '_sample_size_RFR_implemented_array.csv', delimiter=',')
    LearnCurve_RFR_avg_score_array = np.loadtxt('Results/' + energy_method + '_LearnCurve_RFR_avg_score_array.csv', delimiter=',')
    runtime_RFR_array = np.loadtxt('Results/' + energy_method + '_runtime_RFR_array.csv', delimiter=',')

    #Plot the learning curve
    plt.figure()
    plt.loglog(sample_size_KRR_implemented_array, LearnCurve_KRR_avg_score_array, label='KRR', marker='o')
    plt.loglog(sample_size_BRR_implemented_array, LearnCurve_BRR_avg_score_array, label='BRR', marker='^')
    plt.loglog(sample_size_ENR_implemented_array, LearnCurve_ENR_avg_score_array, label='ENR', marker='x')
    plt.loglog(sample_size_RFR_implemented_array, LearnCurve_RFR_avg_score_array, label='RFR', marker='*')
    plt.title('Learning Curve with MAE for ' + energy_method + ' energy')
    plt.xlabel('log of sample size')
    plt.ylabel('log of mean absolute error')
    plt.legend()
    plt.savefig(str('Results/Jan21_MAE_LearnCurve_Comparative_' + energy_method + '.png'), bbox_inches='tight')

    #Plot the time complexity curve
    plt.figure()
    plt.plot(sample_size_KRR_implemented_array, runtime_KRR_array, label='KRR', marker='o')
    plt.plot(sample_size_BRR_implemented_array, runtime_BRR_array, label='BRR', marker='^')
    plt.plot(sample_size_ENR_implemented_array, runtime_ENR_array, label='ENR', marker='x')
    plt.plot(sample_size_RFR_implemented_array, runtime_RFR_array, label='RFR', marker='*')
    plt.title('Time complexity for ' + energy_method + ' energy')
    plt.xlabel('sample size')
    plt.ylabel('time taken to run the method')
    plt.legend()
    plt.savefig(str('Results/Jan21_runtime_Comparative_' + energy_method + '.png'), bbox_inches='tight')

Comparative_Plot_LearnCurve('DFTB')
Comparative_Plot_LearnCurve('ZINDO')