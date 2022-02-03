import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


'''
########################################################################################################################
all_tcs_f = list of features for every test case
plots for all testcases a histogram of every feature
########################################################################################################################
'''
def create_histograms(all_tcs_f):
    tc1_features, tc2_features, tc3_features, tc4_features = all_tcs_f
    channel_names = ['analog_in', 'IM_MIS', 'IP_MIS', 'ITR_MIS', 'SCINT_MIS']
    for c_count, c_name in enumerate(channel_names):
        Path("histograms\{name}".format(name=c_name)).mkdir(exist_ok=True)
        print(c_count)
        c_count = 1
        for f in tc1_features[0].columns:
            plt.figure(figsize=(9, 16))
            plt.subplots_adjust(hspace=0.5)
            plt.suptitle("{channel}: {feature}".format(channel=c_name, feature=f), fontsize=18, y=0.95)

            ax1 = plt.subplot(4, 1, 1)
            ax1.set_title("tc1")
            plt.hist(tc1_features[c_count][f], 250)
            ax2 = plt.subplot(4, 1, 2)
            ax2.set_title("tc2")
            plt.hist(tc2_features[c_count][f], 250)
            ax3 = plt.subplot(4, 1, 3)
            ax3.set_title("tc3")
            plt.hist(tc3_features[c_count][f], 250)
            ax4 = plt.subplot(4, 1, 4)
            ax4.set_title("tc4")
            plt.hist(tc4_features[c_count][f], 250)

            xlim = (min(ax1.get_xlim()[0], ax2.get_xlim()[0],ax3.get_xlim()[0],ax4.get_xlim()[0]), max(ax1.get_xlim()[1], ax2.get_xlim()[1],ax3.get_xlim()[1],ax4.get_xlim()[1]))
            ylim = (min(ax1.get_ylim()[0], ax2.get_ylim()[0],ax3.get_ylim()[0],ax4.get_ylim()[0]), max(ax1.get_ylim()[1], ax2.get_ylim()[1],ax3.get_ylim()[1],ax4.get_ylim()[1]))

            plt.setp(ax1, xlim=xlim, ylim=ylim, yscale="log", xscale="log")
            plt.setp(ax2,xlim=xlim, ylim=ylim, yscale="log", xscale="log")
            plt.setp(ax3, xlim=xlim, ylim=ylim, yscale="log", xscale="log")
            plt.setp(ax4, xlim=xlim, ylim=ylim, yscale="log", xscale="log")
            plt.savefig("histograms/{channel}/{name}.png".format(channel=c_name,name=f))
            plt.close()

            plt.close()
def corr_heatmap(df):
    #taken from https://datagy.io/python-correlation-matrix/
    corr_matrix = df.corr().round(2)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(16, 9))
    sns.set(font_scale=1.3)
    sns.heatmap(corr_matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag', mask=mask)
    plt.yticks(rotation=90)
    plt.show()
    #plt.savefig("tc3_3_corr.png")

'''
tc_features_dfs = array of features
tc_id = used only for naming file
calculates the mean of different configurations of a feature and saves the result into a file.
Columns hardcoded not reusable for other features
'''
def create_mean_df(tc_features_dfs, tc_id, storage_path="feature_mean_tc_"):
    c3_lag = [12, 16]
    number_peaks = [16, 21]
    fft_coeff_real = [21, 26]
    fft_coeff_imag = [26, 32]
    fft_coeff_abs = [32, 38]
    fft_coeff_angle = [38, 44]
    fft_coeff_rval_max = np.arange(44, 53, 4)
    fft_coeff_rval_min = np.arange(45, 54, 4)
    fft_coeff_rval_mean = np.arange(46, 55, 4)
    fft_coeff_rval_var = np.arange(47, 56, 4)

    fft_coeff_intercept_max = np.arange(56, 65, 4)
    fft_coeff_intercept_min = np.arange(57, 66, 4)
    fft_coeff_intercept_mean = np.arange(58, 67, 4)
    fft_coeff_intercept_var = np.arange(59, 68, 4)

    fft_coeff_slope_max = np.arange(68, 77, 4)
    fft_coeff_slope_min = np.arange(69, 78, 4)
    fft_coeff_slope_mean = np.arange(70, 79, 4)
    fft_coeff_slope_var = np.arange(71, 80, 4)

    fft_coeff_stderr_max = np.arange(80, 89, 4)
    fft_coeff_stderr_min = np.arange(81, 90, 4)
    fft_coeff_stderr_mean = np.arange(82, 91, 4)
    fft_coeff_stderr_var = np.arange(83, 92, 4)

    index_mass_quantile = [92, 100]

    spkt_welch_density = [100, 103]
    ar_coefficient = [103, 108]

    energy_ratio = [108, 118]

    ratio_beyond_r_sigma = [118, 128]

    need_mean = [c3_lag, number_peaks, fft_coeff_real, fft_coeff_imag, fft_coeff_abs, fft_coeff_angle, fft_coeff_rval_max, fft_coeff_rval_min, fft_coeff_rval_mean, fft_coeff_rval_var, fft_coeff_intercept_max,
             fft_coeff_intercept_min, fft_coeff_intercept_mean, fft_coeff_intercept_var, fft_coeff_slope_max, fft_coeff_slope_min, fft_coeff_slope_mean,
             fft_coeff_slope_var, fft_coeff_stderr_max, fft_coeff_stderr_min, fft_coeff_stderr_mean, fft_coeff_stderr_var, index_mass_quantile, spkt_welch_density,
             ar_coefficient, energy_ratio, ratio_beyond_r_sigma]

    names = ["c3_lag", "number_peaks", "fft_coeff_real", "fft_coeff_imag", "fft_coeff_abs", "fft_coeff_angle", "fft_coeff_rval_max", "fft_coeff_rval_min", "fft_coeff_rval_mean", "fft_coeff_rval_var", "fft_coeff_intercept_max",
             "fft_coeff_intercept_min", "fft_coeff_intercept_mean", "fft_coeff_intercept_var", "fft_coeff_slope_max", "fft_coeff_slope_min", "fft_coeff_slope_mean",
             "fft_coeff_slope_var", "fft_coeff_stderr_max", "fft_coeff_stderr_min", "fft_coeff_stderr_mean", "fft_coeff_stderr_var", "index_mass_quantile", "spkt_welch_density",
             "ar_coefficient", "energy_ratio", "ratio_beyond_r_sigma"]

    for count, df in enumerate(tc_features_dfs):
        for c, mean_ids in enumerate(need_mean):
            if len(mean_ids) == 2:
                if mean_ids[0]==118:
                    df.iloc[:, mean_ids[0]] = df.iloc[:, mean_ids[0]:mean_ids[1]].mean(axis=1)
                    df.iloc[:, (mean_ids[0]+1):mean_ids[1]] = np.nan
            elif len(mean_ids) == 3:
                df.iloc[:, mean_ids[0]] = df.iloc[:, mean_ids].mean(axis=1)
                df.iloc[:, mean_ids[1::]] = np.nan
            else:
                assert False
            df.columns.values[mean_ids[0]] = names[c] + "_mean"
        df = df.dropna(axis='columns', how='all')
        df.to_pickle(storage_path+str(tc_id)+ "_f_" + str(count) + ".pkl")
if __name__ == '__main__':
    tc_set_name = "mean"
    file_name = 'feature_'+tc_set_name+'_tc_'
    tc1_features = []
    tc2_features = []
    tc3_features = []
    tc4_features = []
    for i in range(5):
        with open("D:\\project_data_pre\\" + file_name +"0_f_" + str(i) + ".pkl", "rb") as fh:
            tc1 = pickle.load(fh)
        tc1_features.append(tc1)
        with open("D:\\project_data_pre\\" + file_name +"1_f_"+ str(i) + ".pkl", "rb") as fh:
            tc2 = pickle.load(fh)
        tc2_features.append(tc2)
        with open("D:\\project_data_pre\\" + file_name +"2_f_" + str(i) + ".pkl", "rb") as fh:
            tc3 = pickle.load(fh)
        tc3_features.append(tc3)
        with open("D:\\project_data_pre\\" + file_name +"3_f_" + str(i) + ".pkl", "rb") as fh:
            tc4 = pickle.load(fh)
        tc4_features.append(tc4)

    all_tc_features = [tc1_features, tc2_features, tc3_features, tc4_features]
    '''calculate mean of features with different configurations'''
    #for c, tc in enumerate(tcs):
    #   create_mean_df(tc, c)

    create_histograms(all_tc_features)


    '''plot corrilation map of selected features need to input wanted testcase channel'''
    #selected = ['c3_lag_mean', 'value__variance', 'fft_coeff_real_mean']
    #corr_heatmap(tc3_features[2][selected])



