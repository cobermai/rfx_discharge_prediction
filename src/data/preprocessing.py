import numpy as np
import pandas as pd
from datetime import datetime
import h5py
from tsfresh import extract_features
import pickle

'''
collects all files that belong to each testcase correctly sorted
'''
def split_data_into_ec(tc_dates, data_file_dir="data/processed"):
    tc_sorted_dates = []
    for i in range(len(tc_dates)):
        temp_lst = []
        for date in tc_dates[i]:
            temp_lst.append(datetime.strptime(date, '%y%m%d%H%M'))

        temp_lst.sort()
        tc_sorted_dates.append(temp_lst)

    ec_data = [[h5py.File(data_file_dir + "hvptf_" + file_name.strftime('%y%m%d%H%M') + ".hdf5", "r") for file_name in ec] for ec in tc_sorted_dates]

    return ec_data

def collect_data_fields(ec_data, pairs, analog_in_combined, ec_id):

    im = [np.hstack([fi["ANALOG_IN"]["IM_MIS"] for fi in ec]) for ec in ec_data]
    ip = [np.hstack([fi["ANALOG_IN"]["IP_MIS"] for fi in ec]) for ec in ec_data]
    itr = [np.hstack([fi["ANALOG_IN"]["ITR_MIS"] for fi in ec]) for ec in ec_data]
    scint = [np.hstack([fi["ANALOG_IN"]["SCINT_MIS"] for fi in ec]) for ec in ec_data]
    df_final = pd.DataFrame(columns=['ANALOG_IN', 'IM_MIS', 'IP_MIS', 'ITR_MIS', 'SCINT_MIS',], index=np.arange(len(ec_data)))

    if ec_id == 1:
        analog_in_combined[ec_id] = np.append(analog_in_combined[ec_id][:19932690], analog_in_combined[ec_id][20550000:37837690])
        im[ec_id] = np.hstack((im[ec_id][:, :19932690], im[ec_id][:, 20550000:37837690]))
        ip[ec_id] = np.hstack((ip[ec_id][:, :19932690], ip[ec_id][:, 20550000:37837690]))
        itr[ec_id] = np.hstack((itr[ec_id][:, :19932690], itr[ec_id][:, 20550000:37837690]))
        scint[ec_id] = np.hstack((scint[ec_id][:, :19932690], scint[ec_id][:, 20550000:37837690]))
    if ec_id== 2:
        analog_in_combined[ec_id] = analog_in_combined[ec_id][:-650000]
        im[ec_id] = im[ec_id][:, :-650000]
        ip[ec_id] = ip[ec_id][:, :-650000]
        itr[ec_id] = itr[ec_id][:, :-650000]
        scint[ec_id] = scint[ec_id][:, :-650000]

    data_pair_lst = []
    for i, pair in enumerate(pairs[ec_id]):
        print(str(i) +" / " + str(len(pairs[ec_id])))
        analog = [[x, i] for x in analog_in_combined[ec_id][pair[0]:pair[1]]]
        im_data = [[x, i] for x in im[ec_id][0][pair[0]:pair[1]]]
        ip_data = [[x, i] for x in ip[ec_id][0][pair[0]:pair[1]]]
        itr_data = [[x, i] for x in itr[ec_id][0][pair[0]:pair[1]]]
        scint_data = [[x, i] for x in scint[ec_id][0][pair[0]:pair[1]]]
        data_pair_lst.append([analog, im_data, ip_data, itr_data, scint_data])
    df_final.at[ec_id, 'ANALOG_IN'] = [t[0] for t in data_pair_lst]
    df_final.at[ec_id, 'IM_MIS'] = [t[1] for t in data_pair_lst]
    df_final.at[ec_id, 'IP_MIS'] = [t[2] for t in data_pair_lst]
    df_final.at[ec_id, 'ITR_MIS'] = [t[3] for t in data_pair_lst]
    df_final.at[ec_id, 'SCINT_MIS'] = [t[4] for t in data_pair_lst]

    return df_final


def calculate_analog_in(data):
    vm_mis_full = [np.hstack([fi["ANALOG_IN"]["VM_MIS"] for fi in ec]) for ec in data]

    vp_mis_full = [np.hstack([fi["ANALOG_IN"]["VP_MIS"] for fi in ec]) for ec in data]
    res = []
    for ec_id in range(len(vm_mis_full)): # ValueError: operands could not be broadcast together with shapes (39009500,) (39009400,) in tc 4
        if vp_mis_full[ec_id][0].shape[0] != vm_mis_full[ec_id][0].shape[0]:
            res.append(vp_mis_full[ec_id][0][:vm_mis_full[ec_id][0].shape[0]] + abs(vm_mis_full[ec_id][0]))
        else:
            res.append(vp_mis_full[ec_id][0] + abs(vm_mis_full[ec_id][0]))

    return res



def extr_features(input_df, row, storage_path="feature_tc_"):
    input_df_n = input_df.to_numpy()

    for col in range(input_df_n.shape[1]):
        print("extracting: "+ str(row) + "/"+  str(col))
        test = input_df_n[row, col]
        df_indivi = pd.DataFrame(columns=['value', 'id'])
        for pair in input_df_n[row, col]:
            pair_df = pd.DataFrame(pair, columns=['value', 'id'])
            df_indivi = df_indivi.append(pair_df, ignore_index=True)
        extracted_features = extract_features(df_indivi, column_id="id", default_fc_parameters=settings)
        extracted_features.to_pickle(storage_path+str(row)+ "_f_" + str(col) + ".pkl")

'''
feature selection taken from https://www.kaggle.com/superluminal098/tsfresh-features-and-regression-blend
'''
settings={
'standard_deviation': None,
 'variance': None,
 'skewness': None,
 'kurtosis': None,

 'count_above_mean': None,
 'count_below_mean': None,
'last_location_of_maximum': None,
 'first_location_of_maximum': None,
 'last_location_of_minimum': None,
'first_location_of_minimum': None,

 'maximum': None,
 'minimum': None,

 'c3': [{'lag': 100}, {'lag': 2000}, {'lag': 3000},{'lag': 10000}],

'number_peaks': [{'n': 1},
                  {'n': 3},
                  {'n': 5},{'n': 100},{'n': 5000}],
'fft_coefficient':
 [{'coeff': 0, 'attr': 'real'},
  {'coeff': 1, 'attr': 'real'},
  {'coeff': 2, 'attr': 'real'},
  {'coeff': 3, 'attr': 'real'},
  {'coeff': 4, 'attr': 'real'},

  {'coeff': 0, 'attr': 'imag'},
  {'coeff': 1, 'attr': 'imag'},
  {'coeff': 2, 'attr': 'imag'},
  {'coeff': 3, 'attr': 'imag'},
  {'coeff': 4, 'attr': 'imag'},
  {'coeff': 5, 'attr': 'imag'},

  {'coeff': 0, 'attr': 'abs'},
  {'coeff': 1, 'attr': 'abs'},
  {'coeff': 2, 'attr': 'abs'},
  {'coeff': 3, 'attr': 'abs'},
  {'coeff': 4, 'attr': 'abs'},
  {'coeff': 5, 'attr': 'abs'},

  {'coeff': 0, 'attr': 'angle'},
  {'coeff': 1, 'attr': 'angle'},
  {'coeff': 2, 'attr': 'angle'},
  {'coeff': 3, 'attr': 'angle'},
  {'coeff': 4, 'attr': 'angle'},
  {'coeff': 5, 'attr': 'angle'},
],
 'agg_linear_trend': [
     {'attr': 'rvalue', 'chunk_len': 500, 'f_agg': 'max'},
  {'attr': 'rvalue', 'chunk_len': 500, 'f_agg': 'min'},
  {'attr': 'rvalue', 'chunk_len': 500, 'f_agg': 'mean'},
  {'attr': 'rvalue', 'chunk_len': 500, 'f_agg': 'var'},
  {'attr': 'rvalue', 'chunk_len': 1000, 'f_agg': 'max'},
  {'attr': 'rvalue', 'chunk_len': 1000, 'f_agg': 'min'},
  {'attr': 'rvalue', 'chunk_len': 1000, 'f_agg': 'mean'},
  {'attr': 'rvalue', 'chunk_len': 1000, 'f_agg': 'var'},
  {'attr': 'rvalue', 'chunk_len': 5000, 'f_agg': 'max'},
  {'attr': 'rvalue', 'chunk_len': 5000, 'f_agg': 'min'},
  {'attr': 'rvalue', 'chunk_len': 5000, 'f_agg': 'mean'},
  {'attr': 'rvalue', 'chunk_len': 5000, 'f_agg': 'var'},
  {'attr': 'intercept', 'chunk_len': 500, 'f_agg': 'max'},
  {'attr': 'intercept', 'chunk_len': 500, 'f_agg': 'min'},
  {'attr': 'intercept', 'chunk_len': 500, 'f_agg': 'mean'},
  {'attr': 'intercept', 'chunk_len': 500, 'f_agg': 'var'},
  {'attr': 'intercept', 'chunk_len': 1000, 'f_agg': 'max'},
  {'attr': 'intercept', 'chunk_len': 1000, 'f_agg': 'min'},
  {'attr': 'intercept', 'chunk_len': 1000, 'f_agg': 'mean'},
  {'attr': 'intercept', 'chunk_len': 1000, 'f_agg': 'var'},
  {'attr': 'intercept', 'chunk_len': 5000, 'f_agg': 'max'},
  {'attr': 'intercept', 'chunk_len': 5000, 'f_agg': 'min'},
  {'attr': 'intercept', 'chunk_len': 5000, 'f_agg': 'mean'},
  {'attr': 'intercept', 'chunk_len': 5000, 'f_agg': 'var'},
  {'attr': 'slope', 'chunk_len': 500, 'f_agg': 'max'},
  {'attr': 'slope', 'chunk_len': 500, 'f_agg': 'min'},
  {'attr': 'slope', 'chunk_len': 500, 'f_agg': 'mean'},
  {'attr': 'slope', 'chunk_len': 500, 'f_agg': 'var'},
  {'attr': 'slope', 'chunk_len': 1000, 'f_agg': 'max'},
  {'attr': 'slope', 'chunk_len': 1000, 'f_agg': 'min'},
  {'attr': 'slope', 'chunk_len': 1000, 'f_agg': 'mean'},
  {'attr': 'slope', 'chunk_len': 1000, 'f_agg': 'var'},
  {'attr': 'slope', 'chunk_len': 5000, 'f_agg': 'max'},
  {'attr': 'slope', 'chunk_len': 5000, 'f_agg': 'min'},
  {'attr': 'slope', 'chunk_len': 5000, 'f_agg': 'mean'},
  {'attr': 'slope', 'chunk_len': 5000, 'f_agg': 'var'},
  {'attr': 'stderr', 'chunk_len': 500, 'f_agg': 'max'},
  {'attr': 'stderr', 'chunk_len': 500, 'f_agg': 'min'},
  {'attr': 'stderr', 'chunk_len': 500, 'f_agg': 'mean'},
  {'attr': 'stderr', 'chunk_len': 500, 'f_agg': 'var'},
  {'attr': 'stderr', 'chunk_len': 1000, 'f_agg': 'max'},
  {'attr': 'stderr', 'chunk_len': 1000, 'f_agg': 'min'},
  {'attr': 'stderr', 'chunk_len': 1000, 'f_agg': 'mean'},
  {'attr': 'stderr', 'chunk_len': 1000, 'f_agg': 'var'},
  {'attr': 'stderr', 'chunk_len': 5000, 'f_agg': 'max'},
  {'attr': 'stderr', 'chunk_len': 5000, 'f_agg': 'min'},
  {'attr': 'stderr', 'chunk_len': 5000, 'f_agg': 'mean'},
  {'attr': 'stderr', 'chunk_len': 5000, 'f_agg': 'var'}],
 'index_mass_quantile': [{'q': 0.1},
  {'q': 0.2},
  {'q': 0.3},
  {'q': 0.4},
  {'q': 0.6},
  {'q': 0.7},
  {'q': 0.8},
  {'q': 0.9}],

 'spkt_welch_density': [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}],
 'ar_coefficient': [{'coeff': 0, 'k': 10},
  {'coeff': 1, 'k': 10},
  {'coeff': 2, 'k': 10},
  {'coeff': 3, 'k': 10},
  {'coeff': 4, 'k': 10}],

 'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0},
  {'num_segments': 10, 'segment_focus': 1},
  {'num_segments': 10, 'segment_focus': 2},
  {'num_segments': 10, 'segment_focus': 3},
  {'num_segments': 10, 'segment_focus': 4},
  {'num_segments': 10, 'segment_focus': 5},
  {'num_segments': 10, 'segment_focus': 6},
  {'num_segments': 10, 'segment_focus': 7},
  {'num_segments': 10, 'segment_focus': 8},
  {'num_segments': 10, 'segment_focus': 9}],
 'ratio_beyond_r_sigma': [{'r': 0.5},
  {'r': 1},
  {'r': 1.5},
  {'r': 2},
  {'r': 2.5},
  {'r': 3},
  {'r': 5},
  {'r': 6},
  {'r': 7},
  {'r': 10}],
'max_langevin_fixed_point': [{'m': 3, 'r': 30}], #nans

}
peak={
'number_peaks': [{'n': 1},
                  {'n': 3},
                  {'n': 5},
                  {'n': 10}]}


if __name__ == '__main__':

    # dates from xlxs files Lista_set_up_2018 and 2019_r1
    tc1 = ["1806250952", "1806280809","1806291527","1807030818","1807051049","1807060814","1807110755","1807111057","1807120739","1807130752","1807160737"] # still missing "1807170836"
    tc2 = ["1807230852","1807270742","1807300742","1808010835","1808020839","1808030746","1808031448","1809180945","1809181400","1809190848","1809210746","1809240747","1809250755","1809261424","1809270957","1810021021"] #  1807240806, ,"1807250802", ,"1807260927","1809200843", ,"1810020938"
    tc3 = ["1810100826","1810110850","1810120822","1810151055","1810170746","1810180743","1810190824","1810260815","1810290734","1810300742","1810310756","1811060736","1811070750","1811080737","1811081205","1811090740","1811120751","1811130745"]
    tc4 = ["1812180919","1901090835","1901100800","1901110805","1901140758","1901150756","1901210743","1901220802","1901230749","1901250744","1901280755","1901310848","1902051555","1902121458","1902121525","1902130906"]
    test_case_dates = [tc1, tc2, tc3, tc4]

    ec_data = split_data_into_ec(test_case_dates)
    analog_in = calculate_analog_in(ec_data)
    with open('data/min_max_pairs_hand.txt', "rb") as fp:
        min_max_pairs = pickle.load(fp)

    for id in range(len(test_case_dates)):
        print("collect data fields")
        df = collect_data_fields(ec_data, min_max_pairs, analog_in, ec_id=id)
        print("extract features")
        extr_features(df, id)

