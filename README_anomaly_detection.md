# Anomaly detection

In src/data/preprocessing.py the code for loading the data and extracting the features with ts fresh can be found.

/data/min_max_pairs_hand.txt contains lists with min-max pairs for all test cases. Those pairs are used in preprocessing.py to cut the data into ramp-ups. The pairs where extracted with scipy.signal.find_peaks(analog_in, prominence=250, width=50) and find_peaks(-1 * (analog_in), prominence=50, width=50, height=-5) for peaks and valleys respectively. Afterwards the min-max pairs were improved by hand.

In src/visualization histograms.py and ploting_helper_functions.py contain functions to create the plots seen in the presentation /documentation/Anamoly_detection_pres.pdf.
