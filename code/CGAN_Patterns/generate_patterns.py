# %%
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
import os


import tensorflow as tf

from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

import time

from CGAN_Patterns.datasets import Datasets
from CGAN_Patterns.paths import Paths
from CGAN_Patterns.documenter import Documenter
from CGAN_Patterns.plotter import Plotter

from CGAN_Patterns.preprocessor import Preprocessor
from CGAN_Patterns.cgan_tf import Cgan_tf
from CGAN_Patterns.distance import Distance



""" Reproducing RAE power series with suitable short names"""
# def reproduce_RAE():
#     rP_RAE = pd.read_csv(Paths.input_path + 'house1_power_blk2.csv')
#     # Mapping RAE Names to AMPd Names
#     UNIX_TS = pd.DataFrame(np.array(rP_RAE['unix_ts']), columns=['UNIX_TS'])
#     CDE = pd.DataFrame(np.array(rP_RAE['sub5'] + rP_RAE['sub6']), columns=['CDE'])
#     CWE = pd.DataFrame(np.array(rP_RAE['sub9']), columns=['CWE'])
#     HPE = pd.DataFrame(np.array(rP_RAE['sub13'] + rP_RAE['sub14']), columns=['HPE'])
#     rP_RAE = pd.concat([UNIX_TS, CDE, CWE, HPE], axis=1)
#     rP_RAE.to_csv(Paths.input_path + 'house1_power_blk2_mod.csv', index=False)
#
# reproduce_RAE()


tf.set_random_seed(1234)
np.random.seed(42)
TIC = time.time()

paths=Paths('generate')
plotter=Plotter(paths)
distance=Distance()
documenter=Documenter(paths)

input_path=paths.enter_input_folder()
documenter.record_gen_input(input_path)

datasets = Datasets(paths.raw_input_path+'data.xml')
preprocessor=Preprocessor(input_path+'filtered_patterns.csv')
cgan_tf = Cgan_tf(paths, preprocessor)


# copy ampd_runs_per_week.csv file from training folder to inference folder
source=pd.read_csv(os.path.join(os.getcwd(),input_path,'ampd_runs_per_week.csv'))
source.to_csv(os.path.join(os.getcwd(),paths.current_path, 'ampd_runs_per_week.csv'), index=False)







# Set Global Random seed to ensure versions compatabilities




"""Generate Synthetic data"""
generated_samples=cgan_tf.generate_pattern(input_path,paths.current_path,preprocessor,plotter.PLOT_SMPLS, datasets.all_short_names)
for i in range(len(datasets.all_short_names)):
    plotter.plot_synth_load('synth_patterns_rounded.csv', datasets.all_short_names[i], nof_samples=plotter.PLOT_SMPLS, r=plotter.PLOT_ROWS, c=plotter.PLOT_COLS)
synth_x=generated_samples[:,2:]
conditions=generated_samples[:,1]
synth_x=preprocessor.scale_x(synth_x)


TOC = time.time() - TIC
documenter.record_gen_time(TOC)
documenter.close_file()