import pandas as pd
import numpy as np
import tensorflow as tf
import time

from CGAN_Habits.documenter import Documenter
from CGAN_Habits.plotter import Plotter
from CGAN_Habits.hyperparam import Hyperparam
from CGAN_Habits.paths import Paths
from CGAN_Habits.preprocessor import Preprocessor
from CGAN_Habits.cgan_tf import Cgan_tf
from CGAN_Habits.kde import Kde
from CGAN_Patterns.datasets import Datasets


tf.set_random_seed(1234)
np.random.seed(42)

TIC = time.time()



paths=Paths('generate')
documenter = Documenter(paths)
preprocessor=Preprocessor(paths)
plotter=Plotter(paths)
kde=Kde(paths)

datasets = Datasets(paths.raw_input_path+'data.xml')

print("Enter Folder containg Habits model")
input_folder=paths.enter_input_folder4gen()
documenter.record_header(input_folder, datasets)

""" Read Real Habits """
df_all_data=pd.read_csv(input_folder+paths.real_habits_fn)


""" sample X for all labels using KDE"""
df_kde= kde.get_kde_samples(df_all_data,documenter,datasets)


real_X=df_all_data[['week','day','hour']].values
real_y=df_all_data['name'].values

preprocessor.scale_fit_x(real_X)
preprocessor.encode_fit_y(real_y.reshape(-1,1))
documenter.record_params(preprocessor)


""" Construct CGAN-Habits"""
cgan_tf=Cgan_tf(preprocessor)
documenter.record_cgan_tf_arch(cgan_tf)





"""Generate Synthetic data"""
generated_samples=cgan_tf.generate_habit(paths, input_folder, preprocessor, Hyperparam.SAMPLES, datasets.all_short_names)


""" Plot results """
for i,value in enumerate(datasets.all_short_names):
    plotter.plot_hist(datasets.all_short_names[i], paths.kde_habits_fn,color='orange',data_type='kde')
    plotter.plot_hist(datasets.all_short_names[i],  paths.synth_habits_fn,color='red',data_type='synth')


TOC = time.time() - TIC
documenter.record_input("Time Elapsed in mins:" + "{0:.2f}".format(TOC / 60) + "\n")
documenter.close_file()