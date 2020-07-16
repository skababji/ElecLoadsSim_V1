import numpy as np
import tensorflow as tf
import time

from CGAN_Habits.documenter import Documenter
from CGAN_Habits.plotter import Plotter
from CGAN_Habits.paths import Paths
from CGAN_Habits.preprocessor import Preprocessor
from CGAN_Habits.cgan_tf import Cgan_tf
from CGAN_Habits.distance import Distance
from CGAN_Habits.loader import Loader
from CGAN_Patterns.datasets import Datasets


tf.set_random_seed(1234)
np.random.seed(42)

TIC = time.time()


paths=Paths('train')
documenter = Documenter(paths)
preprocessor=Preprocessor(paths)
plotter=Plotter(paths)


datasets = Datasets(paths.raw_input_path+'data.xml')

input_folder=paths.enter_input_folder4train()
documenter.record_header(input_folder, datasets)

patterns=preprocessor.read_patterns(input_folder)

""" Engineer three features out of time stamp"""
df_all_data=preprocessor.eng_features(patterns)


""" Get X and y for GAN"""
real_X=df_all_data[['week','day','hour']].values
real_y=df_all_data['name'].values

""" Plot correlation among features """
for load in datasets.all_short_names:
    df_CDE=df_all_data[df_all_data['name']==load]
    plotter.plot_corr(df_CDE[['week','day','hour']], title='Correlation Matrix For '+load)

"""Scale X and encode y """
preprocessor.scale_fit_x(real_X)
preprocessor.encode_fit_y(real_y.reshape(-1,1))
documenter.record_params(preprocessor)

""" Construct Loader """
loader=Loader(preprocessor.scaled_X,preprocessor.encoded_y)

""" Construct CGAN-Habits"""
cgan_tf=Cgan_tf(preprocessor)
documenter.record_cgan_tf_arch(cgan_tf)


"""train gan """
distance=Distance()
cgan_tf.train(paths, preprocessor,loader, distance, datasets)
plotter.plot_gan_loss()
plotter.plot_mmd(datasets)


""" Plot real habits """
for i,value in enumerate(datasets.all_short_names):
    plotter.plot_hist(datasets.all_short_names[i], paths.real_habits_fn,color='green',data_type='real')



TOC = time.time() - TIC
documenter.record_input("Time Elapsed in mins:" + "{0:.2f}".format(TOC / 60) + "\n")
documenter.close_file()