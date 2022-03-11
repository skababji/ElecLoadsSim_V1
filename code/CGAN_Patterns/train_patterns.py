import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

from CGAN_Patterns.datasets import Datasets
from CGAN_Patterns.paths import Paths
from CGAN_Patterns.hyperparam import Hyperparam
from CGAN_Patterns.documenter import Documenter
from CGAN_Patterns.plotter import Plotter

from CGAN_Patterns.preprocessor import Preprocessor
from CGAN_Patterns.cgan_tf import Cgan_tf
from CGAN_Patterns.loader import Loader
from CGAN_Patterns.evaluator import Evaluator
from CGAN_Patterns.distance import Distance


tf.set_random_seed(1234)
np.random.seed(42)


TIC = time.time()

paths=Paths('train')
documenter = Documenter(paths)
documenter.record_header()


power_raw=[]
power_resampled=[]
datasets = Datasets(paths.raw_input_path+'data.xml')

for i in range (datasets.n_datasets):
    this_power_raw = datasets.read_raw(paths.raw_input_path,datasets.datasets[i])
    power_raw.append(this_power_raw)
    this_resampled_power=datasets.resample(this_power_raw,datasets.datasets[i].get('sampling_period'))
    power_resampled.append(this_resampled_power)

documenter.record_datasets(datasets)

plotter = Plotter(paths)
documenter.record_plotter(datasets)


examples=[] # A list of dataframes where the first dataframe correspondeds to the gathered examples from dataset 1 and the second dataframe corresponds to the seconds and so forth..



for i in range(datasets.n_datasets):
    this_examples = datasets.make_examples(datasets.datasets[i].get('data_name'), power_resampled[i], datasets.tmplt_lookup_df[i], documenter)
    examples.append(this_examples)

def ampd_runs_per_week(load_name): # A function returns the number of runs detected for a given load in A<Pd dataset and returns average runs per week for that load. This is used for final aggregation purposes
    this_load_examples=examples[0][examples[0]['name']==load_name]
    no_runs=len(this_load_examples)
    weeks=(this_load_examples['starttime'].max()-this_load_examples['starttime'].min())/60/60/24/7
    runs_per_week=no_runs/weeks
    return runs_per_week

ampd_load_names=examples[0].drop_duplicates(subset='name')['name'].values
ampd_load_runs_per_week=np.array([ampd_runs_per_week(x) for x in ampd_load_names])
ampd_runs_per_week=pd.DataFrame(ampd_load_runs_per_week.reshape(1,-1), columns=ampd_load_names)
ampd_runs_per_week.to_csv(paths.current_path + 'ampd_runs_per_week.csv', index=False)

df_filtered = datasets.combine_examples(examples)
documenter.record_input("Total Training Examples: {0} \n".format(str(df_filtered.shape[0])))
df_filtered.to_csv(paths.current_path + 'filtered_patterns.csv')

for i in range(len(datasets.all_short_names)):
    plotter.plot_real_loads('filtered_patterns.csv', datasets.all_short_names[i], nof_samples=plotter.PLOT_SMPLS, r=plotter.PLOT_ROWS,
                    c=plotter.PLOT_COLS)
    plotter.plot_real_loads_histo('filtered_patterns.csv', datasets.all_short_names[i])

preprocessor=Preprocessor(paths.current_path + 'filtered_patterns.csv')
documenter.record_params(preprocessor, datasets)

loader=Loader(preprocessor.scaled_X,preprocessor.encoded_y)

evaluator=Evaluator(preprocessor)
shuffled_x,shuffled_y=loader.shuffle_x(preprocessor.X.shape[0])
train_x, test_x, train_y, test_y = train_test_split(shuffled_x, shuffled_y, test_size=0.1, random_state=0)
evaluator.train_model(train_x,train_y)
plotter.plot_eval_acc(evaluator.evaluator_history)
plotter.plot_eval_loss(evaluator.evaluator_history)
documenter.record_eval(evaluator)
est_y=evaluator.predict_model(test_x)
scores_real=evaluator.test_model(test_x,test_y)
test_y=preprocessor.encode_y_inverse(test_y)
est_y=preprocessor.encode_y_inverse(est_y)
plotter.plot_confusion_matrix(datasets,'real',test_y,est_y, normalize=True,title='Normalized Confusion Matrix for testing set of Real Data')

cgan_tf = Cgan_tf(paths, preprocessor)
documenter.record_cgan_tf_arch(cgan_tf)

distance=Distance()
cgan_tf.train(datasets,preprocessor,loader,evaluator, distance)
plotter.plot_gan_loss()
plotter.plot_eval_scores_synth(scores_real)
plotter.plot_mmd(datasets)
plotter.plot_mmd_sqrt(datasets)


TOC = time.time() - TIC
documenter.record_input("Time Elapsed in mins:" + "{0:.2f}".format(TOC / 60) + "\n")
documenter.close_file()