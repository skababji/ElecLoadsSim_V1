import os
import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

main_path = "D:/Documents/PhD_UWO/Confrnc_GAN/CGAN_Phase3/"


print('Enter Input Folder:')
trial=input()
input_path =main_path+trial+"/"
print('Reference Input Files for feature engineering exist in: ' + input_path)
PLOT_LOADS = ['CDE', 'DWE', 'FGE', 'HPE']
LOAD_NAMES = ['Cloth Dryer', 'Dishwasher', 'Fridge', 'Heat Pump']
SYMBOLS = ['o-', '*-', 'x-', '.-']
font = {'color': 'black','weight': 'normal','size': 22 }


if not os.path.exists(input_path+"/REPORTS"):
    os.makedirs(input_path+"REPORTS")
output_path = input_path+"REPORTS/"



CDE_real = pd.read_csv(input_path+ "/CDE_real_data.csv")
CDE_real = CDE_real.iloc[:, 1:]

CDE_real_kde = pd.read_csv(input_path+ "/CDE_kde.csv")

CDE_synth = pd.read_csv(input_path+ "/CDE_synth_data.csv")
CDE_synth = CDE_synth.iloc[:, 1:]



def plot_features_histograms(df,name,color, y_label=True):
    fig, axs = plt.subplots(3,figsize=(3,6))
    if y_label:
        y_label='Density'
    else:
        y_label=None

    x = np.asarray(df['week'])
    axs[0].hist(x, bins=52, range=[0, 51], density=True, rwidth=0.75, alpha=1, color=color)
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[0].set_xlabel('Week of Year', fontsize='x-large')
    axs[0].set_ylabel(y_label, fontsize='x-large')
    axs[0].set_ylim([0, 0.06])

    x = np.asarray(df['day'])
    axs[1].hist(x, bins=7, range=[0, 6], density=True, rwidth=0.75, alpha=1, color=color)
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1].set_xlabel('Day of Week', fontsize='x-large')
    axs[1].set_ylabel(y_label, fontsize='x-large')
    axs[1].set_ylim([0, 0.5])

    x = np.asarray(df['hour'])
    axs[2].hist(x, bins=24, range=[0, 23], density=True, rwidth=0.75, alpha=1, color=color)
    axs[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[2].set_xlabel('Hour of Day', fontsize='x-large')
    axs[2].set_ylabel(y_label, fontsize='x-large')
    axs[2].set_ylim([0, 0.2])

    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=1)
    plt.savefig(output_path + name + '.png')




plot_features_histograms(CDE_real,'real', 'blue', y_label=True)
plot_features_histograms(CDE_real_kde,'kde', 'orange',y_label=False)
plot_features_histograms(CDE_synth,'synth', 'red',y_label=False)

