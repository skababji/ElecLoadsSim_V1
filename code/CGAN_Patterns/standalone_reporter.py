# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

main_path = "D:/Documents/PhD_UWO/Confrnc_GAN/CGAN_Phase3/"


print('Enter Trial Name (i.e. subfolder name):')  # Input folder shall contain filtered_patters.csv
trial=input()
input_path =main_path+trial+"/"
print('Reference Input Files exist in: ' + input_path)
SHORT_NAMES = ['CDE', 'DWE', 'FGE', 'HPE']
LONG_NAMES = ['Cloth Dryer', 'Dishwasher', 'Fridge', 'Heat Pump']
SYMBOLS = ['o-', '*-', 'x-', '.-']
MASTER_WINDOW = 40
title_font = {'color': 'black', 'weight': 'normal', 'size': 20}
title_pad = 1.02
label_font = {'color': 'black', 'weight': 'normal', 'size': 16}
label_pad = 4
tick_font = 14
tick_pad = 4
epochs=10000
mmd_steps=np.arange(0,epochs+200,200)


if not os.path.exists(input_path+"/REPORTS"):
    os.makedirs(input_path+"REPORTS")
output_path = input_path+"REPORTS/"



def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma

def plot_mmd():
    mmds = pd.read_csv(input_path + "mmd.csv")
    mmds = mmds[SHORT_NAMES]
    plt.figure()
    for idx,load in enumerate(SHORT_NAMES):
        plt.plot(mmd_steps, movingaverage(mmds.loc[:, load], 1),SYMBOLS[idx], label=LONG_NAMES[idx])
    plt.legend(fontsize='large')
    plt.xlabel('Epoch', fontdict=label_font)
    plt.ylabel('MMD^2',fontdict=label_font)
    plt.tick_params(labelsize=tick_font, pad=title_pad)
    plt.grid()
    # plt.xlim(0,20)
    plt.savefig(output_path + trial+ "_mmds.png")
    plt.show()


def plot_mmd_sqrt():
    mmds = pd.read_csv(input_path + "mmd.csv")
    mmds = mmds[SHORT_NAMES]
    plt.figure()
    for idx,load in enumerate(SHORT_NAMES):
        plt.plot(mmd_steps,movingaverage(np.sqrt(mmds.loc[:, load]), 1),SYMBOLS[idx], label=LONG_NAMES[idx])
    plt.legend(fontsize='large')
    plt.xlabel('Epoch', fontdict=label_font)
    plt.ylabel('MMD',fontdict=label_font)
    plt.tick_params(labelsize=tick_font, pad=title_pad)
    plt.grid()
    plt.savefig(output_path + trial+ "_mmds_sqt.png")
    plt.show()


def plot_class_acc():
    scores_synth_history_df = pd.read_csv(input_path+"scores_synth_history.csv")
    plt.figure()
    plt.plot(scores_synth_history_df.iloc[0:31, 0], scores_synth_history_df.iloc[0:31, 2], label='Synth')
    plt.xlabel('Iteration')
    plt.ylabel('Cassification Accuracy')
    plt.grid()
    plt.savefig(output_path + trial+ "_acc_CGAN_pattern.png")
    plt.show()

def plot_patterns(data, labels, nof_samples=8, r=4, c=2):
    if nof_samples % (r * c) != 0:
        sys.exit("ERROR: PLEASE ENTER NO OF SAMPLES AS MULTIPLIER OF ROWS*COLS IN THE PLOT!")
    cnt = 0
    for k in range(int(nof_samples / (r * c))):
        fig, axs = plt.subplots(r, c)
        fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        if c > 1:
            for i in range(r):
                for j in range(c):
                    axs[i, j].plot(data[cnt, :])
                    axs[i, j].set_title(LONG_NAMES[SHORT_NAMES.index(labels[cnt])], {'fontsize': 12})
                    axs[i, j].set_yticklabels([])
                    axs[i, j].set_xticklabels([])
                    cnt += 1
        else:
            for i in range(r):
                axs[i].plot(data[cnt, :])
                cnt += 1

        fig.savefig(output_path + trial+"_synth_patterns", bbox_inches="tight")
        plt.show()



def plot_gan_loss():
    saved_loss = pd.read_csv(input_path+ 'loss_patterns.csv')
    plt.figure()
    plt.plot(saved_loss.loc[:, 'D_Loss'], label='D_Loss')
    plt.plot(saved_loss.loc[:, 'G_Loss'], label='G_Loss')
    plt.legend(fontsize='large')
    plt.xlabel('Epoch',fontdict=label_font)
    plt.ylabel('Loss',fontdict=label_font)
    plt.tick_params(labelsize=tick_font, pad=title_pad)
    plt.grid()
    plt.ylim(-1, 3)
    plt.savefig(output_path + trial+ "_loss_patterns.png")
    plt.show()


plot_mmd_sqrt()

plot_class_acc()


synth_df = pd.read_csv(input_path+"synth_patterns_rounded.csv")
real_df = pd.read_csv(input_path+ "filtered_patterns.csv")
patterns = []
names = []
for i in range(10):
    for idx, load in enumerate(SHORT_NAMES):
        synth_sample = synth_df.sample()
        name = np.asarray(synth_sample.iloc[:, 1])
        patterns.append(np.asarray(synth_sample.iloc[:, 2:MASTER_WINDOW - 1]))
        names.append(name)

patterns = np.concatenate(patterns, axis=0)
plot_patterns(patterns, names, nof_samples=28, r=7, c=4)

plot_gan_loss()


