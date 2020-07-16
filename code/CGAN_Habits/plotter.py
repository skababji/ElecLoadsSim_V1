import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from CGAN_Habits.hyperparam import Hyperparam

def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma

class Plotter():
    def __init__(self, paths):
        self.paths=paths

    def plot_corr(self,df, title):
        plt.figure(figsize=(12, 10))
        cor = df.corr()
        sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
        plt.title(title)
        plt.savefig(self.paths.current_path+ 'correlation.png')
        plt.show()

    def plot_dist(self,condition,filename):
        df=pd.read_csv(self.paths.current_path+ filename)
        df_load=df[df['load']==condition]
        if Hyperparam.PICKED_FEATURE=='all':
            self.plot_dist_kde3(condition, df_load)
        else:
            self.plot_dist_kde1(condition,df_load)

    def plot_dist_kde1(self, condition, df_load):
        picked_feature=df_load.columns.values[1]
        plt.fill_between(df_load[picked_feature], np.exp(df_load['logprob']), alpha=0.7)
        plt.plot(df_load[picked_feature], np.full_like(df_load[picked_feature], -0.01), '|k', markeredgewidth=1)
        # plt.ylim(-0.02, 0.22)
        plt.xlabel(picked_feature)
        plt.ylabel('Density')
        plt.title('Density Estimation for ' + condition)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)
        plt.savefig(self.paths.current_path + condition + '_' + picked_feature + '_kde_dist.png')
        plt.close()

    def plot_dist_kde3(self,condition,df_load ):
        plt.plot(df_load['week'], np.exp(df_load['logprob']), alpha=0.5, label='week of year')
        plt.plot(df_load['day'], np.exp(df_load['logprob']), alpha=0.5, label='day of week')
        plt.plot(df_load['hour'], np.exp(df_load['logprob']), alpha=0.5, label='hour of day')
        plt.ylabel('Density')
        plt.xlabel('index')
        plt.title('Density Estimations for ' + condition)
        plt.legend()
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)
        plt.savefig(self.paths.current_path + condition + '_kde_dist.png')
        plt.close()

    def plot_hist(self, condition, filename='real_data.csv',color='green', data_type='real'):
        df = pd.read_csv(self.paths.current_path + filename)
        df_load = df[df['name'] == condition]
        if Hyperparam.PICKED_FEATURE == 'all':
            self.plot_hist_3(condition, df_load,color,data_type)
        else:
            self.plot_hist_1(condition, df_load,color,data_type)

    def plot_hist_1(self, condition, df_load,color,data_type):
        picked_feature=df_load.columns.values[1]
        plt.figure()
        plt.hist(df_load[picked_feature], bins=int(np.max(df_load[picked_feature])), range=[0, int(np.max(df_load[picked_feature]))], rwidth=0.75, density=True,
                 color=color)
        plt.title(picked_feature +' Histogram of real data for ' + condition)
        plt.xlabel('Index')
        plt.ylabel('Density')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)
        plt.savefig(self.paths.current_path + 'histo_'+data_type+'_' + condition + ' _' + picked_feature+'.png')
        plt.close()

    def plot_hist_3(self, condition, df_load,color,data_type):
        X = np.asarray(df_load[['week','day','hour']])
        for i,value in enumerate(Hyperparam.FEATURES):
            plt.figure()
            plt.hist(X[:,i], bins=int(np.max(X[:,i])), range=[0, int(np.max(X[:,i]))], rwidth=0.75, density=True,
                     alpha=0.7, label=Hyperparam.FEATURES_NAMES[i], color=color)
            plt.title(Hyperparam.FEATURES_NAMES[i]+' Histogram of sampled data for  ' + condition)
            plt.xlabel('Index')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1)
            plt.savefig(self.paths.current_path + 'histo_'+data_type+'_' + condition +'_'+ Hyperparam.FEATURES[i]+'.png')
            plt.close()


    def plot_gan_loss(self):
        saved_loss = pd.read_csv(self.paths.current_path + self.paths.loss_fn)
        plt.figure()
        plt.plot(saved_loss.loc[:, 'D_Loss'], label='D_Loss')
        plt.plot(saved_loss.loc[:, 'G_Loss'], label='G_Loss')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.ylim(-1, 3)
        plt.savefig(self.paths.current_path + self.paths.loss_fig)
        plt.show()

    def plot_mmd(self, datasets):
        mmds = pd.read_csv(self.paths.current_path + self.paths.mmd_fn)
        mmds = mmds[datasets.all_short_names]
        plt.figure()
        for load in datasets.all_short_names:
            plt.plot(movingaverage(mmds.loc[:, load], 5), label=load)
        plt.legend()
        plt.xlabel('Epoch (x 100)')
        plt.ylabel('MMD')
        plt.grid()
        plt.savefig(self.paths.current_path + self.paths.mmd_fig)
        plt.show()

