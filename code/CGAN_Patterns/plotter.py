import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np


from CGAN_Patterns.datasets import Datasets
from CGAN_Patterns.hyperparam import Hyperparam

def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma

class Plotter():
    PLOT_SMPLS = 200
    PLOT_BINS = 50
    PLOT_ROWS = 5
    PLOT_COLS = 2

    def __init__(self, paths):
        self.paths=paths
        self.title_font = {'color': 'black', 'weight': 'normal', 'size': 20}
        self.title_pad=1.02
        self.label_font={'color': 'black','weight': 'normal','size': 16}
        self.label_pad=4
        self.tick_font=16
        self.tick_pad=8
    
    def plot_power(self,dataset_name,loads_df,loads_list,sampling_period,duration=3):
        plt.rcParams.update({'figure.autolayout': True})
        steps=int(duration*24*60*60/sampling_period)
        for idx, value in enumerate(loads_list):
            intervaled_load_df=loads_df[value][1500:1500+steps]
            plt.figure(figsize=(6,4))
            plt.plot(intervaled_load_df)
            plt.title("Raw data of " + value + " usage habits", fontdict=self.title_font, y=self.title_pad)
            plt.xlabel('Index (Each time step = ' + str(sampling_period/60) + ' minute)', fontdict=self.label_font, labelpad=self.label_pad)
            plt.ylabel('Power - Watts', fontdict=self.label_font, labelpad=self.label_pad)
            plt.xticks(np.arange(1500,1500+len(intervaled_load_df),len(intervaled_load_df)/3))
            plt.tick_params(labelsize=self.tick_font, pad=self.title_pad)
            plt.show()
    
    def plot_templates(self,dataset_name,loads_df,loads_list,sampling_period, tmplt_lookup_df):
        plt.rcParams.update({'figure.autolayout': True})
        for idx, value in enumerate(loads_list):
            start = int(tmplt_lookup_df[value].loc['tmplt_start'])
            end = int(tmplt_lookup_df[value].loc['tmplt_end'])
            plt.figure(figsize=(6,4))
            plt.plot(loads_df[value].iloc[start:end])
            plt.title("Raw data of " + value + " pattern", fontdict=self.title_font, y=self.title_pad)
            plt.xlabel('Index (Each time step = ' + str(sampling_period/60) + ' minute)',fontdict=self.label_font, labelpad=self.label_pad)
            plt.ylabel('Power - Watts', fontdict=self.label_font, labelpad=self.label_pad)
            plt.tick_params(labelsize=self.tick_font,pad=self.title_pad )
            plt.show()
    
    def plot_power2pdf(self,dataset_name,loads_df,loads_list,sampling_period ):
        with PdfPages(self.paths.current_path + "RawData_" + dataset_name + ".pdf") as pdf:
            for idx, value in enumerate(loads_list):
                plt.figure()
                plt.plot(loads_df[value])
                plt.title("Raw Data of Load " + value + " of "+dataset_name+" Dataset", fontdict=self.title_font)
                plt.xlabel('Index (Each time step = ' + str(sampling_period/60) + ' minute)', fontdict=self.label_font)
                plt.ylabel('Power - Watts', fontdict=self.label_font)
                plt.tick_params(labelsize=self.tick_font)
                pdf.savefig()
                plt.close()
    
    def plot_templates2pdf(self,dataset_name,undersampled_loads_df,loads_list,sampling_period, tmplt_start, tmplt_end):
        sampling_period=int(sampling_period)
        step=int(Hyperparam.TS_UNIFIED/sampling_period)
        tmplt_start=(np.array(tmplt_start)/step).astype(int)
        tmplt_end = (np.array(tmplt_end) / step).astype(int)
        with PdfPages(self.paths.current_path + "Templates_" + dataset_name + ".pdf") as pdf:
            for idx, value in enumerate(loads_list):
                plt.figure()
                plt.plot(undersampled_loads_df[value].iloc[tmplt_start[idx]:tmplt_end[idx]])
                plt.title("Raw Data of Load " + value + " of "+ dataset_name+" Dataset", fontdict=self.title_font)
                plt.xlabel('Index (Each time step = ' + str(step*sampling_period/60) + ' minute)', fontdict=self.label_font)
                plt.ylabel('Power - Watts', fontdict=self.label_font)
                plt.tick_params(labelsize=self.tick_font)
                pdf.savefig()
                plt.close()
                
    def plot_real_loads(self,filename, this_load_name, nof_samples=5, r=5, c=1):
        data = pd.read_csv(self.paths.current_path +filename)
        app_df = data.loc[data['name'] == this_load_name]
        app_X = app_df.iloc[:, 3:].sample(
            nof_samples).values
        if nof_samples % (r * c) != 0:
            sys.exit("ERROR: PLEASE ENTER NO OF SAMPLES AS MULTIPLIER OF ROWS*COLS IN THE PLOT!")
        cnt = 0
        for k in range(int(nof_samples / (r * c))):
            fig, axs = plt.subplots(r, c)
            fig.tight_layout()
            if c > 1:
                for i in range(r):
                    for j in range(c):
                        axs[i, j].plot(app_X[cnt, :])
                        cnt += 1
            else:
                for i in range(r):
                    axs[i].plot(app_X[cnt, :])
                    cnt += 1
    
            fig.suptitle(this_load_name + ' Random Real Samples-  Page - ' + str(k + 1), y=1)
            fig.savefig(self.paths.current_path + "REAL_{}_P{}.png".format(this_load_name, int(k + 1)),
                        bbox_inches="tight")
            plt.close()
        
    
    def plot_real_loads_histo(self,filename, load_name, h_bins=50, nof_samples=5, r=5, c=1):
        data = pd.read_csv(self.paths.current_path + filename)
        app_df = data.loc[data['name'] == load_name]
        app_X = app_df.iloc[:, 3:].sample(
            nof_samples).values
        fig, axs = plt.subplots(r, c)
        fig.tight_layout()
        cnt = 0
        if c > 1:
            for i in range(r):
                for j in range(c):
                    axs[i, j].hist(app_X[cnt, :], bins=h_bins)
                    cnt += 1
        else:
            for i in range(r):
                axs[i].hist(app_X[cnt, :], bins=h_bins)
                cnt += 1
    
        fig.suptitle('Histogram for ' + load_name + '-Real', y=1)
        plt.xlabel('Bin')
        fig.savefig(self.paths.current_path + "REAL__hist_{}.png".format(load_name), bbox_inches="tight")
        plt.close()


    def plot_eval_acc(self,evaluator_history):
        plt.figure()
        plt.plot(evaluator_history.history['acc'])
        plt.plot(evaluator_history.history['val_acc'])
        plt.title('Evaluator Accuracy', fontdict=self.label_font)
        plt.ylabel('Accuracy', fontdict=self.label_font)
        plt.xlabel('Epoch', fontdict=self.label_font)
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.paths.current_path + "evaluator_acc.png")
        plt.show()

    def plot_eval_loss(self,evaluator_history):
        plt.figure()
        plt.plot(evaluator_history.history['loss'])
        plt.plot(evaluator_history.history['val_loss'])
        plt.title('Evaluator Loss',fontdict=self.label_font)
        plt.ylabel('Loss',fontdict=self.label_font)
        plt.xlabel('Epoch',fontdict=self.label_font)
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.paths.current_path + "evaluator_loss.png")
        plt.show()

    def plot_confusion_matrix(self,datasets,data_type,y_true, y_pred,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'


        cm = confusion_matrix(y_true, y_pred)
        classes=datasets.all_long_names[np.where(datasets.all_short_names==unique_labels(y_true, y_pred))]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(aspect='auto',xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(self.paths.current_path+ "cm_eval_"+data_type+".png")
        return ax







    def plot_synth_load(self, filename, this_load_name, nof_samples=5, r=5, c=1):
        data = pd.read_csv(self.paths.current_path+ filename)
        app_df = data.loc[data['1'] == this_load_name]
        app_X = app_df.iloc[:, 2:].sample(nof_samples).values
        if nof_samples % (r * c) != 0:
            sys.exit("ERROR: PLEASE ENTER NO OF SAMPLES AS MULTIPLIER OF ROWS*COLS IN THE PLOT!")
        cnt = 0
        for k in range(int(nof_samples / (r * c))):
            fig, axs = plt.subplots(r, c)
            fig.tight_layout()
            if c > 1:
                for i in range(r):
                    for j in range(c):
                        axs[i, j].plot(app_X[cnt, :])
                        cnt += 1
            else:
                for i in range(r):
                    axs[i].plot(app_X[cnt, :])
                    cnt += 1

            fig.suptitle(this_load_name + ' Random Samples Generated by GAN- Page ' + str(k + 1), y=1)
            fig.savefig(self.paths.current_path + "SYNTH_{}_P{}.png".format(this_load_name, int(k + 1)),
                        bbox_inches="tight")
            plt.close()

    def plot_synth_loads_histo(self, datasets, filename, load_name, nof_samples=5, r=5, c=1):
        data = pd.read_csv(self.paths.current_path + filename)
        app_df = data.loc[data['1'] == load_name]
        app_X = app_df.iloc[:, 2:datasets.master_window].sample(
            nof_samples).values  # Get only chosen number of randomly picked patterns to plot them
        fig, axs = plt.subplots(r, c)
        fig.tight_layout()
        cnt = 0
        if c > 1:
            for i in range(r):
                for j in range(c):
                    axs[i, j].hist(app_X[cnt, :], bins=Hyperparam.HISTOGRAM_BINS)
                    cnt += 1
        else:
            for i in range(r):
                axs[i].hist(app_X[cnt, :], bins=Hyperparam.HISTOGRAM_BINS)
                cnt += 1

        fig.suptitle('Histogram for ' + load_name + '-Synth', y=1)
        plt.xlabel('Bin')
        fig.savefig(self.paths.current_path + "FAKE__hist_{}.png".format(load_name), bbox_inches="tight")
        plt.close()


    def plot_eval_scores_synth(self,scores_real,filename='scores_synth_history.csv'):
        scores_synth_history = pd.read_csv(self.paths.current_path+  filename)
        scores_synth_history=scores_synth_history.values
        plt.figure()
        scores_real = scores_real[1] * np.ones_like(scores_synth_history[:, 0])
        plt.plot(scores_synth_history[:, 0], scores_real, label='Real')
        plt.plot(scores_synth_history[:, 0], scores_synth_history[:, 2], label='Synth')
        plt.annotate(r'$\mu= {:.3f}$'.format(scores_real[1]), xy=(-10, scores_real[1] - .05))
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Pattern Generation Accuracy using Evaluator Net - In Training')
        plt.savefig(self.paths.current_path+ "acc_CGAN_pattern_noSpline.png")
        plt.show()

    def plot_gan_loss(self,filename='loss_patterns.csv'):
        saved_loss = pd.read_csv(self.paths.current_path+ filename)
        plt.figure()
        plt.plot(saved_loss.loc[:, 'D_Loss'], label='D_Loss')
        plt.plot(saved_loss.loc[:, 'G_Loss'], label='G_Loss')
        plt.legend()
        plt.xlabel('Epoch',fontdict=self.label_font)
        plt.ylabel('Loss',fontdict=self.label_font)
        plt.grid()
        plt.ylim(-1, 3)
        plt.savefig(self.paths.current_path+"loss_patterns.png")
        plt.show()

    def plot_mmd(self,datasets,filename='mmd.csv',):
        mmds = pd.read_csv(self.paths.current_path +filename)
        mmds = mmds[datasets.all_short_names]
        plt.figure()
        for load in datasets.all_short_names:
            plt.plot(movingaverage(mmds.loc[:, load], 1), label=load)
        plt.legend()
        plt.xlabel('Epoch (x 100)')
        plt.ylabel('MMD^2')
        plt.grid()
        plt.savefig(self.paths.current_path + "mmds.png")
        plt.show()

    def plot_mmd_sqrt(self, datasets, filename='mmd.csv', ):
        mmds = pd.read_csv(self.paths.current_path+ filename)
        mmds = mmds[datasets.all_short_names]
        plt.figure()
        for load in datasets.all_short_names:
            plt.plot(movingaverage(np.sqrt(mmds.loc[:, load]), 1), label=load)
        plt.legend()
        plt.xlabel('Epoch (x 100)')
        plt.ylabel('MMD')
        plt.grid()
        plt.savefig(self.paths.current_path +"mmds_sqrt.png")
        plt.show()
        