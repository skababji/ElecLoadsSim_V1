import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder

from CGAN_Patterns.hyperparam import Hyperparam

def add_one_bin(x, h_bins):
    feature = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        [this_hist, this_bin_edges] = np.histogram(x[i, :], bins=h_bins)
        feature[i] = this_hist[h_bins - 1]
    return feature


def add_all_bins(x, h_bins):
    feature = np.zeros([x.shape[0], h_bins])
    for i in range(x.shape[0]):
        [this_hist, this_bin_edges] = np.histogram(x[i, :], bins=h_bins)
        feature[i, :] = this_hist
    return feature


class Preprocessor():

    def __init__(self,fullfilename):
        dataset= pd.read_csv(fullfilename)
        self.X = dataset.iloc[:, 3:].values
        if Hyperparam.ADNTL_FEATURES != 0:
            if Hyperparam.ADNTL_FEATURES == 1:
                eng_features = add_one_bin(self.X, 3)
                self.X = np.concatenate([self.X, eng_features.reshape(-1, 1)], axis=1)

            elif Hyperparam.ADNTL_FEATURES == Hyperparam.HISTOGRAM_BINS:
                eng_features = add_all_bins(self.X, Hyperparam.HISTOGRAM_BINS)
                self.X = np.concatenate([self.X, eng_features], axis=1)

        self.y = dataset.iloc[:, 1].values.reshape(-1, 1)

        self.X_scaler_type= 'MinMax (-1,1)'
        self.y_encoder_type='One Hot Encoder'

        self.scaled_X=self.scale_fit_x()
        self.encoded_y=self.encode_y()

        self.X_dim=self.scaled_X.shape[1]
        self.y_dim=self.encoded_y.shape[1]


    def scale_fit_x(self):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        return self.scaler.fit_transform(self.X)

    def scale_x(self,x):
        return self.scaler.transform(x)

    def scale_x_inverse(self,x):
        return self.scaler.inverse_transform(self.x)

    def encode_y(self):
        self.encoder = OneHotEncoder()
        return self.encoder.fit_transform(self.y).toarray()

    def encode_y_inverse(self,y):
        return self.encoder.inverse_transform(y)




