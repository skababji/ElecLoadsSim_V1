import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder


class Preprocessor():

    def __init__(self, paths):
        self.paths=paths

    def read_patterns(self, input_folder):
        return pd.read_csv(input_folder + '/filtered_patterns.csv')

    def eng_features(self, df_old):
        df_old = df_old.iloc[:, 1:]
        df_old['starttime'] = pd.to_datetime(df_old['starttime'], unit='s')
        df_new = pd.DataFrame()
        df_new['name'] = df_old['name']
        df_new['week'] = df_old['starttime'].dt.week
        df_new['day'] = df_old['starttime'].dt.weekday
        df_new['hour'] = df_old['starttime'].dt.hour
        df_mean = df_old.iloc[:, 2:]
        df_mean['power_avg'] = df_mean.mean(axis=1)
        df_new = pd.concat([df_new, df_mean.iloc[:, -1]], axis=1)
        df_new = df_new.iloc[:, 0:4]
        df_new.to_csv(self.paths.current_path + self.paths.real_habits_fn, index=False)
        return df_new



    def scale_fit_x(self,x):
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaled_X= self.scaler.fit_transform(x)
        self.X_dim=self.scaled_X.shape[1]
        return self.scaled_X

    def scale_x(self,x):
        return self.scaler.transform(x)

    def scale_x_inverse(self,x):
        return self.scaler.inverse_transform(x)

    def encode_fit_y(self,y):
        self.encoder = OneHotEncoder()
        self.encoded_y= self.encoder.fit_transform(y).toarray()
        self.y_dim=self.encoded_y.shape[1]
        return self.encoded_y

    def encode_y(self,y):
        return self.encoder.transform(self.y).toarray()

    def encode_y_inverse(self,y):
        return self.encoder.inverse_transform(y)



