import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from CGAN_Habits.hyperparam import Hyperparam



class Kde():

    def __init__(self, paths):
        self.paths=paths

    def get_kde_samples(self,df_all_data,documenter,datasets):
        all_X_sampled=[]
        all_y=[]
        # Fit KDE to each appliance
        for i, label in enumerate(datasets.all_short_names):
            this_X = np.array(df_all_data.loc[df_all_data['name'] == label])
            this_X = this_X[:, 1:].astype(float)
            this_kde, _ = self.get_kde_per_load(this_X,documenter,Hyperparam.SAMPLES,label)
            this_X_sampled=this_kde.sample(Hyperparam.SAMPLES)

            all_X_sampled.append(this_X_sampled)
            this_y = np.full([len(this_X_sampled), 1], label)
            all_y.append(this_y)

        all_X_sampled = np.concatenate(all_X_sampled, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        all_y = all_y.reshape(-1, 1)

        df = pd.DataFrame(np.concatenate([all_y, all_X_sampled], axis=1), columns=['name', 'week', 'day', 'hour'])
        df.iloc[:, 1:] = (df.iloc[:, 1:].astype(float)).round(0)
        df.to_csv(self.paths.current_path + self.paths.kde_habits_fn, index=False)

        return df



    def get_kde_per_load(self,x,documenter, n_samples=1000, label='CDE'):
        x = x.astype(float)

        params = {'bandwidth': np.logspace(-0.75, 0.75, 10)}  # [-1,0.38,1,5]
        grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
        grid.fit(x)
        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
        results=grid.cv_results_
        documenter.record_input("best bandwidth for " + label + ": " + str(grid.best_estimator_.bandwidth) + "\n")
        kde = grid.best_estimator_

        features = x.shape[1]
        x_d = np.zeros([n_samples, features])
        for i in range(features):
            x_d[:, i] = np.linspace(np.min(x[:, i]) - 1, np.max(x[:, i]) + 1, n_samples)
        logprob_x_d = kde.score_samples(x_d)

        df_real_d = pd.DataFrame(np.concatenate([x_d, logprob_x_d[:, None]], axis=1),
                                 columns=['week', 'day', 'hour',
                                          'logprob'])
        return kde,df_real_d


