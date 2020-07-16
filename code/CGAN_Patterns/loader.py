import numpy as np

class Loader():
    def __init__(self, features,labels):
        self.x = features
        self.y = labels

    def shuffle_x(self,batch_size, ):
        features=self.x
        labels=self.y
        idx = np.arange(0, len(features))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        data_shuffle = [features[i] for i in idx]
        labels_shuffle = [labels[i] for i in idx]

        return np.asarray(data_shuffle), np.asarray(
            labels_shuffle)