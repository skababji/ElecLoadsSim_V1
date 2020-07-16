from keras.layers import Dense
from keras.models import Sequential

from CGAN_Patterns.hyperparam import Hyperparam

class Evaluator():
    def __init__(self, preprocessor):
        self.X_dim=preprocessor.X_dim
        self.y_dim=preprocessor.y_dim
        self.build_model()


    def build_model(self):
        E_H_DIM=Hyperparam.E_H_DIM
        model = Sequential()
        model.add(Dense(E_H_DIM[0], input_dim=self.X_dim, activation='relu'))
        model.add(Dense(E_H_DIM[1], activation='relu'))
        model.add(Dense(E_H_DIM[2], activation='relu'))
        model.add(Dense(E_H_DIM[3], activation='relu'))
        model.add(Dense(self.y_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[
            'acc'])
        self.model=model

    def train_model(self,train_x,train_y):
        self.evaluator_history = self.model.fit(train_x, train_y, validation_split=0.33, epochs=1000, batch_size=256)


    def test_model(self,test_x,test_y):
        return self.model.evaluate(test_x, test_y)

    def predict_model(self,test_x):
        return self.model.predict(test_x)

