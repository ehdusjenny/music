from keras.models import Sequential
from keras.layers import Dense, Activation

def get_model():
    model = Sequential()
    model.add(Dense(12, input_shape=(12,)))
    model.add(Activation('relu'))
    model.add(Dense(12))
    model.add(Activation('softmax'))
    return model
