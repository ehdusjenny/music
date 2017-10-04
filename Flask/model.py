from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM

def get_model(file_name):
    if os.path.isfile(file_name):
        return load_model(file_name)
    model = Sequential()
    model.add(Dense(12, input_shape=(12,)))
    model.add(Activation('relu'))
    model.add(Dense(12))
    model.add(Activation('softmax'))
    return model

def get_lstm_model(): #???
    model = Sequential()
    model.add(LSTM(32, input_shape=(1,), return_sequences=True))
    model.add(LSTM(12))
    return model
