from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Conv1D
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam

def Triple_RNN(look_back,pred_len):
    model = Sequential()
    model.add(SimpleRNN(units=64,input_shape=(1,look_back), activation="relu", return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(Dense(150, activation='relu'))
    model.add(SimpleRNN(12))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(300,activation="relu"))
    model.add(Dense(150, activation="relu"))
    model.add(Dense(pred_len))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    return model

def Triple_RNN_LR_Decay(look_back,pred_len):
    model = Sequential()
    model.add(SimpleRNN(units=64,input_shape=(1,look_back), activation="relu", return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(Dense(150, activation='relu'))
    model.add(SimpleRNN(12))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(300,activation="relu"))
    model.add(Dense(150, activation="relu"))
    model.add(Dense(pred_len))
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-4,decay_steps=10000,decay_rate=0.9)
    optimizer = Adam(lr_schedule)
    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    return model

def Double_RNN(look_back, pred_len):
    model = Sequential()
    model.add(SimpleRNN(units=64,input_shape=(1,look_back), activation="relu", return_sequences=True))
    model.add(SimpleRNN(32))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(300,activation="relu"))
    model.add(Dense(150, activation="relu"))
    model.add(Dense(pred_len))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    return model

def Double_RNN_LR_Decay(look_back, pred_len):
    model = Sequential()
    model.add(SimpleRNN(units=64,input_shape=(1,look_back), activation="relu", return_sequences=True))
    model.add(SimpleRNN(32))
    model.add(Dense(150, activation='relu'))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(300,activation="relu"))
    model.add(Dense(150, activation="relu"))
    model.add(Dense(pred_len))
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-4,decay_steps=10000,decay_rate=0.9)
    optimizer = Adam(lr_schedule)
    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    return model

def Short_Deep_Net(look_back, pred_len):
    model = Sequential()
    model.add(SimpleRNN(units=256,input_shape=(1,look_back), activation="relu"))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(pred_len))
    optimizer = Adam()
    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    return model

def Short_Deep_LR_Decay(look_back, pred_len):
    model = Sequential()
    model.add(SimpleRNN(units=256,input_shape=(1,look_back), activation="relu"))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(pred_len))
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-4,decay_steps=10000,decay_rate=0.9)
    optimizer = Adam(lr_schedule)
    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    return model

def Simple_RNN_LR_Decay(look_back, pred_len):
    model = Sequential()
    model.add(SimpleRNN(units=16,input_shape=(1,look_back), activation="relu"))
    model.add(Dense(12,activation="relu"))
    model.add(Dense(pred_len))
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-4,decay_steps=10000,decay_rate=0.9)
    optimizer = Adam(lr_schedule)
    model.compile(loss='mae', optimizer=optimizer, metrics=['accuracy'])
    return model

def Simple_RNN(look_back, pred_len):
    model = Sequential()
    model.add(SimpleRNN(units=16,input_shape=(1,look_back), activation="relu"))
    model.add(Dense(12,activation="relu"))
    model.add(Dense(pred_len))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    return model

# Return specified model
def getModel(look_back,pred_len, fcn=Double_RNN_LR_Decay):
    return fcn(look_back,pred_len)



