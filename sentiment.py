import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

# import seaborn
import scipy
from scipy.io import wavfile
import librosa
from librosa.feature import mfcc

sr = 24414

filename_index_max = 83;


# Fill up Dataframe with wav files
df = pd.DataFrame(columns=["original_wav","target"])
for i in range(83):

    df2 = pd.DataFrame([
        [wavfile.read("./data/happy_oaf/{}.wav".format(i+1))[1],0],
        [wavfile.read("./data/happy_yaf/{}.wav".format(i+1))[1],0],
        [wavfile.read("./data/angry_oaf/{}.wav".format(i+1))[1],1],
        [wavfile.read("./data/angry_yaf/{}.wav".format(i+1))[1],1],
        [wavfile.read("./data/sad_oaf/{}.wav".format(i+1))[1],2],
        [wavfile.read("./data/sad_yaf/{}.wav".format(i+1))[1],2]
    ],columns=["original_wav","target"])

    # print(df2.head())
    df = df.append(df2)


len(df)




# make every sound have the same length...
min_wav_length = df.original_wav.apply(lambda x: len(x)).min()

df["wav"] = df.original_wav.map(lambda x : x[:min_wav_length])

# x_wav = np.array(df.wav.tolist())
# x_wav.shape

plt.plot(df.wav.iloc[0])

df["fft"] = df.wav.map(lambda x : abs(scipy.fft(x)))

# x_fft = np.array(df.fft.tolist())
# x_fft.shape

df["mfccs"] = df.wav.apply(lambda x : mfcc(y=x,sr=sr))

mfccs = np.array(df.mfccs.tolist())
# mfccs.shape

# spec_happy = plt.specgram(df.wav.iloc[0],Fs=sr, xextent=(0,30))
# spec_angry = plt.specgram(df.wav.iloc[16],Fs=sr, xextent=(0,30))
# spec_sad = plt.specgram(df.wav.iloc[44],Fs=sr, xextent=(0,30))


# np.array(df.wav.apply(lambda x : x.tolist()).values).shape


# x,y = df["wav"].apply(lambda x : x.reshape(len(x),1).tolist()).values,df["target"].values
# x,y = x_fft,df["target"].values



x,y = mfccs, df.target.values


# dunno why i need to this...
y = [int(x) for x in y]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state=1)


from keras.models import Sequential
from keras.layers import Input,Dense,Dropout,Activation,BatchNormalization,Convolution1D,AveragePooling1D,MaxPooling1D,Flatten
from sklearn.metrics import classification_report,log_loss,zero_one_loss
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD,RMSprop,Adam
from keras.layers.recurrent import LSTM,GRU
from keras.callbacks import History
hist = History()

model = Sequential((
    Convolution1D(64, kernel_size=3, activation='relu', input_shape=(x[0].shape[0], x[0].shape[1])),
    # BatchNormalization(momentum=0.2),
    AveragePooling1D(),
    Convolution1D(64, kernel_size=3, activation='relu'),
    # BatchNormalization(momentum=0.2),
    AveragePooling1D(),
    Flatten(),
    Dense(128, activation='relu'),
    # BatchNormalization(momentum=0.2),
    Dense(128, activation='relu'),
    # BatchNormalization(),
    Dense(len(np.bincount(y)), activation='softmax'),
))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4),metrics=["accuracy"])

model.fit(x_train, to_categorical(y_train),
    epochs=10,
    batch_size=5,
    validation_data=(x_test, to_categorical(y_test)),
    shuffle=True,
    verbose=2,
    callbacks=[hist]
)


len(x_test)




# save model
filename = "models/cnn_sentiments.h5";
model.save(filename)
