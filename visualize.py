import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

# import seaborn
import scipy
from scipy.io import wavfile
import librosa
from librosa.feature import mfcc



df = pd.DataFrame({
    "happy":[wavfile.read("./data/wassim_voice.wav")],
    "angry":[wavfile.read("./alexandre_voice.wav")],
    "sad":[wavfile.read("./patrick_voice.wav")]
})

df.iloc[0]


sr_wassim, x_wassim = wavfile.read("./wassim_voice.wav")
sr_alexandre, x_alexandre = wavfile.read("./alexandre_voice.wav")
sr_patrick, x_patrick = wavfile.read("./patrick_voice.wav")


plt.plot(x_wassim)
plt.plot(x_alexandre)
plt.plot(x_patrick)


x_wassim_spec = plt.specgram(x_wassim,Fs=sr_wassim, xextent=(0,30))
x_patrick_spec = plt.specgram(x_patrick,Fs=sr_patrick, xextent=(0,30))
x_alexandre_spec = plt.specgram(x_alexandre,Fs=sr_alexandre, xextent=(0,30))


fft_wassim = abs(scipy.fft(x_wassim))
fft_alexandre = abs(scipy.fft(x_alexandre))
fft_patrick= abs(scipy.fft(x_patrick))


plt.plot(fft_wassim)
plt.plot(fft_alexandre)
plt.plot(fft_patrick)

mfccs_wassim = mfcc(y=x_wassim,sr=sr_wassim)
mfccs_alexandre = mfcc(y=x_alexandre,sr=sr_alexandre)
mfccs_patrick = mfcc(y=x_patrick,sr=sr_patrick)


mfccs_wassim.shape,mfccs_alexandre.shape,mfccs_patrick.shape
