import sys
import speech_recognition as sr
# import sphinxbase
# import pocketsphinx

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)


    with open("temp.wav","wb") as f:
        f.write(audio.get_wav_data())


import scipy
import numpy as np
from scipy.io import wavfile
from librosa.feature import mfcc
from keras.models import load_model
model = load_model("./models/cnn_sentiments.h5")

sr = 24414
wav_length = 31446

wav = wavfile.read("temp.wav")[1][:wav_length];

mfccs = np.array(mfcc(y=wav,sr=sr))
mfccs = np.expand_dims(mfccs,axis=0)




preds = model.predict(mfccs)
targets = ["happy","angry","sad"]
print(targets[np.argmax(preds)])
