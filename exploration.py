#
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# %matplotlib inline
#
#
# import seaborn
# from scipy.io import wavfile
#
# fs,x = wavfile.read("./data/test.wav")
#
# np.array(x)
#
# plt.plot(x)
#

#

import speech_recognition as sr

writing_file = open("first.txt","w")
r = sr.Recognizer()

while True:
    with sr.Microphone() as source:
        print("listening through: ",source)
        audio = r.listen(source)
        # my_audio_input = r.recognize_sphinx(audio)
        # print("my_audio_input is: ", my_audio_input)

        try:
            my_audio_input = r.recognize_sphinx(audio)

            print(my_audio_input)
            writing_file.write(my_audio_input)
            writing_file.write("\n")


        except sr.UnknownValueError:
            print("error...")
            pass
