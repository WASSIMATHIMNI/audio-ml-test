import sys
import speech_recognition as sr
# import sphinxbase
# import pocketsphinx

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

with open("{}_voice.wav".format(sys.argv[1]),"wb") as f:
    f.write(audio.get_wav_data())
