import speech_recognition as sr
# import sphinxbase
# import pocketsphinx

r = sr.Recognizer()
with sr.AudioFile("voice.wav") as source:
    audio = r.record(source)

try:
    print("you said: ", r.recognize_sphinx(audio))
except LookupError:
    print("Could not understand audio")
