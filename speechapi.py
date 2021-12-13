import joblib

from flask import Flask, request, render_template
import json
import numpy as np

import librosa
import soundfile
import os, glob, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# recoding imports
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
########

from werkzeug.utils import redirect, secure_filename
from werkzeug.datastructures import  FileStorage

app = Flask(__name__,static_folder="/home/prince/project/Major_Project_Final UI/static")

# model = joblib.load(r'fuel_comsumption.pkl')
model = joblib.load(r'randomforestfull_model.pkl')


#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))

    return result
@app.route('/record',methods=['GET'])
def record():
    # Sampling frequency
    freq = 48000

    # Recording duration
    duration = 3

    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
                    samplerate=freq, channels=1)

    # Record audio for the given number of seconds
    sd.wait()

    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write("static/recording0.wav", freq, recording)

    # Convert the NumPy array to audio file
    wv.write("static/recording1.wav", recording, freq, sampwidth=2)
    feature=extract_feature("static/recording0.wav", mfcc=True, chroma=False, mel=False)
    print(feature.shape)
    pre = feature.reshape(1,-1)
    res = model.predict(pre)
    print(res)
    emotion = str(res)
    return render_template('home.html',emotion=emotion)


def contact():
    if request.method == 'GET':
        if request.form['submit_button'] == 'Do Something':
            pass # do something
        elif request.form['submit_button'] == 'Do Something Else':
            pass # do something else
        else:
            pass # unknown
    # elif request.method == 'GET':
    #     return render_template('contact.html', form=form)

@app.route('/uploadfile',methods=['GET','POST'])
def uploadfile():
    emotion = ""
    if request.method == 'POST':
        if "file" not in request.files:
            return redirect(request.url)
        f = request.files['file']
        if f.filename=="":
            return redirect(request.url)
        filePath = secure_filename(f.filename)
        f.save(filePath)
        print(filePath)
        feature=extract_feature(filePath, mfcc=True, chroma=False, mel=False)
        print(feature.shape)
        pre = feature.reshape(1,-1)
        res = model.predict(pre)
        print(res)
        emotion = str(res)
        return render_template('home.html',emotion=emotion)
    else:
        if "file" not in request.files:
            return render_template('home.html',emotion=emotion)
        return "File Upload Failed!"
    # return render_template('home.html')

@app.route('/')
def home():
   return render_template('home.html',emotion="")

if __name__=='__main__':
    app.run(debug=True)
