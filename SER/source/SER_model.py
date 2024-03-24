import os
import sys
import glob
import numpy as np
import pandas as pd
import scipy.io.wavfile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential, Model, model_from_json
from keras.layers import (
    Dense, Embedding, LSTM, Input, Flatten, Dropout, Activation,
    Conv1D, MaxPooling1D, AveragePooling1D
)
from keras.preprocessing import sequence
from keras.utils import pad_sequences, to_categorical
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import librosa
import librosa.display
import tensorflow as tf

from matplotlib.pyplot import specgram


class SER_model:
    def __init__(self):
        self.loaded_model = None
        self.lb = LabelEncoder()
        self.lb.fit(
            ["female_angry",
             "female_calm",
             "female_fearful",
             "female_happy",
             "female_sad",
             "male_angry",
             "male_calm",
             "male_fearful",
             "male_happy",
             "male_sad"]
        )

    """
    Function to preprocess the audio file, <!only used internally!>

    Parameters:
    audio_path (str): path to the audio file

    Returns:
    feature (np.array): array of audio features

    Usage:
    self.preprocess_audio(audio_path)
    """

    def preprocess_audio(self, audio_path):
        X, sample_rate = librosa.load(
            audio_path,
            res_type="kaiser_fast",
            duration=2.5,
            sr=22050 * 2,
            offset=0.5,
        )
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(
            librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0
        )
        feature = mfccs
        return feature

    def model_predict(self, audio_path):
        feature = self.preprocess_audio(audio_path)
        feature_df = pd.DataFrame(data=feature)
        feature_df = feature_df.stack().to_frame().T
        feature_2d = np.expand_dims(feature_df, axis=2)
        livepreds = self.loaded_model.predict(
            feature_2d, batch_size=32, verbose=1
        )
        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        livepredictions = self.lb.inverse_transform(liveabc)
        return livepredictions

    def load_model(self, model_path, weights_path):
        json_file = open(model_path, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(weights_path)
        opt = keras.optimizers.RMSprop(learning_rate=0.00001)
        self.loaded_model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

# Path: SER/source/SER_model.py
# Compare this snippet from Facial_Emotion_Recognition/generate_imagetocsv.py:
#     "disgust": 1,
#     "fear": 2,
#     "happy": 3,
#     "sad": 4,
#     "surprise": 5,
#     "neutral": 6
# }
#
