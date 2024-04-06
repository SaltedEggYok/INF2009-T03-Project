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
import SER_utils

"""
Class to predict the emotion from the audio file

Attributes:
loaded_model (keras.model): model to predict the emotion
lb (LabelEncoder): label encoder for the emotions

Methods:
__init__(self): constructor
load_model(self, model_path, weights_path): load the model
preprocess_audio(self, audio_path): preprocess the audio file (internal use only)
model_predict(self, audio_path): predict the emotion

"""


class SER_model:

    # declare variables
    loaded_model = None  # holds the model
    lb = LabelEncoder()  # label encoder for emotions
    """
    Schema for the results variable:
    results = {
        timestamp: {
            userID : emotion
        }
    }
    """
    results = None  # holds the results of predictions

    def __init__(self):
        self.loaded_model = None
        self.lb = LabelEncoder()
        self.lb.fit(SER_utils.FEELING_LIST)
        self.results = None

    """
    load_model:
    Function to load the model and weights

    Parameters:
    model_path (str): path to the model file
    weights_path (str): path to the weights file

    Returns:
    None

    Usage:
    ser_model = SER_model()
    ser_model.load_model(model_path, weights_path)
    """

    def load_model(self, model_path, weights_path):
        # load json and create model
        json_file = open(model_path, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # load weights into model
        self.loaded_model.load_weights(weights_path)

        # compile the model
        opt = keras.optimizers.RMSprop(learning_rate=0.00001)

        # evaluate loaded model on test data
        self.loaded_model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

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
        # get librosa features
        X, sample_rate = librosa.load(
            audio_path,
            res_type="kaiser_fast",
            duration=2.5,
            sr=22050 * 2,
            offset=0.5,
        )
        # augmenting data
        sample_rate = np.array(sample_rate)
        feature = np.mean(librosa.feature.mfcc(
            y=X, sr=sample_rate, n_mfcc=13), axis=0)
        return feature

    """
    Function to predict the emotion from the audio file

    Parameters:
    audio_path (str): path to the audio file

    Returns:
    livepredictions (str): predicted emotion

    """

    def model_predict(self, audio_path, timestamp):
        # preprocess the audio, get features
        feature = self.preprocess_audio(audio_path)

        # augmenting data, converting to df
        feature_df = pd.DataFrame(data=feature)
        feature_df = feature_df.stack().to_frame().T
        feature_2d = np.expand_dims(feature_df, axis=2)

        # predict the emotion, calling the model
        livepreds = self.loaded_model.predict(
            feature_2d, batch_size=32, verbose=1)
        print(livepreds)
        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        print(liveabc)

        # converting the prediction (interger form) to emotion (string form), might not be needed in final product
        livepredictions = self.lb.inverse_transform(liveabc)

        self.add_to_results(timestamp, liveabc[0])
        return livepredictions

    """
    Function to add the results to the results variable (for internal use)
    Schema for the results variable:
    results = {
        timestamp: {
            userID : emotion
        }
    }

    Parameters:
    timestamp (int): timestamp of the prediction
    userID (int): userID of the user
    emotion (str OR int???): emotion predicted

    Returns:
    None
    """

    def add_to_results(self, timestamp, emotion, userID=0):
        # declare a dictionary to store the results if it doesn't exist
        if self.results is None:
            self.results = {}

        # reformat the timestamp so the time can be a key in the dictionary
        hour, minute, second, _ = map(int, timestamp.split(':'))            
        time_tuple = (hour, minute, second)

        if time_tuple not in self.results:
            self.results[time_tuple] = {}

        # TODO : CHANGE THIS, CURRENT HARDCODED
        # generalize emotions according to FEELING_LIST
        if emotion == 1 or emotion == 6:  # calm
            # add the results to the dictionary
            self.results[time_tuple][userID] = "Neutral"
        elif emotion == 3 or emotion == 8:  # happy
            # add the results to the dictionary
            self.results[time_tuple][userID] = "Positive"
        elif emotion == 0 or emotion == 5 or emotion == 2 or emotion == 7 or emotion == 4 or emotion == 9:  # angry, fearful, sad
            # add the results to the dictionary
            self.results[time_tuple][userID] = "Negative"
        # # add the results to the dictionary
        # self.results[timestamp][userID] = emotion

    def get_results(self):
        return self.results

    def export_result(self, filename):
        # export the results to a txt file
        with open(filename, "w") as file:
            file.write(str(self.results))

    def sync_time(self):
        # open the final output file
        output_file = open("SER/pred/outputs.txt", "w")

        # access dictionary
        if len(self.results) > 0:
            # iterate through the dictionary
            for idx, (key, value) in enumerate(self.results.items()):
                # if it's the first element, set the starting time to be used to offset all other times
                if idx == 0:
                    starting_hour, starting_minute, starting_second = key
                    output_key = (0,0,0)
                    output_file.write(f"{output_key}: {value}\n")
                    continue
                
                # offset all times by the starting time
                hour, minute, second = key
                # if the hour is less than the starting hour, it means it's the next day
                if hour < starting_hour:
                    hour += 24  
                # if the minute is less than the starting minute, it means the hour is less than the starting hour, same for seconds
                if minute < starting_minute:
                    hour -= 1
                    minute += 60
                if second < starting_second:
                    minute -= 1
                    second += 60

                # write the offset time to the file
                output_key = (hour - starting_hour, minute - starting_minute, second - starting_second)
                output_file.write(f"{output_key}: {value}\n")

