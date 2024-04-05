import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import SER_utils


class SER_trainmodel:
    df_features = None
    feelings_list = SER_utils.FEELING_LIST
    df = None
    loaded_model = None

    def __init__(self):
        self.df = None
        self.loaded_model = None
        self.lb = LabelEncoder()
        self.lb.fit(self.feelings_list)

        # initialize dataframe
        self.df_features = pd.DataFrame(columns=["feature"])

    def clear_audio_features(self):
        self.df_features = pd.DataFrame(columns=["feature"])

    """
    Function to add audio features to the dataframe

    Parameters:
    audio_path (str): path to the audio files

    Returns:
    None

    Usage:
    ser = SER_trainmodel()
    ser.add_audio_features(audio_path)
    """

    def add_audio_features(self, audio_path):
        # access folder
        audio_files = os.listdir(audio_path)

        # df index counter
        counter = 0

        for index, y in enumerate(audio_files):
            # filter out unwanted files (only taking specfic feelings)
            if (audio_files[index][6:-16] != "01"
                and audio_files[index][6:-16] != "07"
                and audio_files[index][6:-16] != "08"
                and audio_files[index][:2] != "su"  # remove?
                and audio_files[index][:1] != "n"  # remove?
                and audio_files[index][:1] != "d"  # remove?
                ):
                # load audio file
                X, sample_rate = librosa.load(
                    audio_path + y,
                    res_type="kaiser_fast",
                    duration=2.5,
                    sr=22050 * 2,
                    offset=0.5,
                )
                # get sample rate
                sample_rate = np.array(sample_rate)
                # get mfccs - Mel-frequency cepstral coefficients
                feature = np.mean(librosa.feature.mfcc(
                    y=X, sr=sample_rate, n_mfcc=13), axis=0)
                self.df_features.loc[counter] = [feature]
                counter = counter + 1

    def train_model(self):
        # load json and create model
        json_file = open("SER/model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(
            "SER/saved_models/Emotion_Voice_Detection_Model.h5")

        # DEBUG
        print("Loaded model from disk")

        # compile the model
        self.loaded_model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )

        # get the features from the dataframe
        df3 = pd.DataFrame(self.df_features["feature"].values.tolist())

        # get the labels
        labels = pd.DataFrame(self.feelings_list)
        newdf = pd.concat([df3, labels], axis=1)

        # rename the columns
        rnewdf = newdf.rename(index=str, columns={"0": "label"})
        rnewdf = rnewdf.fillna(0)

        # get the features and labels
        X = rnewdf.iloc[:, :-1]
        y = rnewdf["label"]

        # encode the labels
        y = pd.get_dummies(y)

        # split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(X), np.array(y), test_size=0.2
        )

        # get the model #TODO: check if this is necessary
        # self.model = self.loaded_model

        # train the model
        self.loaded_model.fit(
            X_train,
            y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_test, y_test)
        )

        # save the model
        model_path = "SER/saved_models/Emotion_Voice_Detection_Model.h5"
        self.model.save(model_path)
        print("Model saved to disk at : " + model_path)
        return model_path


#         # X, sample_rate = librosa.load(
#         #     audio_path,
#         #     res_type="kaiser_fast",
#         #     duration=2.5,
#         #     sr=22050 * 2,
#         #     offset=0.5,
#         # )
#         # sample_rate = np.array(sample_rate)
#         # mfccs = np.mean(
#         #     librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0
#         # )
#         # feature = mfccs
#         # return feature


# mylist = os.listdir('RawData/')
# df = pd.DataFrame(columns=['feature'])
# bookmark = 0
# for index, y in enumerate(mylist):
#     if mylist[index][6:-16] != '01' and mylist[index][6:-16] != '07' and mylist[index][6:-16] != '08' and mylist[index][:2] != 'su' and mylist[index][:1] != 'n' and mylist[index][:1] != 'd':
#         X, sample_rate = librosa.load(
#             'RawData/'+y, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
#         sample_rate = np.array(sample_rate)
#         mfccs = np.mean(librosa.feature.mfcc(y=X,
#                                              sr=sample_rate,
#                                              n_mfcc=13),
#                         axis=0)
#         feature = mfccs
#         # [float(i) for i in feature]
#         # feature1=feature[:135]
#         df.loc[bookmark] = [feature]
#         bookmark = bookmark+1
