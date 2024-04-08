import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import SER_utils
from keras.utils import np_utils

class SER_trainmodel:
    '''
    Class to train the model for Speech Emotion Recognition
    '''
    
    df_features = None
    feelings_list = SER_utils.FEELING_LIST
    df = None
    loaded_model = None
    train_feelings = []

    def __init__(self):
        '''
        Constructor
        '''
        self.df = None
        self.loaded_model = None
        self.lb = LabelEncoder()
        self.lb.fit(self.feelings_list)
        
        self.train_feelings = []

        # initialize dataframe
        self.df_features = pd.DataFrame(columns=["feature"])

    def clear_audio_features(self):
        '''
        Function to clear the audio features dataframe
        :param None
        :return: None
        '''
        self.df_features = pd.DataFrame(columns=["feature"])

    def add_audio_features(self, audio_path):
        '''
        Function to add audio features to the dataframe
        :param audio_path: path to the audio files
        :type audio_path: str
        :return: None
        '''
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
                
                # add label to train_feelings
                if audio_files[index][18:-4] == "02":
                    if audio_files[index][6:-16] == "02":
                        self.train_feelings.append("female_calm")
                    elif audio_files[index][6:-16] == "03":
                        self.train_feelings.append("female_happy")
                    elif audio_files[index][6:-16] == "04":
                        self.train_feelings.append("female_sad")
                    elif audio_files[index][6:-16] == "05":
                        self.train_feelings.append("female_angry")
                    elif audio_files[index][6:-16] == "06":
                        self.train_feelings.append("female_fearful")
                else:
                    if audio_files[index][6:-16] == "02":
                        self.train_feelings.append("male_calm")
                    elif audio_files[index][6:-16] == "03":
                        self.train_feelings.append("male_happy")
                    elif audio_files[index][6:-16] == "04":
                        self.train_feelings.append("male_sad")
                    elif audio_files[index][6:-16] == "05":
                        self.train_feelings.append("male_angry")
                    elif audio_files[index][6:-16] == "06":
                        self.train_feelings.append("male_fearful")
                
                    
        #convert train_feelings to df
        self.train_feelings = pd.DataFrame(self.train_feelings)
                
          

    def train_model(self):
        '''
        Function to train the model
        :param None
        :return: model_path (str): path to the saved model
        '''
        # load json and create model
        json_file = open("SER/saved_models/model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights(
            "SER/saved_models_backup/Emotion_Voice_Detection_Model.h5")

        # DEBUG
        print("Loaded model from disk")

        opt = tf.keras.optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)
        # compile the model
        self.loaded_model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )


        # get the features from the dataframe
        df3 = pd.DataFrame(self.df_features["feature"].values.tolist())

        # get the labels
        labels = pd.DataFrame(self.train_feelings)
        newdf = pd.concat([df3, labels], axis=1)

        # rename the columns
        rnewdf = newdf.rename(index=str, columns={"0": "label"})
        rnewdf = rnewdf.fillna(0)
        
        newdf1 = np.random.rand(len(rnewdf)) < 0.7
        train = rnewdf[newdf1]
        test = rnewdf[~newdf1]
        trainfeatures = train.iloc[:, :-1]
        trainlabel = train.iloc[:, -1:]
        print(trainlabel)
        testfeatures = test.iloc[:, :-1]
        testlabel = test.iloc[:, -1:]
        
        
        X_train = np.array(trainfeatures)
        y_train = np.array(trainlabel)
        X_test = np.array(testfeatures)
        y_test = np.array(testlabel)

        lb = LabelEncoder()

        y_train = np_utils.to_categorical(lb.fit_transform(y_train))
        print(y_train)
        y_test = np_utils.to_categorical(lb.fit_transform(y_test))

        x_traincnn =np.expand_dims(X_train, axis=2)
        x_testcnn= np.expand_dims(X_test, axis=2)
        
        print(x_traincnn.shape, x_testcnn.shape, y_train.shape, y_test.shape)
        
        cnnhistory=self.loaded_model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))

        # save the model
        model_path = "SER/saved_models/Emotion_Voice_Detection_Model.h5"
        self.loaded_model.save(model_path)
        print("Model weights saved to disk at : " + model_path)
        return model_path


if __name__ == "__main__":
    ser = SER_trainmodel()
    ser.add_audio_features("SER/our_data/processed/")
    ser.train_model()
    # print(ser.df_features)
    # print(ser.feelings_list)
    # print(ser.df)
    # print(ser.loaded_model)
    # print(ser.lb)
    # print("Done!)