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
from keras.preprocessing import sequence, text
from keras.utils import pad_sequences, to_categorical
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import librosa
import librosa.display
import tensorflow as tf

from matplotlib.pyplot import specgram

dir = "SER/"
# list of directories for the audio files
# audio_path = "SpeechEmotionRecognition/data/testdata/Actor_24/"
# mylist = os.listdir(audio_path)

# print(mylist) #DEBUG

# getting features from audio files
# df = pd.DataFrame(columns=["feature"])
# counter = 0
# for index, y in enumerate(mylist):
#     if (
#         mylist[index][6:-16] != "01"
#         and mylist[index][6:-16] != "07"
#         and mylist[index][6:-16] != "08"
#         and mylist[index][:2] != "su"
#         and mylist[index][:1] != "n"
#         and mylist[index][:1] != "d"
#     ):
#         X, sample_rate = librosa.load(
#             audio_path + y,
#             res_type="kaiser_fast",
#             duration=2.5,
#             sr=22050 * 2,
#             offset=0.5,
#         )
#         sample_rate = np.array(sample_rate)
#         mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
#         feature = mfccs
#         # [float(i) for i in feature]
#         # feature1=feature[:135]
#         df.loc[counter] = [feature]
#         counter = counter + 1

# print("loaded audio features")

# df3 = pd.DataFrame(df["feature"].values.tolist())
#newdf = pd.concat([df3, labels], axis=1)

#rnewdf = newdf.rename(index=str, columns={"0": "label"})
#rnewdf = rnewdf.fillna(0)

# loading json and creating model

json_file = open(dir + "model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(
    dir + "saved_models/Emotion_Voice_Detection_Model.h5"
)
print("Loaded model from disk")

opt = keras.optimizers.RMSprop(learning_rate=0.00001)
# evaluate loaded model on test data
loaded_model.compile(
    loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)
# score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# # fill the NaN values with 0
# filled_df = df.fillna(0)
# #following split data into test and train step, but no actual splitting
# test = filled_df
# #print(test)
# #loaded_model.summary()
# testfeatures = filled_df.iloc[:,:]
# test_set = np.array(testfeatures)
# expand_test_set = np.expand_dims(test_set, axis=2)
# #preds = loaded_model.predict(expand_test_set, batch_size=32, verbose=1)

# #print(preds)

# newdf1 = np.random.rand(len(rnewdf)) < 0.8
# train = rnewdf[newdf1]
# test = rnewdf[~newdf1]

# trainfeatures = train.iloc[:, :-1]
# trainlabel = train.iloc[:, -1:]
# testfeatures = test.iloc[:, :-1]
# testlabel = test.iloc[:, -1:]


#X_train = np.array(trainfeatures)
#y_train = np.array(trainlabel)
#X_test = np.array(testfeatures)
#y_test = np.array(testlabel)


#y_train = to_categorical(lb.fit_transform(y_train))
#y_test = to_categorical(lb.fit_transform(y_test))

#x_traincnn = np.expand_dims(X_train, axis=2)
#x_testcnn = np.expand_dims(X_test, axis=2)


# preds = loaded_model.predict(x_testcnn, batch_size=32, verbose=1)

# print(preds)

# preds1 = preds.argmax(axis=1)

# print(preds1)

# abc = preds1.astype(int).flatten()

# predictions = (lb.inverse_transform((abc)))

# testing 1 file directly

#livedf= pd.DataFrame(columns=['feature'])
X, sample_rate = librosa.load(dir + 'data/testdata/Actor_24/03-01-01-01-01-01-24.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
featurelive = mfccs
livedf2 = featurelive

livedf2= pd.DataFrame(data=livedf2)

livedf2 = livedf2.stack().to_frame().T

twodim= np.expand_dims(livedf2, axis=2)

livepreds = loaded_model.predict(twodim,
                         batch_size=32,
                         verbose=1)

print(livepreds)

livepreds1=livepreds.argmax(axis=1)

liveabc = livepreds1.astype(int).flatten()

print(liveabc)

lb = LabelEncoder()
lb.fit(["female_angry",
        "female_calm",
        "female_fearful",
        "female_happy",
        "female_sad",
        "male_angry",
        "male_calm",
        "male_fearful",
        "male_happy",
        "male_sad"])

livepredictions = (lb.inverse_transform((liveabc)))
print(livepredictions)
