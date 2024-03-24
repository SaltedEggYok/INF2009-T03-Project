import SER.source.SER_model as SER_model
import SER.source.SER_trainmodel as SER_trainmodel

if __name__ == "__main__":
    # train model
    ser_train = SER_trainmodel.SER_trainmodel()
    ser_train.add_audio_features("SER/datasets/our_data/")

    # run model
    ser_model = SER_model.SER_model()
    ser_model.load_model("SER/saved_models/Emotion_Voice_Detection_Model.h5")
    # ser_model.preprocess_audio("SER/datasets/our_data/")
    ser_model.model_predict("SER/datasets/our_data/(put file here)")
