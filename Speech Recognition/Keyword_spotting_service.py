import numpy as np
import tensorflow.keras as keras
import librosa

MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050  # 1 sec


class _Keyword_spotting_service:
    model = None
    _mappings = [
        "go"
    ]

    _instance = None

    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)  # (# segments, # coefficients)

        # convert 2d MFCCs array into 4d array ->(# samples, # segments , # coefficients , # channels)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # mack prediction
        predictions = self.model.predict(MFCCs)  # [[0.1,0.6,0.1,....]]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        # load the audio file
        signal, sr = librosa.load(file_path)
        # ensure consistency
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T


def Keyword_spotting_service():
    if _Keyword_spotting_service._instance is None:
        _Keyword_spotting_service._instance = _Keyword_spotting_service()
        _Keyword_spotting_service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_spotting_service._instance


if __name__ == "__main__":
    kss = Keyword_spotting_service()

    keyword1 = kss.predict("/Users/kumaransubramaniam/PycharmProjects/pythonProject/test/go.wav")

    print(f"predicted Keyword: {keyword1}")