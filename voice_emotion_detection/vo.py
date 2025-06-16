import sounddevice as sd
import numpy as np
import librosa
from keras.models import load_model

def extract_features(y, sr=16000, max_len=160):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] > max_len:
        mfcc = mfcc[:, :max_len]
    else:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    return mfcc

def predict_voice_emotion(model_path=r"D:\multi_model_project\voice_emotion_detection\model_3.h5"):
    print("🎤 Recording voice emotion for 2 seconds...")
    SAMPLE_RATE = 16000
    DURATION = 2
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"❌ Failed to load voice emotion model: {e}")
        return "neutral"

    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)

    features = extract_features(audio)
    input_data = np.expand_dims(features, axis=0)  # shape (1,40,160)

    proba = model.predict(input_data, verbose=0)[0]
    pred = emotion_labels[np.argmax(proba)]
    print(f"🎯 Detected voice emotion: {pred}")
    return pred

if __name__ == "__main__":
    predict_voice_emotion()
