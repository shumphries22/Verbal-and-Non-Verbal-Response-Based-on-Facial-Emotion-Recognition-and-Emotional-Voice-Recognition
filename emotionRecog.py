import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#function to extract features from audio file
def extract_feature(fileName, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(fileName) as soundFile:
        X = soundFile.read(dtype="float32")
        sample_rate = soundFile.samplerate
        
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

emotionList = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'fearful',
    '05': 'disgust',
    '06': 'sad',
    '07': 'surprised'
    }

observedEmotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def loadData(dataset_path, test_size=0.2):
    x, y = [], []
    for file in glob.glob(os.path.join(dataset_path, "Actor_*/*.wav")):
        file_name = os.path.basename(file)
        emotion_code = file_name.split("-")[2]
        emotion = emotions[emotion_code]
        
        if emotion not in observedEmotions:
            continue
            
        try:
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
            
    return train_test_split(np.array(x), y, test_size=test_size, random_state=101)

def trainModel(X_train, y_train):
    model = MLPClassifier(
        alpha=0.01,
        batch_size=256,
        epsilon=1e-08,
        hidden_layer_sizes=(300,),
        learning_rate='adaptive',
        max_iter=500
    )
    model.fit(X_train, y_train)
    return model

def evaluateModel(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("Accuracy: {:.2f}%".format(accuracy*100))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

def saveModel(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def loadSavedModel(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

#function to predict emotion from audio file
def predictEmotion(model, audio_path):
    """Predict emotion from a new audio file"""
    try:
        features = extract_feature(audio_path)
        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        print("\nPrediction Results:")
        print(f"Predicted emotion: {prediction}")
        print("\nConfidence levels:")
        for emotion, prob in zip(model.classes_, probabilities):
            print(f"{emotion}: {prob:.2%}")
            
        return prediction
    except Exception as e:
        print(f"Error predicting emotion: {str(e)}")
        return None

if __name__ == "__main__":
    #configuration
    datasetPath = "/path/to/RAVDESS/Dataset"  # Update this path
    modelPath = "emotion_recognition_model.pkl"
    
    #load and prepare data
    print("Loading dataset and extracting features...")
    X_train, X_test, y_train, y_test = loadData(datasetPath, test_size=0.25)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features extracted: {X_train.shape[1]}")
    
    #train model
    print("\nTraining model...")
    model = trainModel(X_train, y_train)
    
    #evaluate model
    print("\nEvaluating model performance...")
    evaluateModel(model, X_test, y_test)
    
    #save model
    saveModel(model, modelPath)
    
    #example prediction
    TEST_AUDIO = "path/to/your/test_audio.wav"  # Replace with your test file
    if os.path.exists(TEST_AUDIO):
        print(f"\nPredicting emotion for {TEST_AUDIO}...")
        loaded_model = loadSavedModel(modelPath)
        predictEmotion(loaded_model, TEST_AUDIO)
    else:
        print("\nNo test audio file found for prediction demonstration.")