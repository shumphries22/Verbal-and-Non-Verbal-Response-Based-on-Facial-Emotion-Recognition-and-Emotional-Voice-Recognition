import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#function to extract features from audio file
def extractFeature(fileName, mfcc=True, chroma=True, mel=True, min_length=2048):
    try:
        with soundfile.SoundFile(fileName) as soundFile:
            X = soundFile.read(dtype="float32")
            sampleRate = soundFile.samplerate
            
            if len(X) < min_length:
                print(f"File too short ({len(X)} samples < {min_length}): {fileName}")
                return None
                
            features = []
            
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sampleRate, n_mfcc=40).T, axis=0)
                features.append(mfccs)
                
            if chroma:
                stft = np.abs(librosa.stft(X, n_fft=min(2048, len(X))))
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampleRate).T, axis=0)
                features.append(chroma)
                
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sampleRate).T, axis=0)
                features.append(mel)
                
            if features:
                return np.concatenate(features)
            return None
            
    except Exception as e:
        print(f"Error processing {fileName}: {str(e)}")
        return None

emotionList = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',       
    '05': 'angry',      
    '06': 'fearful',    
    '07': 'disgust',    
    '08': 'surprised'   
}

observedEmotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def loadData(datasetPath, testSize=0.2):
    x, y = [], []
    skippedFiles = 0
    
    for file in glob.glob(os.path.join(datasetPath, "Actor_*/*.wav")):
        try:
            fileName = os.path.basename(file)
            parts = fileName.split("-")
            if len(parts) < 3:
                continue
                
            emotion_code = parts[2]
            emotion = emotionList.get(emotion_code)
            
            if emotion not in observedEmotions:
                continue
                
            feature = extractFeature(file)
            if feature is not None:
                x.append(feature)
                y.append(emotion)
            else:
                skippedFiles += 1
        except Exception as e:
            print(f"Skipping {file} due to error: {str(e)}")
            skippedFiles += 1
            
    print(f"Skipped {skippedFiles} files due to errors or short length")
    return train_test_split(np.array(x), y, testSize=testSize, random_state=101)

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
def predictEmotion(model, audioPath):
    try:
        features = extractFeature(audioPath)
        if features is None:
            print("Feature extraction failed - file may be too short or corrupted")
            return None
            
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
        print(f"Prediction error: {str(e)}")
        return None

if __name__ == "__main__":
    datasetPath = "C:/Users/rjthornberry/Documents/RAVDESS" 
    modelPath = "emotion_recognition_model.pkl"
    
    if not os.path.exists(datasetPath):
        print(f"Dataset path not found: {datasetPath}")
        exit(1)
    
    print("Loading dataset and extracting features...")
    X_train, X_test, y_train, y_test = loadData(datasetPath, testSize=0.25)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("Error: No valid training data found")
        exit(1)
    
    print(f"\nData Summary:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Features extracted: {X_train.shape[1]}")
    
    print("\nTraining model...")
    model = trainModel(X_train, y_train)
    
    print("\nEvaluating model performance...")
    evaluateModel(model, X_test, y_test)
    
    saveModel(model, modelPath)
    
    testAudio = "C:/Users/rjthornberry/Downloads/harvard.wav"
    if os.path.exists(testAudio):
        print(f"\nPredicting emotion for {testAudio}...")
        loaded_model = loadSavedModel(modelPath)
        result = predictEmotion(loaded_model, testAudio)
        if result is None:
            print("Could not predict emotion - check audio file format and length")
    else:
        print("\nTest audio file not found - prediction skipped")
