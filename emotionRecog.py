import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Emotion mapping (RAVDESS standard)
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Emotions to include in model
SELECTED_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    """Extract audio features with consistent dimensions"""
    try:
        with soundfile.SoundFile(file_path) as audio_file:
            audio = audio_file.read(dtype="float32")
            sr = audio_file.samplerate
            
            if len(audio) < 2048:  # Minimum length for analysis
                return None
                
            features = []
            
            # MFCC (40 coefficients)
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
                features.extend(mfccs)
            
            # Chroma (12 features)
            if chroma:
                stft = np.abs(librosa.stft(audio, n_fft=min(2048, len(audio))))
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
                features.extend(chroma)
            
            # Mel Spectrogram (128 features)
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
                features.extend(mel)
                
            return np.array(features)
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_and_prepare_data(dataset_path):
    """Load dataset and prepare features/labels"""
    features, labels = [], []
    
    for file in glob.glob(os.path.join(dataset_path, "Actor_*/*.wav")):
        try:
            filename = os.path.basename(file)
            emotion_code = filename.split("-")[2]
            emotion = EMOTION_MAP[emotion_code]
            
            if emotion not in SELECTED_EMOTIONS:
                continue
                
            feature = extract_features(file)
            if feature is not None:
                features.append(feature)
                labels.append(emotion)
                
        except Exception as e:
            print(f"Skipping {file}: {str(e)}")
    
    return np.array(features), np.array(labels)

def train_and_export_model(dataset_path, output_path="emotion_model.pkl"):
    """Main training function"""
    print("Loading data and extracting features...")
    X, y = load_and_prepare_data(dataset_path)
    
    if len(X) == 0:
        raise ValueError("No valid training data found")
    
    print(f"\nTraining on {len(X)} samples with {X.shape[1]} features each")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    
    print("\nTraining model...")
    model = MLPClassifier(
        alpha=0.01,
        batch_size=256,
        hidden_layer_sizes=(300,),
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True
    )
    model.fit(X_train, y_train)
    
    print("\nModel evaluation:")
    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    print(f"\nSaving model to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)
    
    return output_path

if __name__ == "__main__":
    DATASET_PATH = "D://uni//HRI//RAVDESS"
    MODEL_OUTPUT = "C://Users//rjthornberry//Desktop//emotion_model.pkl"  
    try:
        model_path = train_and_export_model(DATASET_PATH, MODEL_OUTPUT)
        print(f"\nModel successfully trained and saved to: {model_path}")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
