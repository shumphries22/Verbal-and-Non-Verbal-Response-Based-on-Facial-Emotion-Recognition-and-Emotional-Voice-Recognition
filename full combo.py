import cv2
import numpy as np
import pyttsx3
import sounddevice as sd
import soundfile as sf
import librosa
from datetime import datetime
from openai import OpenAI
import os
import pickle
import threading
import time
import speech_recognition as sr
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json

facial_emotion = "unknown"

# Initialize APIs
client = OpenAI(api_key="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")  # Replace with your actual API key

# Load emotion recognition model for audio (pretrained)
emotion_model_path = 'D://Downloads//emotion_model.pkl'
try:
    with open(emotion_model_path, 'rb') as f:
        emotion_model = pickle.load(f)
    scaler = StandardScaler()
except FileNotFoundError:
    print(f"Warning: Emotion model not found at {emotion_model_path}. Emotion analysis will be disabled.")
    emotion_model = None

# Load facial expression recognition model
json_path = 'D://uni//HRI//FaceModel//nNModel.json'
weights_path = 'D://uni//HRI//FaceModel//nNModel.weights.h5'

with open(json_path, 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights(weights_path)

# Class labels for facial expression recognition
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Audio parameters
SAMPLE_RATE = 16000
SILENCE_DURATION = 2  # seconds of silence to stop recording
ENERGY_THRESHOLD = 500  # Adjust based on your microphone
MIN_RECORDING_LENGTH = 1  # seconds

# Global variables
is_recording = False
audio_frames = []
current_transcript = ""

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Functions for audio and speech processing
def extract_features(audio_data, sample_rate):
    """Extract audio features for emotion analysis"""
    features = []
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    features.extend(mfccs_mean)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)
    features.extend(chroma_mean)
    
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_mean = np.mean(mel.T, axis=0)
    features.extend(mel_mean)
    
    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    contrast_mean = np.mean(contrast.T, axis=0)
    features.extend(contrast_mean)
    
    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sample_rate)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)
    features.extend(tonnetz_mean)
    
    return np.array(features)

def analyze_emotion(audio_file):
    """Analyze emotion from audio file"""
    if emotion_model is None:
        return "unknown"
    
    try:
        # Load audio file
        audio_data, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
        
        # Extract features
        features = extract_features(audio_data, sample_rate)
        features = features.reshape(1, -1)
        features = scaler.transform(features)
        
        # Predict emotion
        emotion = emotion_model.predict(features)[0]
        return emotion
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return "unknown"

def transcribe_audio(audio_file):
    """Transcribe audio file to text"""
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def get_chatgpt_response(transcript, audio_emotion, facial_emotion):
    """Get response from ChatGPT based on transcript and two emotions"""
    prompt = f"The user said: '{transcript}'. "
    
    if audio_emotion != "unknown":
        prompt += f"They seem to be feeling {audio_emotion} based on their tone of voice. "
    
    if facial_emotion != "unknown":
        prompt += f"They also appear to be feeling {facial_emotion} based on their facial expression. "
    
    prompt += "Please respond appropriately based on both emotions and the transcript."
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting ChatGPT response: {e}")
        return "I'm sorry, I couldn't process that request."


def record_callback(indata, frames, time, status):
    """Callback function for sounddevice to record audio"""
    global is_recording, audio_frames
    if is_recording:
        audio_frames.append(indata.copy())

def save_audio(frames, sample_rate, filename):
    """Save recorded audio to file"""
    audio_data = np.concatenate(frames, axis=0)
    sf.write(filename, audio_data, sample_rate)

def start_recording():
    """Start recording audio"""
    global is_recording, audio_frames
    is_recording = True
    audio_frames = []
    print("\nRecording started... Speak now!")

def monitor_silence():
    """Monitor for silence to end recording"""
    global is_recording, facial_emotion
    
    while True:
        if is_recording:
            # Calculate current audio energy
            if audio_frames:
                current_audio = np.concatenate(audio_frames[-int(SAMPLE_RATE*0.5):], axis=0)  # last 0.5 second
                energy = np.sum(current_audio**2) / len(current_audio)
                
                if energy < ENERGY_THRESHOLD:
                    silence_start = time.time()
                    while time.time() - silence_start < SILENCE_DURATION:
                        time.sleep(0.1)
                        current_audio = np.concatenate(audio_frames[-int(SAMPLE_RATE*0.5):], axis=0)
                        energy = np.sum(current_audio**2) / len(current_audio)
                        if energy >= ENERGY_THRESHOLD:
                            break
                    else:
                        # If we've had continuous silence for SILENCE_DURATION seconds
                        recording_duration = len(np.concatenate(audio_frames, axis=0)) / SAMPLE_RATE
                        if recording_duration >= MIN_RECORDING_LENGTH:
                            is_recording = False
                            response, _ = process_conversation()
                            if response:
                                print(f"\nAssistant: {response}")
                                engine.say(response)
                                engine.runAndWait()  # Read out the response
        time.sleep(0.1)

def process_conversation(facial_emotion):
    """Main processing function"""
    global audio_frames, current_transcript, is_recording
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = f"recording_{timestamp}.wav"
    
    # Save and process audio
    save_audio(audio_frames, SAMPLE_RATE, audio_file)
    
    # Transcribe
    current_transcript = transcribe_audio(audio_file)
    if not current_transcript:
        print("No speech detected.")
        return None, None
    
    # Analyze emotion from audio
    audio_emotion = analyze_emotion(audio_file)
    
    # Analyze emotion from facial expression (get frame from webcam already being processed)
    print(f"\nTranscript: {current_transcript}")
    print(f"Detected Audio Emotion: {audio_emotion}")
    print(f"Detected Facial Emotion: {facial_emotion}")
    
    # Get ChatGPT response with both emotions
    response = get_chatgpt_response(current_transcript, audio_emotion, facial_emotion)
    
    # Clear audio frames for next recording
    audio_frames = []
    
    return response, (audio_emotion, facial_emotion)


def process_face_expression(frame):
    """Process face expression using webcam"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    facial_emotion = "unknown"  # Default value
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Preprocess face for prediction
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)  # add channel dimension

        # Predict expression
        prediction = model.predict(face)
        facial_emotion = class_labels[np.argmax(prediction)]
        cv2.putText(frame, facial_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame, facial_emotion

# Initialize webcam
# cap = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def main():
    # Start audio recording thread
    threading.Thread(target=monitor_silence, daemon=True).start()
    
    print("Interactive conversation system ready.")
    print("Press Enter to start recording, then speak. The system will automatically detect when you stop talking.")
    
    while True:
        input("Press Enter to start recording...")
        if not is_recording:
            start_recording()
            # Start audio stream
            with sd.InputStream(callback=record_callback, channels=1, samplerate=SAMPLE_RATE):
                while is_recording:
                    time.sleep(0.1)

        # Real-time webcam emotion detection
        ret, frame = cap.read()
        if not ret:
            break
        frame, facial_emotion = process_face_expression(frame)
        cv2.imshow('Facial Expression Recognition', frame)

        # Handle conversation after silence detected
        if not is_recording:
            response, _ = process_conversation(facial_emotion)
            if response:
                print(f"Assistant: {response}")
                engine.say(response)
                engine.runAndWait()  # Read out the response

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
