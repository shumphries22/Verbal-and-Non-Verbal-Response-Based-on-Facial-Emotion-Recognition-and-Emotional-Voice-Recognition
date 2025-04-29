import cv2
import numpy as np
# import pyttsx3
import socket
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
from tensorflow.keras.models import load_model
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# GLOBAL VARIABLES
facial_emotion = "unknown"
is_recording = False
audio_frames = []
current_transcript = ""

# Initialize OpenAI API
client = OpenAI(api_key = "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")  # <-- replace with your actual key

# Load models
emotion_model_path = 'C://Users//rjthornberry//Documents//HRI//emotion_model.pkl'
try:
    with open(emotion_model_path, 'rb') as f:
        emotion_model = pickle.load(f)
    scaler = StandardScaler()  # Ideally, load scaler from model file
except FileNotFoundError:
    print(f"Warning: Emotion model not found at {emotion_model_path}. Audio emotion analysis disabled.")
    emotion_model = None

"""
json_path = 'D://uni//HRI//FaceModel//nNModel.json'
weights_path = 'D://uni//HRI//FaceModel//nNModel.weights.h5'
with open(json_path, 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(weights_path) 
"""

model = load_model('C://Users//rjthornberry//Documents//HRI//fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize text-to-speech engine
# engine = pyttsx3.init()

# Initialize webcam and face detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Audio parameters
SAMPLE_RATE = 16000
SILENCE_DURATION = 2
ENERGY_THRESHOLD = 500
MIN_RECORDING_LENGTH = 1

# --- AUDIO FUNCTIONS ---

def extract_features(audio_data, sample_rate):
    """Extract audio features."""
    features = []
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    tonnetz = librosa.feature.tonnetz(y=audio_data, sr=sample_rate)

    features.extend(np.mean(mfccs.T, axis=0))
    features.extend(np.mean(chroma.T, axis=0))
    features.extend(np.mean(mel.T, axis=0))
    features.extend(np.mean(contrast.T, axis=0))
    features.extend(np.mean(tonnetz.T, axis=0))
    
    return np.array(features)

def analyze_emotion(audio_file):
    """Analyze emotion from audio."""
    if emotion_model is None:
        return "unknown"
    try:
        audio_data, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
        features = extract_features(audio_data, sample_rate).reshape(1, -1)
        features = scaler.transform(features)  # assuming model expects scaled input
        emotion = emotion_model.predict(features)[0]
        return emotion
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return "unknown"

def transcribe_audio(audio_file):
    """Transcribe audio to text."""
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Speech not recognized.")
        return ""
    except sr.RequestError as e:
        print(f"Speech service error: {e}")
        return ""

def get_chatgpt_response(transcript, audio_emotion, facial_emotion):
    """Get response from ChatGPT."""
    prompt = f"The user said: '{transcript}'. "
    if audio_emotion != "unknown":
        prompt += f"They seem to feel {audio_emotion} based on tone. "
    if facial_emotion != "unknown":
        prompt += f"Their facial expression suggests {facial_emotion}. "
    prompt += "Please respond appropriately."

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
        print(f"OpenAI API error: {e}")
        return "Sorry, I couldn't process that."
    
def speak_via_pepper_socket(response, pepper_ip="169.254.156.17", port=10000):
    """Send the ChatGPT response to Pepper's TTS socket server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((pepper_ip, port))
            s.sendall(response.encode("utf-8"))
    except Exception as e:
        print(f"[Socket Error] Could not send to Pepper: {e}")

def record_callback(indata, frames, time_info, status):
    """Capture audio callback."""
    global is_recording, audio_frames
    if is_recording:
        audio_frames.append(indata.copy())

def save_audio(frames, sample_rate, filename):
    """Save recorded audio."""
    audio_data = np.concatenate(frames, axis=0)
    sf.write(filename, audio_data, sample_rate)

def start_recording():
    """Start recording."""
    global is_recording, audio_frames
    is_recording = True
    audio_frames = []
    print("Recording started... Speak now!")

# --- MONITORING FUNCTIONS ---

def monitor_silence():
    """Monitor silence to stop recording."""
    global is_recording, facial_emotion
    while True:
        if is_recording and audio_frames:
            current_audio = np.concatenate(audio_frames[-int(SAMPLE_RATE*0.5):], axis=0)
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
                    recording_duration = len(np.concatenate(audio_frames, axis=0)) / SAMPLE_RATE
                    if recording_duration >= MIN_RECORDING_LENGTH:
                        is_recording = False
                        response, _ = process_conversation(facial_emotion)
                        if response:
                            print(f"Assistant: {response}")
                            # engine.say(response)
                            # engine.runAndWait()
                            sock.sendall(response.encode('utf-8'))


        time.sleep(0.1)

def process_conversation(facial_emotion="unknown"):
    """Handle processing after recording."""
    global audio_frames, current_transcript

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = f"recording_{timestamp}.wav"
    save_audio(audio_frames, SAMPLE_RATE, audio_file)

    current_transcript = transcribe_audio(audio_file)
    if not current_transcript:
        print("No speech detected.")
        return None, None

    audio_emotion = analyze_emotion(audio_file)

    print(f"\nTranscript: {current_transcript}")
    print(f"Audio Emotion: {audio_emotion}")
    print(f"Facial Emotion: {facial_emotion}")

    response = get_chatgpt_response(current_transcript, audio_emotion, facial_emotion)
    audio_frames = []

    return response, (audio_emotion, facial_emotion)

def process_face_expression(frame):
    global facial_emotion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))  # <-- Resize to 64x64!
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)  # Shape: (1, 64, 64, 1)

        prediction = model.predict(face)
        emotion_label = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        facial_emotion = emotion_label
        text = f"{emotion_label} ({confidence*100:.1f}%)"

        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# --- MAIN LOOP ---

def main():
    server_address = ('169.254.156.17', 10000)  # Replace 'localhost' with your server IP if needed
    print('Connecting to {} port {}'.format(*server_address), file=sys.stderr)
    sock.connect(server_address)
    
    threading.Thread(target=monitor_silence, daemon=True).start()
    print("Interactive system ready.")
    print("Press Enter to start speaking...")
    try:
        while True:
            input("Press Enter to start recording...")
            if not is_recording:
                start_recording()
                with sd.InputStream(callback=record_callback, channels=1, samplerate=SAMPLE_RATE):
                    while is_recording:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = process_face_expression(frame)
                        cv2.imshow('Facial Expression Recognition', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
    finally:
        print('Closing socket', file=sys.stderr)
        sock.close()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
