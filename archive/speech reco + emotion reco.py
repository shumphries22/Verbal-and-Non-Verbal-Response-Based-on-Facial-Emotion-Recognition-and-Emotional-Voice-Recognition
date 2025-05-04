import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime
import openai
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import threading
import time

# Initialize APIs
openai.api_key = 'your-openai-api-key'  # Replace with your actual API key

# Load emotion recognition model (pretrained)
# You'll need to have a trained model saved as 'emotion_model.pkl'
try:
    with open('emotion_model.pkl', 'rb') as f:
        emotion_model = pickle.load(f)
    scaler = StandardScaler()
except FileNotFoundError:
    print("Warning: Emotion model not found. Emotion analysis will be disabled.")
    emotion_model = None

# Audio parameters
SAMPLE_RATE = 16000
SILENCE_DURATION = 2  # seconds of silence to stop recording
ENERGY_THRESHOLD = 500  # Adjust based on your microphone
MIN_RECORDING_LENGTH = 1  # seconds

# Global variables
is_recording = False
audio_frames = []
current_transcript = ""

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

def record_callback(indata, frames, time, status):
    """Callback function for sounddevice to record audio"""
    global is_recording, audio_frames
    if is_recording:
        audio_frames.append(indata.copy())

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

def get_chatgpt_response(transcript, emotion):
    """Get response from ChatGPT based on transcript and emotion"""
    prompt = f"The user said: '{transcript}'. "
    if emotion != "unknown":
        prompt += f"They seem to be feeling {emotion}. "
    prompt += "Please respond appropriately."
    
    try:
        response = openai.ChatCompletion.create(
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

def save_audio(frames, sample_rate, filename):
    """Save recorded audio to file"""
    audio_data = np.concatenate(frames, axis=0)
    sf.write(filename, audio_data, sample_rate)

def process_conversation():
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
    
    # Analyze emotion
    emotion = analyze_emotion(audio_file)
    
    print(f"\nTranscript: {current_transcript}")
    print(f"Detected Emotion: {emotion}")
    
    # Get ChatGPT response
    response = get_chatgpt_response(current_transcript, emotion)
    
    # Clear audio frames for next recording
    audio_frames = []
    
    return response, emotion

def monitor_silence():
    """Monitor for silence to end recording"""
    global is_recording
    
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
                                # Here you would add text-to-speech to speak the response
        time.sleep(0.1)

def start_recording():
    """Start recording audio"""
    global is_recording, audio_frames
    is_recording = True
    audio_frames = []
    print("\nRecording started... Speak now!")

def main():
    # Start silence monitoring thread
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

if __name__ == "__main__":
    main()
