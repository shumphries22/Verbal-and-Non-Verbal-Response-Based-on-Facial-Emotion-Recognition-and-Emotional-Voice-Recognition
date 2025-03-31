import speech_recognition as sr
import pyttsx3
import threading
from datetime import datetime, timedelta
from openai import OpenAI
import sys
import time

#initialize OpenAI client
client = OpenAI(api_key="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")  # Replace with your actual API key

#initialize text-to-speech
engine = pyttsx3.init()

#global variables
currentText = []
silenceTimer = None
filecounter = 1
ignoreUntil = None

def speak(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def saveAndProcessText(textContent):
    global fileCounter, ignoreUntil
    try:
        #save to file
        filename = f"conversation_{fileCounter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(textContent)
        print(f"Saved conversation to {filename}")
        fileCounter += 1
        
        # Set ignore period (now + 20 seconds)
        ignoreUntil = datetime.now() + timedelta(seconds=20)
        print(f"Ignoring speech until {ignoreUntil.strftime('%H:%M:%S')}")
        
        # Send to ChatGPT
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": textContent}]
        )
        chatgptResponse = response.choices[0].message.content
        print("ChatGPT:", chatgptResponse)
        speak(chatgptResponse)
        return chatgptResponse
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return None

def resetSilenceTimer():
    global silenceTimer
    if silenceTimer is not None:
        silenceTimer.cancel()
    silenceTimer = threading.Timer(2.0, handleSilence)
    silenceTimer.start()

def handleSilence():
    global currentText, ignoreUntil
    if currentText and (ignoreUntil is None or datetime.now() > ignoreUntil):
        textToProcess = ' '.join(currentText)
        saveAndProcessText(textToProcess)
        currentText = []

def checkMicrophone():
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Microphone test successful")
            return r
    except Exception as e:
        print(f"Microphone error: {e}")
        speak("Microphone not detected. Please check your microphone connection.")
        sys.exit(1)

def recordText(r):
    global ignoreUntil
    while True:
        try:
            #check if we're in ignore period
            if ignoreUntil and datetime.now() < ignoreUntil:
                time.sleep(0.1)
                continue
                
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                print("Listening...")
                audio2 = r.listen(source2)
                
                if ignoreUntil and datetime.now() < ignoreUntil:
                    continue
                    
                resetSilenceTimer()
                
                MyText = r.recognize_google(audio2)
                return MyText

        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except sr.UnknownValueError:
            print("Silence detected, waiting...")
            continue

#main program
if __name__ == "__main__":
    try:
        print("Initializing speech-to-ChatGPT program...")
        recognizer = checkMicrophone()
        speak("System ready. You may begin speaking.")
        
        while True:
            text = recordText(recognizer)
            if ignoreUntil is None or datetime.now() > ignoreUntil:
                print("You said:", text)
                currentText.append(text)
            else:
                print("Ignoring speech during cooldown period")

    except KeyboardInterrupt:
        if currentText and (ignoreUntil is None or datetime.now() > ignoreUntil):
            saveAndProcessText(' '.join(currentText))
        if silenceTimer is not None:
            silenceTimer.cancel()
        print("\nProgram terminated gracefully")
