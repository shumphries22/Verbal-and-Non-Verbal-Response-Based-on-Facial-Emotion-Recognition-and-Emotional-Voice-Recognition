import speech_recognition as sr
import pyttsx3
import threading
import time
from datetime import datetime
from openai import OpenAI
import os

#global variables
currentText = []
silenceTimer = None
fileCounter = 1
isProcessing = False
recognizer = sr.Recognizer() 

#initialise ChatGPT
client = OpenAI(api_key= "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")

#initialise text to speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def processText(textContent):
    """Handle text processing and ChatGPT interaction"""
    global isProcessing, fileCounter
    
    isProcessing = True
    try:
        #save to file
        os.makedirs("conversations", exist_ok=True)
        filename = f"conversations/conversation_{fileCounter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(textContent)
        print(f"Saved conversation to {filename}")
        fileCounter += 1
        
        #get ChatGPT response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": textContent}]
        )
        chatgptResponse = response.choices[0].message.content
        print("ChatGPT:", chatgptResponse)
        speak(chatgptResponse)
        
    except Exception as e:
        print(f"Processing error: {e}")
    finally:
        isProcessing = False

def resetSilenceTimer():
    global silenceTimer
    if silenceTimer is not None:
        silenceTimer.cancel()
    silenceTimer = threading.Timer(5.0, checkSilence)
    silenceTimer.start()

def checkSilence():
     global currentText
     if currentText and not isProcessing:
        textToProcess = ' '.join(currentText)
        threading.Thread(target=processText, args=(textToProcess,)).start()
        currentText = []

def listenContinuously():
    """Main listening loop"""
    global currentText
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening continuously... Press Ctrl+C to stop")
        
        while True:
            try:
                if isProcessing:
                    time.sleep(0.1)
                    continue
                    
                print("\nReady for input...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    currentText.append(text)
                    resetSilenceTimer()
                    
                except sr.UnknownValueError:
                    print("Couldn't understand - try speaking again")
                    continue
                    
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    time.sleep(1)
                    continue
                    
            except KeyboardInterrupt:
                print("\nStopping...")
                if silenceTimer:
                    silenceTimer.cancel()
                if currentText:
                    processText(' '.join(currentText))
                break
                
            except Exception as e:
                print(f"Unexpected error: {e}")
                time.sleep(1)
                continue

if __name__ == "__main__":
    try:
        speak("System ready. Start speaking when ready.")
        listenContinuously()
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        print("Program ended")
