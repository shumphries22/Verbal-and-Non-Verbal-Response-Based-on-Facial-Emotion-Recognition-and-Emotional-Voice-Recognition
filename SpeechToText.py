import speech_recognition as sr
import pyttsx3
import time
import threading
from datetime import datetime
from openai import OpenAI

r = sr.Recognizer()
engine = pyttsx3.init()
currentText = []
silenceTimer = None
fileCounter = 1

openAIClient = OpenAI(api_key="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")

def speak(text):
    engine.say(text)
    engine.runAndWait()

def saveText(textContent):
    global file_counter
    filename = f"conversation_{file_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w') as f:
        f.write(textContent)
    print(f"Saved conversation to {filename}")
    fileCounter += 1

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": textContent}]
        )
        chatgptResponse = response.choices[0].message.content
        print("ChatGPT:", chatgptResponse)
        speak(chatgptResponse)  
        return chatgptResponse
    
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def resetSilenceTimer():
    global silenceTimer
    if silenceTimer is not None:
        silenceTimer.cancel()
    silencetimer = threading.Timer(2.0, handleSilence)
    silenceTimer.start()

def handleSilence():
    global currentText
    if currentText: 
        textToSave = ' '.join(currentText)
        saveText(textToSave)
        currentText = []  

def recordText():
    while True:
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                print("Listening...")

                audio2 = r.listen(source2)

                resetSilenceTimer()

                MyText = r.recognize_google(audio2)

                return MyText

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("Unknown error occurred")

try:
    print("Speech to ChatGPT program started. Press Ctrl+C to stop.")
    while True:
        text = recordText()
        print("You said:", text)
        currentText.append(text)

except KeyboardInterrupt:
    if currentText:
        saveText(' '.join(currentText))
    if silenceTimer is not None:
        silenceTimer.cancel()
    print("\nProgram terminated")
