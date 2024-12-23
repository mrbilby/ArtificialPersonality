import pyttsx3
# Initialize the TTS engine
engine = pyttsx3.init()
# List available voices
voices = engine.getProperty('voices')
# Print available voices and select one
"""
for index, voice in enumerate(voices):
    print(f"Voice {index}: {voice.name} - {voice.id}")
"""
# Set the desired voice (change index as needed)
engine.setProperty('voice', voices[170].id)  # Change the index to choose a different voice
# Adjust the speech rate
engine.setProperty('rate', 170)  # Set the desired rate
# Generate the speech
engine.say("Hello, I'm Bob, your virtual assistant!")
engine.runAndWait()