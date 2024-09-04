import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import pyttsx3

wisper_model = whisper.load_model("base")
model_name = "EleutherAI/gpt-neo-1.3B"  # or another model of your choice
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def process_request(text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=50)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def listen():
    sample_rate = 44100  # Sample rate in Hz
    duration = 5  # Duration of recording in seconds
    output_file = "output_audio.wav"

    print("Recording...")

    # Record audio from the microphone
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=2, dtype='int16')

    # Wait for the recording to complete
    sd.wait()

    # Save the recorded audio to a .wav file
    write(output_file, sample_rate, audio_data)

    print(f"Recording finished. Saved to {output_file}")

def process():
    model = whisper.load_model("base")
    result = model.transcribe("output_audio.wav")
    return result["text"]



def listen_for_phrase(target_phrase):
    print(f"Listening for the phrase '{target_phrase}'...")

    # Define audio parameters
    sample_rate = 16000  # Whisper uses 16kHz as its sample rate

    def callback(indata, frames, time, status):
        """Callback function to process the audio blocks."""
        if status:
            print(status)

        # Convert the audio to a format Whisper can process
        audio_data = torch.from_numpy(indata).float()
        audio_data = audio_data.flatten().numpy()

        # Perform the speech recognition
        result = wisper_model.transcribe(audio_data, language="en", fp16=False)
        spoken_text = result['text'].strip().lower()

        print(f"You said: {spoken_text}")

        # Check if the recognized text contains the target phrase
        if target_phrase.lower() in spoken_text:
            print("Target phrase detected! Stopping...")
            raise sd.CallbackStop()

    # Start the audio stream and listen
    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, dtype='float32'):
        sd.sleep(-1)  # Keep the stream open until CallbackStop is raised

def say(text):

    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

    # Speak the text
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait()


if __name__ == "__main__":
    while True:
        try:
            listen()
            processed_text = process()
            Reply = process_request(processed_text)
            say(Reply)
        except KeyboardInterrupt:
            print("Manual interruption. Exiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            # Optionally, continue listening or handle the error as needed

