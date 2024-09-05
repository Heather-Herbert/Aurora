from PyQt5 import QtWidgets, QtGui
from scipy.io.wavfile import write
from transformers import AutoModelForCausalLM, AutoTokenizer

import pyttsx3
import sounddevice as sd
import sys
import threading
import time
import whisper

model_name = "gpt2"  # or another model of your choice
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# local_model_path = "./local_model/Meta-Llama3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(local_model_path)
# model = AutoModelForCausalLM.from_pretrained(local_model_path)

def process_request(text):
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a pad token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    # Set pad_token_id explicitly in the model config if it's not already set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare a contextual prompt
    prompt = f"You are a desktop assistant, please provide a detailed answer to the following question: {text}"

    # Tokenize the input and return both input_ids and attention_mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=100)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Generate the output with input_ids and attention_mask
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        min_length=30,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the output to a string, skipping special tokens
    response = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    print(prompt)
    print(response)

    return response


def listen():
    sample_rate = 44100  # Sample rate in Hz
    duration = 5  # Duration of recording in seconds
    output_file = "output_audio.wav"

    # Record audio from the microphone
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=2, dtype='int16')

    # Wait for the recording to complete
    sd.wait()

    # Save the recorded audio to a .wav file
    write(output_file, sample_rate, audio_data)


def process():
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe("output_audio.wav")
    return result["text"]

def say(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    engine.setProperty('voice', voices[25].id)  # Example: change the index as needed
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)
    engine.say(text)

    engine.runAndWait()


class SystemTrayApp(QtWidgets.QSystemTrayIcon):
    def __init__(self, icon, icon_running, parent=None):
        super().__init__(icon, parent)
        self.icon_idle = icon
        self.icon_running = icon_running
        self.task_running = False  # Track whether the task is running

        # Create the menu
        self.menu = QtWidgets.QMenu(parent)
        self.run_task_action = self.menu.addAction("Run Task")
        self.exit_action = self.menu.addAction("Exit")

        # Connect the menu actions
        self.run_task_action.triggered.connect(self.on_run_task)
        self.exit_action.triggered.connect(self.on_exit)

        # Set the menu to the tray icon
        self.setContextMenu(self.menu)

        # Connect the `activated` signal to handle left-click
        self.activated.connect(self.on_tray_icon_activated)

    def on_tray_icon_activated(self, reason):
        # Check if the left mouse button was clicked
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            self.on_run_task()

    def on_run_task(self):
        if not self.task_running:
            print("Task triggered")
            # Change the icon to indicate task is running
            self.setIcon(self.icon_running)

            # Disable the button to prevent it from being clicked twice
            self.run_task_action.setEnabled(False)

            # Run the task in a separate thread
            threading.Thread(target=self.run_task).start()

    def run_task(self):
        try:
            self.task_running = True
            # Simulate task process
            listen()
            processed_text = process()
            reply = process_request(processed_text)
            say(reply)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Task finished, reset icon and re-enable the button
            self.setIcon(self.icon_idle)
            self.run_task_action.setEnabled(True)
            self.task_running = False

    def on_exit(self):
        QtWidgets.qApp.quit()


def main():
    app = QtWidgets.QApplication(sys.argv)

    icon_idle = QtGui.QIcon(QtGui.QPixmap("idle_icon.png"))  # Load idle icon
    icon_running = QtGui.QIcon(QtGui.QPixmap("running_icon.png"))  # Load running icon

    tray_icon = SystemTrayApp(icon_idle, icon_running)

    # Show the tray icon
    tray_icon.show()

    # Run the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        # Start the application in the main thread
        main()

        # Keep the main thread alive if necessary
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Manual interruption. Exiting...")
