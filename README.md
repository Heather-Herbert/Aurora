# Desktop Assistant with System Tray Icon

This project implements a desktop assistant with a system tray icon, voice recognition, and text-to-speech functionality. It uses various libraries like PyQt5 for GUI, Whisper for voice-to-text, Hugging Face Transformers for generating responses, and pyttsx3 for text-to-speech synthesis.

**Due to the limitations of everything running locally, this requires a high end computer just to get running with a decent model. GPT2 (the current defult) gives to very strange answers.
Think of this more as a proof of concept then a finished product.**

## Features

-   **System Tray Icon**: A tray icon is available for easy access. Users can trigger the assistant by clicking on it.
-   **Voice Recognition**: Records a short voice clip (5 seconds by default) using the system's microphone and processes it into text using Whisper.
-   **Text Processing**: The assistant takes the transcribed voice text and generates a response using a Hugging Face transformer model.
-   **Text-to-Speech**: The response is spoken back to the user using pyttsx3's text-to-speech engine.

## Dependencies

This project requires the following Python packages:

-   `PyQt5`: For creating the system tray icon and the GUI components.
-   `whisper`: For voice-to-text transcription.
-   `transformers`: Hugging Face's library for working with transformer models.
-   `sounddevice`: For recording audio from the microphone.
-   `scipy`: For writing recorded audio to a file.
-   `pyttsx3`: For text-to-speech functionality.

You can install these dependencies via `pip`:

```pip install PyQt5 whisper transformers sounddevice scipy pyttsx3```

Additionally, you need to have a pre-trained Hugging Face model for text processing, which will be loaded during runtime.

## How It Works

1.  **System Tray Icon**: Once the program starts, a system tray icon will appear, allowing users to interact with it through a context menu.
    
    -   Left-click on the tray icon to trigger the voice assistant.
    -   Right-click to view options such as "Run Task" (which triggers the assistant) and "Exit".
2.  **Voice Recognition**: When triggered, the assistant records a 5-second audio clip using the system's microphone.
    
3.  **Speech-to-Text**: The recorded audio is transcribed into text using the Whisper model.
    
4.  **Text Processing**: The transcribed text is passed through a Hugging Face transformer model, which generates a response.
    
5.  **Text-to-Speech**: The generated response is spoken out loud using `pyttsx3`.
    

## Usage

1.  Clone the repository:

```
git clone https://github.com/Heather-Herbert/Aurora.git 
cd Aurora
```

2. Install the dependencies:

```
pip install -r requirements.txt
```

3. Run the project:

```
python Aurora.py
```
When you run the application, a system tray icon will appear. You can left-click the tray icon to trigger the assistant. The assistant will listen to your question, process it, and respond back with an answer using text-to-speech.

### Configuration

-   **Audio Duration**: The assistant is configured to listen for 5 seconds by default. You can modify the `duration` in the `listen()` function if needed.
-   **Voice Settings**: The TTS voice and speed can be customized in the `say()` function by changing the `voice` and `rate` properties.

## Known Issues

-   The assistant may not work properly if the microphone is not available or permissions are not set correctly.
-   The default behavior of the assistant when generating responses uses `clean_up_tokenization_spaces=True` to ensure proper formatting.

## Contributions

Feel free to submit issues or pull requests if you have any improvements or find any bugs.
