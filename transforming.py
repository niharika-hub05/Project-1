import speech_recognition as sr
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label, Button, PhotoImage
import threading  # To run speech recognition in a separate thread to avoid UI freezing

# --- Configuration ---
MODEL_ID = "runwayml/stable-diffusion-v1-5"  # Choose a Stable Diffusion model
IMAGE_SAVE_PATH = "generated_image.png"
USE_CUDA = True  # Set to True if you have a CUDA-enabled GPU
# --- End Configuration ---


class SpeechToImageApp:
    def __init__(self, root):
        self.root = root
        root.title("Speech to Image Generator")
        root.geometry("600x600") # Adjusted size for better layout

        self.recognizer = sr.Recognizer()
        self.pipeline = None  # Initialize here to ensure it's accessible after initialization
        self.load_stable_diffusion() # Load Stable Diffusion at startup

        self.text_prompt = tk.StringVar()
        self.image = None # Initialize image attribute

        # --- UI Elements ---
        self.prompt_label = Label(root, text="Spoken Prompt:")
        self.prompt_label.pack(pady=(10, 0))

        self.prompt_entry = tk.Entry(root, textvariable=self.text_prompt, width=50)
        self.prompt_entry.pack(pady=5)

        self.generate_button = Button(root, text="Speak and Generate!", command=self.start_speech_recognition)
        self.generate_button.pack(pady=10)

        self.image_label = Label(root, text="Image will appear here")
        self.image_label.pack(pady=10)

        self.status_label = Label(root, text="Ready...", wraplength=550)  # Status label
        self.status_label.pack(side="bottom", pady=5)  # Position at the bottom

    def load_stable_diffusion(self):
        """Load the Stable Diffusion pipeline"""
        self.update_status("Loading Stable Diffusion model... This may take a while.")
        try:
            self.pipeline = StableDiffusionPipeline.from_pretrained(MODEL_ID)
            if USE_CUDA:
                try:
                    self.pipeline.to("cuda")
                    self.update_status("Stable Diffusion loaded successfully on GPU (CUDA).")
                except Exception as e:
                    self.update_status(f"Failed to load model to GPU: {e}.  Falling back to CPU. This will be much slower.")
            else:
                self.update_status("Stable Diffusion loaded successfully on CPU. GPU usage is disabled.")

        except Exception as e:
            self.update_status(f"Error loading Stable Diffusion: {e}")
            self.pipeline = None  # Disable pipeline
            self.generate_button.config(state=tk.DISABLED)  # Disable button if the pipeline didn't load.


    def start_speech_recognition(self):
        """Starts speech recognition in a separate thread"""
        threading.Thread(target=self.recognize_speech_and_generate, daemon=True).start()

    def recognize_speech_and_generate(self):
        """Recognizes speech, updates the prompt, and generates the image."""
        self.update_status("Listening...")  # Update Status

        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5)  # Adjusted timeout

            self.update_status("Processing speech...")
            recognized_text = self.recognize_google_speech(audio)

            if recognized_text:
                self.update_status(f"You said: {recognized_text}")
                self.text_prompt.set(recognized_text)  # Update Text Entry
                self.generate_image_from_text(recognized_text)
            else:
                self.update_status("Could not understand speech.")

        except sr.WaitTimeoutError:
            self.update_status("No speech detected during the timeout.")
        except Exception as e:
            self.update_status(f"Error during speech recognition: {e}")
        finally:
             self.update_status("Ready...")

    def recognize_google_speech(self, audio):
        """Recognizes speech using Google Speech Recognition"""
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            self.update_status("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            self.update_status(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def generate_image_from_text(self, prompt):
         """Generates an image from the given text prompt using Stable Diffusion."""

         if self.pipeline is None:
             self.update_status("Stable Diffusion not loaded. Cannot generate image.")
             return

         try:
             self.update_status("Generating image... this may take a moment.")
             image = self.pipeline(prompt).images[0]

             image.save(IMAGE_SAVE_PATH)
             self.display_image(image) # Directly pass the PIL Image
             self.update_status("Image generated successfully.")

         except Exception as e:
             self.update_status(f"Error generating image: {e}")

    def display_image(self, image):
        """Displays the generated image in the UI."""
        self.image = image
        tk_image = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=tk_image)
        self.image_label.image = tk_image  # Keep a reference!

    def update_status(self, message):
        """Updates the status label with the given message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()  # Force update to show the message immediately


if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechToImageApp(root)
    root.mainloop()