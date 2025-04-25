import os
import time
import tempfile
import numpy as np
import gradio as gr
import logging
import json
import pyaudio
import wave
import threading
from main import HealthcareAIAssistant

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Recording variables
recording_state = {"is_recording": False, "audio_path": None, "frames": [], "recording_thread": None}

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024


def create_gradio_interface():
    assistant = HealthcareAIAssistant()

    # Create language choices for dropdown
    language_choices = [tuple(item) for item in assistant.languages.items()]

    def record_audio(filename):
        """Function to record audio to a file"""
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        logging.info("* Recording started")

        frames = []
        recording_state["frames"] = frames
        recording_state["is_recording"] = True

        # Record audio in chunks until stopped
        while recording_state["is_recording"]:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        logging.info("* Recording stopped")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recorded audio to the file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))

        logging.info(f"* Audio saved to {filename}")
        return filename

    def start_recording():
        """Start audio recording in a separate thread"""
        try:
            # Create a temporary file for recording
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio_filepath = temp_file.name
            temp_file.close()

            recording_state["audio_path"] = audio_filepath
            recording_state["is_recording"] = True

            # Start recording in a separate thread
            recording_thread = threading.Thread(target=record_audio, args=(audio_filepath,))
            recording_thread.daemon = True
            recording_state["recording_thread"] = recording_thread
            recording_thread.start()

            logging.info(f"Started recording to {audio_filepath}")
            return "⏹️ Stop", "Recording in progress...", None
        except Exception as e:
            logging.error(f"Recording error: {str(e)}")
            return "🎙️ Record", f"Error starting recording: {str(e)}", None

    def stop_recording():
        """Stop audio recording and return path"""
        try:
            recording_state["is_recording"] = False

            # Wait for recording thread to finish
            if recording_state["recording_thread"] and recording_state["recording_thread"].is_alive():
                recording_state["recording_thread"].join(timeout=3.0)

            # Check if file was created and has content
            if recording_state["audio_path"] and os.path.exists(recording_state["audio_path"]):
                file_size = os.path.getsize(recording_state["audio_path"])
                logging.info(f"Recorded audio file size: {file_size} bytes")

                if file_size > 0:
                    return "🎙️ Record", "Recording complete.", recording_state["audio_path"]
                else:
                    return "🎙️ Record", "Recording stopped but file is empty. Please try again.", None
            else:
                return "🎙️ Record", "No recording was saved.", None
        except Exception as e:
            logging.error(f"Stop recording error: {str(e)}")
            return "🎙️ Record", f"Error stopping recording: {str(e)}", None

    def process_audio(audio_path, language_tuple):
        """Process the audio file and generate results"""
        try:
            if not audio_path:
                return "Please record or upload audio first.", "", "{}", "", "", None

            # Show processing message
            gr.Info("Processing audio... Please wait.")

            # Log audio path and check if it exists
            logging.info(f"Processing audio from path: {audio_path}")
            if not os.path.exists(audio_path):
                return f"Audio file not found at {audio_path}", "", json.dumps(
                    {"error": "File not found"}), "", "", None

            # Extract language code from the tuple
            language = language_tuple[0] if isinstance(language_tuple, tuple) else "en"

            # Transcribe audio - show progress
            gr.Info("Transcribing audio...")
            transcription = assistant.transcribe_audio(audio_path, language)
            logging.info(f"Transcription result: {transcription[:100]}...")  # Log first 100 chars

            if transcription and not transcription.startswith("Error"):
                # Generate medical notes
                gr.Info("Generating medical notes...")
                medical_notes = assistant.analyze_text(transcription, language)

                # Generate EMR content
                gr.Info("Creating EMR content...")
                emr_content = assistant.generate_emr_content(transcription, language)

                # Generate prescription and explanation
                gr.Info("Generating prescription...")
                prescription_text, prescription_explanation = assistant.generate_prescription(medical_notes, language)

                # Generate speech for medical notes and prescription explanation
                gr.Info("Creating audio response...")
                tts_output_path = os.path.join(tempfile.gettempdir(), f"medical_notes_audio_{int(time.time())}.mp3")
                notes_and_explanation = medical_notes + "\n\n" + prescription_explanation
                assistant.text_to_speech_with_gtts(notes_and_explanation, tts_output_path, language)

                gr.Info("Processing complete!")
                return transcription, medical_notes, emr_content, prescription_text, prescription_explanation, tts_output_path
            else:
                return transcription, "", json.dumps({"error": "Transcription failed"}), "", "", None
        except Exception as e:
            logging.error(f"Process audio error: {str(e)}")
            return f"Error processing audio: {str(e)}", "", json.dumps({"error": str(e)}), "", "", None

    def process_text(text_input, language_tuple):
        """Process text input and generate results"""
        try:
            if not text_input:
                return "", "", "{}", "", "", None

            # Show processing message
            gr.Info("Processing your text input... Please wait.")

            # Extract language code from the tuple
            language = language_tuple[0] if isinstance(language_tuple, tuple) else "en"

            # Generate medical notes
            gr.Info("Generating medical notes...")
            medical_notes = assistant.analyze_text(text_input, language)

            # Generate EMR content
            gr.Info("Creating EMR content...")
            emr_content = assistant.generate_emr_content(text_input, language)

            # Generate prescription and explanation
            gr.Info("Generating prescription...")
            prescription_text, prescription_explanation = assistant.generate_prescription(medical_notes, language)

            # Generate speech for medical notes and prescription explanation
            gr.Info("Creating audio response...")
            tts_output_path = os.path.join(tempfile.gettempdir(), f"text_notes_audio_{int(time.time())}.mp3")
            notes_and_explanation = medical_notes + "\n\n" + prescription_explanation
            assistant.text_to_speech_with_gtts(notes_and_explanation, tts_output_path, language)

            gr.Info("Processing complete!")
            return text_input, medical_notes, emr_content, prescription_text, prescription_explanation, tts_output_path
        except Exception as e:
            logging.error(f"Process text error: {str(e)}")
            return text_input, f"Error processing text: {str(e)}", json.dumps({"error": str(e)}), "", "", None

    # Create a custom theme with medical colors
    medical_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
    ).set(
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_700",
        button_secondary_background_fill="*neutral_200",
        button_secondary_background_fill_hover="*neutral_300",
    )

    # Create the Gradio interface with improved UI
    with gr.Blocks(title="Healthcare AI Assistant", theme=medical_theme) as app:
        gr.Markdown("# 🏥 Healthcare AI Assistant")

        with gr.Tabs():
            with gr.TabItem("Voice Input"):
                with gr.Row():
                    with gr.Column(scale=1):
                        language_selector = gr.Dropdown(
                            choices=language_choices,
                            value=language_choices[0] if language_choices else ("en", "English"),
                            label="Language",
                            type="value"
                        )

                        with gr.Group():
                            with gr.Row():
                                start_btn = gr.Button("🎙️ Record", variant="primary", size="lg")
                                stop_btn = gr.Button("⏹️ Stop", variant="secondary", size="lg")

                            recording_status = gr.Textbox(label="Status", value="Ready to begin")
                            audio_path = gr.Textbox(visible=False)

                        gr.Markdown("---")

                        with gr.Accordion("Audio Input Options", open=False):
                            uploaded_audio = gr.Audio(type="filepath", label="Upload Audio File")

                        process_btn = gr.Button("🔄 Process Audio", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        transcript_output = gr.Textbox(label="Transcription", lines=4)

                        with gr.Tabs():
                            with gr.TabItem("Medical Notes"):
                                medical_notes_output = gr.Textbox(label="", lines=10)

                            with gr.TabItem("Prescription"):
                                prescription_output = gr.Textbox(label="Prescription", lines=3)
                                prescription_explanation_output = gr.Textbox(label="Explanation", lines=5)

                            with gr.TabItem("EMR Data"):
                                emr_output = gr.JSON(label="")

                        with gr.Row():
                            audio_output = gr.Audio(label="Audio Summary", visible=True)

            with gr.TabItem("Text Input"):
                with gr.Row():
                    with gr.Column(scale=1):
                        text_language = gr.Dropdown(
                            choices=language_choices,
                            value=language_choices[0] if language_choices else ("en", "English"),
                            label="Language",
                            type="value"
                        )
                        text_input = gr.Textbox(
                            label="Patient Description",
                            lines=8,
                            placeholder="Describe symptoms, medical history, and current condition..."
                        )
                        process_text_btn = gr.Button("🔄 Generate Medical Notes", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        text_transcript = gr.Textbox(label="Patient Description", lines=4)

                        with gr.Tabs():
                            with gr.TabItem("Medical Notes"):
                                text_medical_notes = gr.Textbox(label="", lines=10)

                            with gr.TabItem("Prescription"):
                                text_prescription_output = gr.Textbox(label="Prescription", lines=3)
                                text_prescription_explanation = gr.Textbox(label="Explanation", lines=5)

                            with gr.TabItem("EMR Data"):
                                text_emr_output = gr.JSON(label="")

                        with gr.Row():
                            text_audio_output = gr.Audio(label="Audio Summary", visible=True)

        # Footer
        gr.Markdown("---")
        with gr.Accordion("About", open=False):
            gr.Markdown("""
            ### Healthcare AI Assistant
            This application helps medical providers capture and process patient information efficiently.

            **Features:**
            - Speech-to-text for natural patient interviews
            - Medical note generation based on patient descriptions
            - EMR data structure creation
            - Prescription recommendations with patient explanations
            - Multilingual support
            """)

        # Set up event handlers
        start_btn.click(start_recording, inputs=[], outputs=[start_btn, recording_status, audio_path])

        stop_btn.click(
            stop_recording,
            inputs=[],
            outputs=[stop_btn, recording_status, audio_path]
        )

        # Process audio from recording or upload
        def process_from_any_source(recorded_path, uploaded_path, language_tuple):
            path_to_use = uploaded_path if uploaded_path else recorded_path
            if not path_to_use:
                return "No audio available. Please record or upload first.", "", json.dumps(
                    {"error": "No audio available"}), "", "", None
            return process_audio(path_to_use, language_tuple)

        process_btn.click(
            process_from_any_source,
            inputs=[audio_path, uploaded_audio, language_selector],
            outputs=[transcript_output, medical_notes_output, emr_output, prescription_output,
                     prescription_explanation_output, audio_output]
        )

        # Text input processing
        process_text_btn.click(
            process_text,
            inputs=[text_input, text_language],
            outputs=[text_transcript, text_medical_notes, text_emr_output, text_prescription_output,
                     text_prescription_explanation, text_audio_output]
        )

    return app


# Launch the app
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=False)  # Set share=False for security/privacy