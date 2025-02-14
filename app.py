import streamlit as st
import faster_whisper
import pyaudio
import wave
import io
import os
import datetime
from googleapiclient.discovery import build
from google.oauth2 import service_account
from dateutil import parser
import pytz
import spacy
from nltk.tokenize import word_tokenize
import nltk

nltk.download(['punkt', 'punkt_tab'])
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# ✅ Load Whisper Model (Audio to Text Transcription)
whisper_model = faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")

# ✅ Google Calendar API Authentication
SCOPES = ['https://www.googleapis.com/auth/calendar']
SERVICE_ACCOUNT_FILE = "whisper-task-scheduler.json"  # Your service account JSON file

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
calendar_service = build("calendar", "v3", credentials=credentials)

# 🎤 Audio Recording Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048

audio = pyaudio.PyAudio()

def record_audio(duration=5):
    """🎙️ Record audio from microphone"""
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    st.write("🔴 Recording...")
    frames = []
    
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    st.write("🛑 Recording Stopped.")
    
    stream.stop_stream()
    stream.close()
    
    return b''.join(frames)

def extract_event_details(text):
    """Extract event details from the transcribed text"""
    # Using SpaCy for named entity recognition and NLTK for tokenizing
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    event_name = None
    event_date = None
    event_time = None
    
    # Extracting event name and date using keywords (assuming event name is the first part)
    event_name = text.split(" at ")[0]  # Assuming 'at' is used before the time in the transcription
    
    # Searching for date and time in the transcription
    for ent in doc.ents:
        if ent.label_ == "TIME":
            event_time = ent.text
        if ent.label_ == "DATE":
            event_date = ent.text
    
    if not event_date:  # If no date is found, default to tomorrow
        event_date = "tomorrow"
    
    return event_name, event_date, event_time

def handle_relative_date(event_date):
    """Handle relative dates like 'tomorrow'"""
    today = datetime.date.today()
    if event_date == "tomorrow":
        return (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    return event_date

def handle_relative_time(event_time, event_date):
    """Handle relative times like '5 pm tomorrow'"""
    if event_time and "tomorrow" in event_time:
        event_time = event_time.replace("tomorrow", "").strip()  # Removing 'tomorrow' from time
        event_date = handle_relative_date(event_date)  # Ensure the date is tomorrow
    return event_time, event_date

def convert_to_24hr_format(event_time):
    """Convert time in '12 pm' or '5 pm' format to 24-hour time format"""
    try:
        return datetime.datetime.strptime(event_time, '%I %p').strftime('%H:%M')
    except ValueError:
        return event_time

def add_event_to_calendar(event_name, event_date, event_time):
    """📅 Adds an event to Google Calendar with correct timezone handling"""
    if not event_name or not event_time:
        return "❌ Invalid event details"

    # Handle date formatting (e.g., tomorrow, next week, etc.)
    event_date = handle_relative_date(event_date)
    
    # Convert time to 24-hour format (e.g., 5 pm -> 17:00)
    event_time_parsed = convert_to_24hr_format(event_time)

    try:
        # Handle time formatting (ensure proper 24-hour format)
        event_time, event_date = handle_relative_time(event_time_parsed, event_date)  # Correct relative time

        event_datetime_str = f"{event_date} {event_time}"
        event_time_parsed = parser.parse(event_datetime_str).strftime("%H:%M")

    except Exception as e:
        return f"❌ Error parsing time: {e}"

    try:
        # Convert to datetime with explicit timezone
        karachi_tz = pytz.timezone("Asia/Karachi")
        
        # Combine date and time
        naive_dt = parser.parse(f"{event_date} {event_time_parsed}")
        
        # Localize to Karachi time
        local_dt = karachi_tz.localize(naive_dt)
        
        # Create event with 1-hour duration
        end_dt = local_dt + datetime.timedelta(hours=1)
        
        event = {
            'summary': event_name,
            'start': {
                'dateTime': local_dt.isoformat(),
                'timeZone': 'Asia/Karachi'
            },
            'end': {
                'dateTime': end_dt.isoformat(),
                'timeZone': 'Asia/Karachi'
            }
        }

        created_event = calendar_service.events().insert(
            calendarId='chayeshanaeem123@gmail.com',  # Use your calendar ID
            body=event
        ).execute()
        
        return f"✅ Event created successfully: {created_event.get('htmlLink')}"
        
    except Exception as e:
        return f"❌ Failed to create event: {str(e)}"

# Streamlit Interface
st.title("Whisper + Event Scheduling App")
st.write("Record audio, transcribe it using Whisper, and create a calendar event based on the extracted details.")

if st.button('Record and Analyze'):
    audio_data = record_audio(duration=10)  # Record for 10 seconds

    # 🎧 Convert raw bytes to WAV file
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)

    # 📝 Transcribe audio using Whisper model
    wav_buffer.seek(0)
    segments, _ = whisper_model.transcribe(wav_buffer, vad_filter=True)
    transcription = " ".join([segment.text for segment in segments])

    st.subheader("Transcribed Text:")
    st.write(transcription)

    # Extract event details from transcription
    event_name, event_date, event_time = extract_event_details(transcription)
    
    st.subheader("Extracted Event Details:")
    st.write(f"Event Name: {event_name}")
    st.write(f"Event Date: {event_date}")
    st.write(f"Event Time: {event_time}")

    # Add event to Google Calendar
    if event_name and event_time:
        calendar_response = add_event_to_calendar(event_name, event_date, event_time)
    else:
        calendar_response = "No event detected"

    st.subheader("Google Calendar Response:")
    st.write(calendar_response)
