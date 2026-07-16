from openai import OpenAI
from dotenv import load_dotenv
import wave
import threading
import tempfile
import os
import re
import arabic_reshaper
from bidi.algorithm import get_display
import pytest
pyaudio = pytest.importorskip("pyaudio")

# =========================
# OpenAI Setup
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Arabic Terminal Fix
# =========================
def fix_arabic_text(text):
    try:
        return get_display(arabic_reshaper.reshape(str(text)))
    except Exception:
        return str(text)

def print_ar(text):
    print(fix_arabic_text(text))

# =========================
# Audio Settings
# =========================
RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
DEVICES_TO_TRY = [14, 1, 7, 0, 15, 8, 2]

# =========================
# Record Audio
# =========================
def record_push_to_talk():
    input("\n🎤 Press Enter to START recording...")
    print("🔴 Recording... Press Enter to STOP")

    audio = pyaudio.PyAudio()
    stream = None

    for device_id in DEVICES_TO_TRY:
        try:
            stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=CHUNK
            )
            print(f"✅ Using microphone device: {device_id}")
            break
        except Exception:
            print(f"⚠️ Device {device_id} failed")

    if stream is None:
        audio.terminate()
        print("❌ No microphone worked.")
        return None

    frames = []
    stop_recording = False

    def stop_on_enter():
        nonlocal stop_recording
        input()
        stop_recording = True

    t = threading.Thread(target=stop_on_enter)
    t.daemon = True
    t.start()

    while not stop_recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.close()

    with wave.open(temp_file.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    print("✅ Recording saved")
    return temp_file.name

# =========================
# OpenAI Audio Transcription
# =========================
def transcribe_with_openai(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language="ar",
            response_format="text",
            prompt="""
هذا تسجيل لطفل أو مستخدم يتحدث باللهجة الفلسطينية أو الشامية.
اكتب الكلام كما هو بالعربية بدون ترجمة.
حافظ على اللهجة الفلسطينية مثل: بدي، هسا، شو، ليش، مش، منيح، خايف.
لا تضف شرحاً أو علامات غير موجودة.
"""
        )

    return transcription.strip()

# =========================
# NLP Preprocessing
# =========================
def normalize_arabic(text):
    text = str(text)

    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
    text = text.replace("ـ", "")

    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")

    text = re.sub(r"[^\u0600-\u06FF0-9\s؟،.!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def palestinian_postprocess(text):
    replacements = {
        "هلأ": "هسا",
        "هلا": "هسا",
        "شي": "إشي",
        "كويس": "منيح",
        "اناا": "انا",
        "ااه": "آه",
        "اه": "آه",
    }

    words = text.split()
    fixed_words = [replacements.get(w, w) for w in words]
    text = " ".join(fixed_words)

    text = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1 \1", text)
    return text.strip()

def remove_common_noise(text):
    noise_phrases = [
        "اشتركوا في القناة",
        "شكرا للمشاهدة",
        "ترجمة نانسي قنقر",
        "موسيقى",
        "تصفيق",
    ]

    for phrase in noise_phrases:
        text = text.replace(phrase, "")

    return re.sub(r"\s+", " ", text).strip()

def nlp_preprocess_pipeline(raw_text):
    text = remove_common_noise(raw_text)
    text = normalize_arabic(text)
    text = palestinian_postprocess(text)
    return text

# =========================
# Main Loop
# =========================
while True:
    audio_path = record_push_to_talk()

    if audio_path is None:
        break

    print("\n⏳ Transcribing with OpenAI Audio...")
    raw_text = transcribe_with_openai(audio_path)

    print("\n==============================")
    print("🟡 RAW OPENAI AUDIO TEXT")
    print("==============================")
    print_ar(raw_text)

    processed_text = nlp_preprocess_pipeline(raw_text)

    print("\n==============================")
    print("🟢 AFTER NLP PREPROCESSING")
    print("==============================")
    print_ar(processed_text)

    os.remove(audio_path)

    again = input("\nRecord another test? y/n: ")
    if again.lower() != "y":
        break

print("\n✅ Test finished.")