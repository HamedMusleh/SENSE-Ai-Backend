import pyaudio
import wave
import threading
import tempfile
import os
import re
import whisper
import arabic_reshaper
from bidi.algorithm import get_display

# =========================
# Arabic Terminal Fix
# =========================
def fix_arabic_text(text):
    try:
        reshaped_text = arabic_reshaper.reshape(str(text))
        return get_display(reshaped_text)
    except Exception:
        return str(text)

def print_ar(text):
    print(fix_arabic_text(text))


# =========================
# Audio Settings
# =========================
RATE = 16000          # أفضل لـ Whisper
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16

# جرّب هاي الأجهزة وعدلها حسب جهازك
DEVICES_TO_TRY = [14, 1, 7, 0, 15, 8, 2]


# =========================
# Load Whisper
# =========================
print("⏳ Loading Whisper model...")
model = whisper.load_model("medium")   # جرب: small / medium
print("✅ Whisper ready")


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
# Whisper Transcription
# =========================
def transcribe_with_whisper(audio_path):
    result = model.transcribe(
        audio_path,
        language="ar",
        task="transcribe",
        fp16=False,
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.45,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4
    )
    return result["text"].strip()


# =========================
# NLP Preprocessing Layer
# =========================
def normalize_arabic(text):
    text = str(text)

    # إزالة التشكيل
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)

    # إزالة التطويل
    text = text.replace("ـ", "")

    # توحيد الحروف
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")

    # تنظيف رموز غريبة
    text = re.sub(r"[^\u0600-\u06FF0-9\s؟،.!]", " ", text)

    # مسافات
    text = re.sub(r"\s+", " ", text).strip()

    return text


def palestinian_postprocess(text):
    """
    تصحيح خفيف للهجة الفلسطينية بدون تغيير المعنى بقوة.
    الهدف تحسين مخرجات Whisper فقط.
    """

    replacements = {
        "اناا": "انا",
        "اه": "آه",
        "ااه": "آه",
        "يعني يعني": "يعني",
        "بس بس": "بس",
        "كتير كتير": "كتير",
        "اشي": "إشي",
        "شي": "إشي",
        "بديش": "بديش",
        "مش": "مش",
        "ليش": "ليش",
        "شو": "شو",
        "هسا": "هسا",
        "هلأ": "هسا",
        "هلا": "هسا",
        "منيح": "منيح",
        "كويس": "منيح",
        "خايف": "خايف",
        "خايفة": "خايفة",
        "زعلان": "زعلان",
        "زعلانة": "زعلانة",
        "تعبان": "تعبان",
        "تعبانة": "تعبانة",
    }

    words = text.split()
    fixed_words = []

    for w in words:
        fixed_words.append(replacements.get(w, w))

    text = " ".join(fixed_words)

    # حذف تكرار نفس الكلمة أكثر من مرتين
    text = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1 \1", text)

    return text.strip()


def remove_common_whisper_noise(text):
    noise_phrases = [
        "ترجمة نانسي قنقر",
        "ترجمة نانسي قنقر",
        "اشتركوا في القناة",
        "شكرا للمشاهدة",
        "موسيقى",
        "تصفيق",
    ]

    for phrase in noise_phrases:
        text = text.replace(phrase, "")

    text = re.sub(r"\s+", " ", text).strip()
    return text


def nlp_preprocess_pipeline(raw_text):
    step1 = remove_common_whisper_noise(raw_text)
    step2 = normalize_arabic(step1)
    step3 = palestinian_postprocess(step2)
    return step3


# =========================
# Main Test Loop
# =========================
while True:
    audio_path = record_push_to_talk()

    if audio_path is None:
        break

    print("\n⏳ Transcribing with Whisper...")
    raw_text = transcribe_with_whisper(audio_path)

    print("\n==============================")
    print("🟡 RAW WHISPER TEXT")
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