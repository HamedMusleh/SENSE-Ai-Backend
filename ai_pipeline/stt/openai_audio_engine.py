from openai import OpenAI
from dotenv import load_dotenv
import pyaudio
import wave
import threading
import tempfile
import os
import json
import re
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
    print(str(text))


# =========================
# OpenAI Setup
# =========================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENSE_VECTOR_STORE_ID = os.getenv("SENSE_VECTOR_STORE_ID")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)



def load_prompt(relative_path):
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"Prompt file is empty: {path}")

    return content

# =========================
# Audio Setup
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
        print("❌ No microphone worked. Close Discord/Zoom/Browser and try again.")
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

def nlp_preprocess_pipeline(raw_text):
    text = remove_common_noise(raw_text)
    text = normalize_arabic(text)
    text = palestinian_postprocess(text)
    return text


# =========================
# OpenAI Audio Transcription
# =========================
def transcribe_with_openai_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language="ar",
            response_format="text",
            prompt=load_prompt("prompts/transcription_prompt.txt")
        )

    raw_text = transcription.strip()
    processed_text = nlp_preprocess_pipeline(raw_text)

    return raw_text, processed_text

# =========================
# Teta AI Reply
# =========================
def ask_teta_reply(child_text, conversation_history):
    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [SENSE_VECTOR_STORE_ID],
                "max_num_results": 3
            }
        ],
        input=[
            {
                "role": "system",
                "content": load_prompt("prompts/teta_system_prompt.txt")
            },
            {
                "role": "user",
                "content": f"""
Use the SENSE resources for mental-health triage, safety rules, risk classification, and supportive response behavior.
For harmless general questions outside the resources, answer briefly using general knowledge.
If any risk signal appears, prioritize SENSE safety rules..

Conversation so far:
{json.dumps(conversation_history, ensure_ascii=False)}

Latest child message:
{child_text}
"""
            }
        ]
    )

    return response.output_text.strip()
# =========================
# Final Full Conversation Analysis
# =========================
def final_session_analysis(conversation_history):
    response = client.responses.create(
        model="gpt-4.1-mini",
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [SENSE_VECTOR_STORE_ID],
                "max_num_results": 5
            }
        ],
        input=[
            {
                "role": "system",
                "content": load_prompt("prompts/final_analysis_prompt.txt")
            },
            {
                "role": "user",
                "content": f"""
Use the SENSE annotation guide, safety rules, and triage labels from the uploaded resources.
Apply the rules strictly.

Conversation history:
{json.dumps(conversation_history, ensure_ascii=False)}
"""
            }
        ]
    )

    return response.output_text.strip()
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": load_prompt("prompts/final_analysis_prompt.txt")
            },
            {
                "role": "user",
                "content": json.dumps(conversation_history, ensure_ascii=False)
            }
        ]
    )

    return response.output_text.strip()
# =========================
# Print Final Report
# =========================
def print_final_report(final_json):
    print("\n==============================")
    print("📊 FINAL CONVERSATION REPORT")
    print("==============================")

    print("\n📝 Message-by-message report:")

    for report in final_json.get("message_reports", []):
        print("\n------------------------------")
        print(f"Message {report.get('turn')}:")

        print("\nChild Text:")
        print_ar(report.get("child_text", ""))

        print("\nSummary:")
        print_ar(report.get("message_summary", ""))

        print("\nSignals:")
        print(report.get("emotional_signals", []))

        print("\nMessage Risk:")
        print("\nMessage Label:")
        print(report.get("message_label", ""))

        print("\nRisk Signal:")
        print(report.get("risk_signal", ""))

        print("\nNeeds Human Review:")
        print(report.get("needs_human_review", ""))
        print(report.get("message_risk_level", ""))

        print("\nNote:")
        print_ar(report.get("note", ""))

    print("\n==============================")
    print("🌍 Overall Final Result")
    print("==============================")

    print("\nOverall Summary:")
    print_ar(final_json.get("overall_summary", ""))

    print("\nKey Patterns:")
    print(final_json.get("key_patterns_across_conversation", []))

    print("\nConcerning Phrases:")
    print(final_json.get("concerning_phrases", []))

    print("\nFinal Risk Level:")
    print(final_json.get("final_risk_level", ""))

    print("\nFinal Reason:")
    print_ar(final_json.get("final_reason", ""))

    print("\nRecommendation:")
    print_ar(final_json.get("recommendation", ""))

    print("\nSpecialist Notes:")
    print_ar(final_json.get("specialist_notes", ""))
    print("\nFinal Label:")
    print(final_json.get("final_label", ""))

    print("\nSafety Disclaimer:")
    print_ar(final_json.get("safety_disclaimer", ""))


# =========================
# Main Loop
# =========================
conversation_history = []
turn_number = 1

print("\n✅ SENSE Voice Prototype Started")
print("🎧 Audio: OpenAI Audio Transcription")
print("🧠 LLM: Teta AI + Final Session Report")

while True:
    audio_path = record_push_to_talk()

    if audio_path is None:
        break

    try:
        print("\n⏳ Transcribing with OpenAI Audio...")
        raw_text, child_text = transcribe_with_openai_audio(audio_path)

        print("\n==============================")
        print("🟡 RAW OPENAI AUDIO TEXT")
        print("==============================")
        print_ar(raw_text)

        print("\n==============================")
        print("🟢 AFTER NLP PREPROCESSING")
        print("==============================")
        print_ar(child_text)

        conversation_history.append({
            "turn": turn_number,
            "raw_transcription": raw_text,
            "child_text": child_text
        })

        print("\n🌸 Teta AI Reply:")
        reply = ask_teta_reply(child_text, conversation_history)
        print_ar(reply)

    except Exception as e:
        print("❌ Error:", e)

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

    turn_number += 1

    again = input("\nRecord another child message? y/n: ")
    if again.lower() != "y":
        break


# =========================
# Final Report
# =========================
if len(conversation_history) > 0:
    print("\n⏳ Generating final session report...")

    try:
        final_result = final_session_analysis(conversation_history)
        final_json = json.loads(final_result)
        print_final_report(final_json)

    except Exception as e:
        print("❌ Final analysis error:", e)
        print("\nRaw final result:")
        print(final_result if "final_result" in locals() else "")

else:
    print("No conversation recorded.")