
import pyaudio


RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
DEVICES_TO_TRY = [14, 1, 7, 0, 15, 8, 2]

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
