import re
import asyncio, concurrent.futures, os, tempfile, logging
logger = logging.getLogger(__name__)
TETA_VOICE = "ar-JO-SanaNeural"
TETA_RATE  = "+0%"
TETA_PITCH = "+0Hz"

async def _synthesize_async(text, output_path):
    import edge_tts
    communicate = edge_tts.Communicate(text=text, voice=TETA_VOICE, rate=TETA_RATE, pitch=TETA_PITCH)
    await communicate.save(output_path)
    return output_path

def _run_in_thread(text, output_path):
    asyncio.run(_synthesize_async(text, output_path))

def synthesize_teta_voice(text, output_path=None):
    if not text or not text.strip():
        return ""
    import re
    text = re.sub(r'[^\u0600-\u06FF\s\.\،\؟\!a-zA-Z0-9]', '', text)
    text = text.strip()
      
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, prefix="teta_tts_")
        output_path = tmp.name
        tmp.close()
    try:
        print(f"TTS: synthesizing {len(text)} chars", flush=True)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_in_thread, text, output_path)
                future.result(timeout=30)
        else:
            asyncio.run(_synthesize_async(text, output_path))
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("TTS produced empty file")
        print(f"TTS ready: {os.path.getsize(output_path)} bytes", flush=True)
        return output_path
    except Exception as e:
        print(f"TTS failed: {e}", flush=True)
        if output_path and os.path.exists(output_path):
            try: os.remove(output_path)
            except: pass
        return ""

def synthesize_teta_voice_to_base64(text):
    import base64
    audio_path = synthesize_teta_voice(text)
    if not audio_path:
        return ""
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        try: os.remove(audio_path)
        except: pass
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        print(f"TTS base64 ready: {len(encoded)} chars", flush=True)
        return encoded
    except Exception as e:
        print(f"TTS base64 failed: {e}", flush=True)
        return ""
