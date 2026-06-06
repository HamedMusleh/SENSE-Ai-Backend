from ai_pipeline.audio_emotion.wav2vec_emotion_engine import Wav2VecEmotionEngine

_engine = None


def get_emotion_engine():
    global _engine

    if _engine is None:
        print("[EmotionService] Loading Wav2VecEmotionEngine once...")
        _engine = Wav2VecEmotionEngine()

    return _engine


def analyze_audio_emotion(audio_path):
    engine = get_emotion_engine()
    return engine.analyze(audio_path)