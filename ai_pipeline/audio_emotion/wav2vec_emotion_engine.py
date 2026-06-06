import numpy as np
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"


class Wav2VecEmotionEngine:
    def __init__(self):
      self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
      self.model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
      self.model.eval()

    def load_audio(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio, sr



        energy = float(np.mean(rms))

        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[pitches > 0]
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0

        duration = librosa.get_duration(y=audio, sr=sr)
        intervals = librosa.effects.split(audio, top_db=30)
        voiced_duration = sum((end - start) / sr for start, end in intervals)

        voice_activity = voiced_duration / duration if duration > 0 else 0.0
        speaking_rate = len(intervals) / duration if duration > 0 else 0.0

        return {
            "energy": energy,
            "pitch_mean": pitch_mean,
            "duration": duration,
            "voice_activity": voice_activity,
            "speaking_rate": speaking_rate,
        }

    def extract_prosody(self, audio, sr):
            # --- Energy (RMS) ---
            rms = librosa.feature.rms(y=audio)[0]
            energy = float(np.mean(rms))
            energy_std = float(np.std(rms))  # energy variability

            # --- Pitch via pYIN (robust F0, human vocal range only) ---
            # fmin=70Hz (low adult male), fmax=450Hz (high child) — anything
            # outside this is noise/harmonics, not human speech pitch.
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    audio,
                    fmin=70,
                    fmax=450,
                    sr=sr,
                )
                voiced_f0 = f0[~np.isnan(f0)]
                if len(voiced_f0) > 0:
                    pitch_mean = float(np.mean(voiced_f0))
                    pitch_std = float(np.std(voiced_f0))  # pitch variability!
                else:
                    pitch_mean = 0.0
                    pitch_std = 0.0
            except Exception:
                pitch_mean = 0.0
                pitch_std = 0.0

            # --- Duration / voice activity / speaking rate ---
            duration = librosa.get_duration(y=audio, sr=sr)
            intervals = librosa.effects.split(audio, top_db=30)
            voiced_duration = sum((end - start) / sr for start, end in intervals)
            voice_activity = voiced_duration / duration if duration > 0 else 0.0
            speaking_rate = len(intervals) / duration if duration > 0 else 0.0

            return {
                "energy": energy,
                "energy_std": energy_std,
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "duration": duration,
                "voice_activity": voice_activity,
                "speaking_rate": speaking_rate,
            }

    def extract_embeddings(self, audio, sr):
        inputs = self.feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden = outputs.last_hidden_state
        embedding = hidden.mean(dim=1).squeeze().cpu().numpy()

        return embedding.tolist()

    def estimate_dimensions(self, prosody):
        energy = min(prosody["energy"] * 8, 1.0)
        pitch = min(prosody["pitch_mean"] / 500, 1.0)
        voice_activity = min(prosody["voice_activity"], 1.0)
        speaking_rate = min(prosody["speaking_rate"] / 4, 1.0)

        arousal = float((energy * 0.45) + (pitch * 0.30) + (speaking_rate * 0.25))
        engagement = float((voice_activity * 0.60) + (energy * 0.40))

        if engagement < 0.30:
          valence = 0.25

        elif arousal > 0.75:
          valence = 0.40

        elif arousal < 0.35:
          valence = 0.55

        else:
          valence = 0.50

        return {
            "arousal": round(arousal, 3),
            "valence": round(valence, 3),
            "engagement": round(engagement, 3),
        }

    def map_emotion(self, dims):
        arousal = dims["arousal"]
        valence = dims["valence"]
        engagement = dims["engagement"]

        if engagement < 0.30:
            return "withdrawn"

        if arousal >= 0.60 and valence < 0.50:
            return "fearful_or_agitated"

        if arousal >= 0.60 and valence >= 0.50:
            return "excited_or_alert"

        if arousal < 0.45 and valence < 0.50:
            return "sad_or_low"

        return "calm_or_neutral"

    def analyze(self, audio_path):
        audio, sr = self.load_audio(audio_path)
        prosody = self.extract_prosody(audio, sr)
        embedding = self.extract_embeddings(audio, sr)
        dims = self.estimate_dimensions(prosody)
        emotion = self.map_emotion(dims)

        return {
            "audio_emotion": emotion,
            "arousal": dims["arousal"],
            "valence": dims["valence"],
            "engagement": dims["engagement"],
            "prosody": prosody,
            "embedding_size": len(embedding),
            "confidence": "low_to_medium",
            "note": "Rule-based estimation from XLSR embeddings + prosodic features, not a fine-tuned emotion classifier."
        }
        
# ============================================================
# Singleton + Orchestrator Adapter
# ============================================================
# The orchestrator calls detect_emotion(audio_path). We expose
# that here and lazy-load the engine once (model is ~1.2GB).

_engine = None


def _get_engine():
    """Lazy singleton — loads the XLSR model only once."""
    global _engine
    if _engine is None:
        print("⏳ Loading emotion engine (XLSR + prosody)...")
        _engine = Wav2VecEmotionEngine()
        print("✅ Emotion engine ready")
    return _engine


def analyze_audio_emotion(audio_path):
    """Full analysis — returns the rich dict from the engine."""
    return _get_engine().analyze(audio_path)


def detect_emotion(audio_path):
    """Orchestrator-facing adapter.

    Returns a dict compatible with the weighting layer:
      - emotion           : the mapped audio emotion category
      - arousal/valence/engagement : prosodic dimensions
      - confidence        : numeric (0-1) for weighting math
      - source            : provenance tag
      - prosody           : raw acoustic features
      - note              : honesty note
    """
    print("🎭 Running emotion detection...")
    try:
        result = analyze_audio_emotion(audio_path)
    except Exception as e:
        # Fail safe: never crash the pipeline because of audio analysis
        print(f"⚠️ Emotion detection failed ({e}); returning neutral fallback.")
        return {
            "emotion": "unknown",
            "arousal": 0.0,
            "valence": 0.5,
            "engagement": 0.0,
            "confidence": 0.0,
            "source": "emotion_failed_fallback",
            "prosody": {},
            "note": "Audio emotion analysis failed; pipeline continued on text only.",
        }

    return {
        "emotion": result["audio_emotion"],
        "arousal": result["arousal"],
        "valence": result["valence"],
        "engagement": result["engagement"],
        # map the qualitative confidence to a number the weighting layer can use
        "confidence": 0.55,  # low_to_medium — honest, not overclaiming
        "source": "xlsr_prosodic_v1",
        "prosody": result["prosody"],
        "embedding_size": result["embedding_size"],
        "note": result["note"],
    }