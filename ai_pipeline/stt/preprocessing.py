# =========================
# NLP Preprocessing
# =========================
import re
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
