from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_prompt(relative_path):
    path = PROJECT_ROOT / relative_path

    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()

    if not content:
        raise ValueError(f"Prompt file is empty: {path}")

    return content