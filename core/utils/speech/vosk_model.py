import os
from vosk import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "vosk-model-small-en-us-0.15")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "Vosk model not found. Download and unzip "
        "'vosk-model-small-en-us-0.15' into core/utils/speech/models/"
    )

VOSK_MODEL = Model(MODEL_PATH)
