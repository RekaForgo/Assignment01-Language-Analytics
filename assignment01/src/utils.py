"""
One-time setup.
Run automatically at the top of analysis.py, or manually: python utils.py
"""

import subprocess
import sys

import nltk

NLTK_RESOURCES = ("punkt_tab", "stopwords", "vader_lexicon")
SPACY_MODEL    = "en_core_web_sm"


def ensure_nltk():
    for resource in NLTK_RESOURCES:
        nltk.download(resource, quiet=True)


def ensure_spacy_model():
    try:
        import spacy
        spacy.load(SPACY_MODEL)
    except OSError:
        print(f"[setup] spaCy model '{SPACY_MODEL}' not found — downloading...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
            check=True,
        )
        print(f"[setup] '{SPACY_MODEL}' ready.")


def setup():
    ensure_nltk()
    ensure_spacy_model()


if __name__ == "__main__":
    setup()
