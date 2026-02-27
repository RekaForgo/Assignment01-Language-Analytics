"""
Lexical analysis: Fairy Tales vs. Contemporary Novels
Dataset: NarraDetect (Piper et al. 2025) — https://aclanthology.org/2025.wnu-1.1/
"""

import os
import string
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import ttest_ind

from utils import setup
setup()

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Tee:
    """Mirrors output to a file so print() appears in both terminal and log."""
    def __init__(self, path):
        self._file   = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout   = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


tee = Tee(os.path.join(OUTPUT_DIR, "results.txt"))


GENRES     = ["FAIRY", "NOVEL-CONT"]
STOP_WORDS = set(stopwords.words("english"))
PUNCT      = set(string.punctuation) | {
    "``", "''", "--", "...", "’", "“", "”", "—",   # add curly quotes & em dash
    "'s", "n't", "'re", "'ve", "'d", "'ll",
}


def save(name):
    """Save and close the current figure."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name))
    plt.close()


# Load & filter 
df = pd.read_csv("data/narradetect.csv")
df = df[df["genre"].isin(GENRES)].copy()

# Tokenisation
df["sentences"]      = df["text"].apply(sent_tokenize)
df["n_sentences"]    = df["sentences"].apply(len)
df["tokens_raw"]     = df["text"].apply(word_tokenize)
df["TTR_raw"]        = df["tokens_raw"].apply(lambda ts: len(set(ts)) / len(ts)) #ttr without normalisation of text. Not used.

def clean(tokens):
    return [t.lower() for t in tokens if t.lower() not in STOP_WORDS and t not in PUNCT] #lowercase, remove stopwords and punctuation.

df["tokens"]   = df["tokens_raw"].apply(clean)
df["n_tokens"] = df["tokens"].apply(len)
df["n_types"]  = df["tokens"].apply(lambda ts: len(set(ts)))
df["TTR"]      = df["n_types"] / df["n_tokens"]


# Hapax rate

def hapax_rate(tokens):
    counts = Counter(tokens)
    return sum(1 for c in counts.values() if c == 1) / len(tokens)

df["hapax_rate"]        = df["tokens"].apply(hapax_rate)
df["text_length_chars"] = df["text"].str.len()

#TTR
for col, label, fname in [
    ("TTR_raw", "TTR (raw)",                         "ttr_raw.png"),
    ("TTR",     "TTR (lowercased, no stopwords/punct)", "ttr_cleaned.png"),
]:
    stats = df.groupby("genre")[col].agg(mean="mean", sd="std")
    print(f"\n=== {label} ===")
    print(stats)
    stats.to_csv(os.path.join(OUTPUT_DIR, fname.replace(".png", ".csv")))

    sns.boxplot(data=df, x="genre", y=col)
    plt.title(f"{label} by Genre")
    save(fname)


# Corpus summary 
summary = df.groupby("genre").agg(
    n_texts            = ("text", "count"),
    avg_characters     = ("text_length_chars", "mean"),
    sd_characters      = ("text_length_chars", "std"),
    avg_tokens         = ("n_tokens", "mean"),
    sd_tokens          = ("n_tokens", "std"),
    avg_types          = ("n_types", "mean"),
    sd_types           = ("n_types", "std"),
    total_tokens       = ("n_tokens", "sum"),
)

corpus_types = (
    df.groupby("genre")["tokens"]
      .apply(lambda texts: len(set(t for ts in texts for t in ts)))
)

summary["corpus_types"] = corpus_types
summary["corpus_TTR"]   = summary["corpus_types"] / summary["total_tokens"]

print("\n=== Corpus Summary ===")
print(summary)
summary.to_csv(os.path.join(OUTPUT_DIR, "corpus_summary.csv"))


#Sentence length sanity check (according to the paper it is 5 sentences each)
sent_stats = df.groupby("genre")["n_sentences"].agg(mean="mean", sd="std")
print("\n=== Sentences per Text ===")
print(sent_stats)
sent_stats.to_csv(os.path.join(OUTPUT_DIR, "sentence_length.csv"))

sns.boxplot(data=df, x="genre", y="n_sentences")
plt.title("Sentences per Text by Genre")
plt.ylim(2, 8)
save("sentence_length.png")

# Hapax rate
hapax_stats = df.groupby("genre")["hapax_rate"].agg(mean="mean", sd="std", n="count")
print("\n=== Hapax Rate ===")
print(hapax_stats)
hapax_stats.to_csv(os.path.join(OUTPUT_DIR, "hapax_rate.csv"))

means = df.groupby("genre")["hapax_rate"].mean()
sds   = df.groupby("genre")["hapax_rate"].std()
ses   = sds / np.sqrt(df.groupby("genre")["hapax_rate"].count())
x     = np.arange(len(means))

plt.figure(figsize=(8, 5))
plt.bar(x, means, alpha=0.6)
plt.errorbar(x, means, yerr=sds, capsize=8, fmt='none', label="SD")
plt.errorbar(x, means, yerr=ses, capsize=4, fmt='none', color="red", label="SE")
plt.xticks(x, means.index)
plt.ylabel("Hapax Rate")
plt.title("Mean Hapax Rate per Genre")
plt.legend()
save("hapax_rate.png")


# Statistical test: Hapax rate

fairy = df.loc[df["genre"] == "FAIRY",      "hapax_rate"]
novel = df.loc[df["genre"] == "NOVEL-CONT", "hapax_rate"]

t_stat, p_value = ttest_ind(fairy, novel, equal_var=False)

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled_sd

d = cohens_d(fairy, novel)
print("\n=== t-test: Hapax Rate (FAIRY vs NOVEL-CONT) ===")
print(f"  t = {t_stat:.4f},  p = {p_value:.4e},  Cohen's d = {d:.4f}")


# Most common tokens
def top_tokens(genre, n=20):
    tokens = (t for ts in df.loc[df["genre"] == genre, "tokens"] for t in ts)
    return pd.DataFrame(Counter(tokens).most_common(n), columns=["token", "freq"])

top = pd.concat(
    [top_tokens("FAIRY").rename(columns={"token": "fairy_token", "freq": "fairy_freq"}),
     top_tokens("NOVEL-CONT").rename(columns={"token": "novel_token", "freq": "novel_freq"})],
    axis=1,
)
print("\n=== Top 20 Tokens per Genre ===")
print(top.to_string(index=False))
top.to_csv(os.path.join(OUTPUT_DIR, "top_tokens.csv"), index=False)


#SpaCy: POS tagging & NER
# Install model with: python -m spacy download en_core_web_sm

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")

    pos_rows, ner_rows = [], []
    for _, row in df.iterrows():
        doc    = nlp(row["text"])
        alpha  = [t for t in doc if t.is_alpha]
        total  = len(alpha) or 1
        ents   = doc.ents

        pos_counts = Counter(t.pos_ for t in alpha)
        pos_rows.append({
            "genre": row["genre"],
            **{tag: pos_counts.get(tag, 0) / total for tag in ("NOUN", "VERB", "ADJ", "PRON")},
        })
        ner_rows.append({
            "genre":        row["genre"],
            "ner_count":    len(ents),
            "ner_per_1000": len(ents) / total * 1000,
            **Counter(e.label_ for e in ents),
        })

    # POS
    df_pos   = pd.DataFrame(pos_rows)
    pos_mean = df_pos.groupby("genre").mean()
    print("\n=== POS Proportions ===")
    print(pos_mean)
    pos_mean.to_csv(os.path.join(OUTPUT_DIR, "pos_proportions.csv"))

    df_long = df_pos.melt(id_vars="genre", var_name="POS", value_name="proportion")
    sns.boxplot(data=df_long, x="POS", y="proportion", hue="genre", showfliers=False)
    plt.title("POS Proportions per Text by Genre")
    plt.ylabel("Proportion")
    save("pos_proportions.png")

    # NER
    df_ner      = pd.DataFrame(ner_rows).fillna(0)
    ner_density = df_ner.groupby("genre")["ner_per_1000"].agg(mean="mean", sd="std")
    print("\n=== NER Density (per 1000 tokens) ===")
    print(ner_density)
    ner_density.to_csv(os.path.join(OUTPUT_DIR, "ner_density.csv"))

    sns.boxplot(data=df_ner, x="genre", y="ner_per_1000")
    plt.title("Named Entity Density per Text (per 1000 tokens)")
    plt.ylabel("Entities per 1000 Tokens")
    save("ner_density.png")

    entity_cols = [c for c in df_ner.columns if c not in ("genre", "ner_count", "ner_per_1000")]
    mean_ents   = df_ner.groupby("genre")[entity_cols].mean()
    mean_ents.to_csv(os.path.join(OUTPUT_DIR, "ner_types.csv"))
    mean_ents.T.sort_values("FAIRY", ascending=False).head(6).plot(kind="bar", figsize=(8, 5))
    plt.ylabel("Average Count per Text")
    plt.title("Top Named Entity Types per Genre")
    save("ner_types.png")

    #Function words
    FUNCTION_TAGS = {"PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}

    fw_rows = []

    for _, row in df.iterrows():
        doc = nlp(row["text"])
        alpha = [t for t in doc if t.is_alpha]
        total = len(alpha) or 1
    
        fw_count = sum(1 for t in alpha if t.pos_ in FUNCTION_TAGS)
    
        fw_rows.append({
            "genre": row["genre"],
            "fw_ratio": fw_count / total
        })
    
    df_fw = pd.DataFrame(fw_rows)
    fw_stats = df_fw.groupby("genre")["fw_ratio"].agg(mean="mean", sd="std")
    df_fw["fw_percent"] = df_fw["fw_ratio"] * 100
    print("\n=== Function Word Proportion ===")
    print(fw_stats)

except OSError:
    print("\n[SpaCy] Model 'en_core_web_sm' not found — skipping POS/NER.")
    print("  Fix: python -m spacy download en_core_web_sm")


#Sentiment (using VADER by NLTK)

sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["text"].apply(lambda s: sia.polarity_scores(str(s))["compound"])

sentiment_stats = df.groupby("genre")["sentiment"].agg(mean="mean", sd="std")
print("\n=== Sentiment (VADER compound) ===")
print(sentiment_stats)
sentiment_stats.to_csv(os.path.join(OUTPUT_DIR, "sentiment.csv"))

df_nonzero = df[df["sentiment"] != 0]
fig, axes  = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (genre, color) in zip(axes, [("FAIRY", "#1f77b4"), ("NOVEL-CONT", "#ff7f0e")]):
    sns.histplot(
        df_nonzero.loc[df_nonzero["genre"] == genre, "sentiment"],
        kde=True, stat="density", ax=ax, color=color,
    )
    ax.set_title(genre)
    ax.set_xlabel("Compound score")
axes[0].set_ylabel("Density")
plt.suptitle("Sentiment Distribution by Genre")
save("sentiment_distribution.png")


print(f"\nDone. All outputs saved to '{OUTPUT_DIR}/'")
tee.close()
