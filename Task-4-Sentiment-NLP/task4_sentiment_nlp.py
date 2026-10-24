"""
Task 4 - NLP Sentiment Classification on Customer Reviews (TF-IDF)
Datheon ML & Data Science Internship

Pipeline:
1. Load sentiment_reviews.csv URL, else built-in sample data (offline safe)
2. Clean text: lowercase, HTML remove, punctuation remove, stopword remove
3. TF-IDF (max_features=5000)
4. Train MultinomialNB + LogisticRegression
5. predict_sentiment() for live custom input
"""

import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# small stopword list (no NLTK download needed)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "it", "this", "that", "to", "of", "for", "with", "on", "in", "i",
    "you", "we", "they", "my", "our", "very", "so", "too", "as", "at"
}

# fallback data - balanced set with repeated keywords so TF-IDF generalises
# (Datheon dataset URL currently 404, so this built-in set is used)
FALLBACK_DATA = [
    ("I love this product amazing quality fast delivery", 1),
    ("Excellent movie fantastic acting great story", 1),
    ("Great value money highly recommend love it", 1),
    ("Superb experience will buy again love quality", 1),
    ("Awesome customer service quick support great", 1),
    ("Very happy purchase five stars excellent", 1),
    ("Best product used works perfectly love", 1),
    ("Wonderful beautiful easy use great", 1),
    ("Good quality satisfied happy customer love", 1),
    ("Fantastic exceeded expectations excellent love", 1),
    ("Loved brilliant useful great product", 1),
    ("Perfect exactly wanted happy excellent", 1),
    ("Love amazing wonderful fantastic superb", 1),
    ("Great excellent awesome best happy recommend", 1),
    ("Happy satisfied love great quality product", 1),
    ("Excellent fast delivery love amazing service", 1),
    ("Wonderful movie love fantastic acting", 1),
    ("Best purchase happy great value love", 1),
    ("Superb quality love highly recommend excellent", 1),
    ("Awesome brilliant perfect love great", 1),
    ("Terrible product waste money poor quality", 0),
    ("Worst movie boring dull hate it", 0),
    ("Poor quality broke one day terrible", 0),
    ("Awful experience never buying again hate", 0),
    ("Bad customer service rude terrible", 0),
    ("Disappointed not worth price poor", 0),
    ("Horrible defective useless worst", 0),
    ("Hate ugly hard use terrible", 0),
    ("Low quality cheap material poor", 0),
    ("Failed did not work terrible worst", 0),
    ("Regret buying total scam awful", 0),
    ("Boring slow annoying worst hate", 0),
    ("Terrible awful horrible worst hate poor", 0),
    ("Worst poor bad terrible disappointed regret", 0),
    ("Hate terrible poor quality awful product", 0),
    ("Bad horrible disappointing poor service", 0),
    ("Useless broken defective terrible worst", 0),
    ("Poor dull boring worst hate movie", 0),
    ("Awful rude bad terrible service", 0),
    ("Waste scam regret terrible poor", 0),
]


def clean_text(text):
    """lowercase, remove HTML, punctuation, stopwords."""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)          # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)       # keep letters only
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def load_reviews():
    url = "https://raw.githubusercontent.com/datasets/sample-reviews/main/sentiment_reviews.csv"
    # try a couple of known column layouts
    try:
        df = pd.read_csv(url)
        print(f"Loaded CSV from URL, columns: {list(df.columns)}")
        # auto-detect text + label columns
        text_col = None
        label_col = None
        for c in df.columns:
            cl = c.lower()
            if cl in ("review", "text", "comment", "sentence"):
                text_col = c
            if cl in ("sentiment", "label", "polarity", "target"):
                label_col = c
        if text_col and label_col:
            df = df[[text_col, label_col]].dropna()
            df.columns = ["text", "label"]
            # normalize labels to 0/1
            if df["label"].dtype == object:
                df["label"] = df["label"].astype(str).str.lower().map(
                    {"positive": 1, "pos": 1, "1": 1, "negative": 0, "neg": 0, "0": 0}
                ).fillna(0).astype(int)
            print(f"Using URL dataset: {df.shape}")
            return df
        else:
            raise ValueError("column mismatch, using fallback")
    except Exception as e:
        print(f"URL load failed ({e}), using built-in sample data.")
    df = pd.DataFrame(FALLBACK_DATA, columns=["text", "label"])
    return df


def plot_confusion(cm, savepath):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - Sentiment (NB)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print(f"Saved: {savepath}")


# global vectorizer + model for live prediction
vectorizer = TfidfVectorizer(max_features=5000)
nb_model = MultinomialNB()


def predict_sentiment(text):
    """Live test helper - pass any custom phrase."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = nb_model.predict(vec)[0]
    proba = nb_model.predict_proba(vec)[0]
    label = "Positive " if pred == 1 else "Negative "
    return label, float(max(proba))


def main():
    df = load_reviews()
    df["cleaned"] = df["text"].apply(clean_text)
    print(df.head(3).to_string())
    print(f"\nTotal samples: {len(df)}, Pos: {(df['label']==1).sum()}, Neg: {(df['label']==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], df["label"], test_size=0.25, random_state=42,
        stratify=df["label"] if len(df) >= 8 else None
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF shape: {X_train_tfidf.shape}")

    # Model 1: Naive Bayes
    nb_model.fit(X_train_tfidf, y_train)
    pred_nb = nb_model.predict(X_test_tfidf)
    print("\n--- MultinomialNB ---")
    print(f"Accuracy: {accuracy_score(y_test, pred_nb):.4f}")
    print(classification_report(y_test, pred_nb, target_names=["Negative", "Positive"], zero_division=0))

    # Model 2: Logistic Regression (benchmark)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_tfidf, y_train)
    pred_lr = lr.predict(X_test_tfidf)
    print("--- LogisticRegression (benchmark) ---")
    print(f"Accuracy: {accuracy_score(y_test, pred_lr):.4f}")
    print(classification_report(y_test, pred_lr, target_names=["Negative", "Positive"], zero_division=0))

    plot_confusion(confusion_matrix(y_test, pred_nb),
                   os.path.join(OUTPUT_DIR, "confusion_matrix_sentiment.png"))

    # live demo
    print("\n--- Live predict_sentiment() demo ---")
    for s in ["I absolutely love this, amazing!", "Worst product, totally useless",
              "Datheon internship is great"]:
        label, conf = predict_sentiment(s)
        print(f"'{s}' -> {label} (conf {conf:.2f})")

    # interactive loop
    print("\nType your own review to test (type 'quit' to exit):")
    while True:
        try:
            t = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if t.lower() in ("quit", "exit", "bye"):
            print("Bye! Task 4 done ")
            break
        if not t:
            continue
        label, conf = predict_sentiment(t)
        print(f"Bot: {label} (confidence {conf:.2f})")


if __name__ == "__main__":
    main()
