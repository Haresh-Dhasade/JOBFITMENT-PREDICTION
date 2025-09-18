# train_model.py
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_PATH = "/Users/hareshdhasade/Development/project_web/job-fitment-prediction/data/resume_labeled.csv"
MODELS_DIR = "models"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # compact and fast

os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    return df

def main():
    df = load_data(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    print(f"Loaded {len(texts)} examples. Using embedder {EMBED_MODEL_NAME} ...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = LogisticRegression(max_iter=2000, solver="saga", multi_class="multinomial")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(clf, os.path.join(MODELS_DIR, "job_clf.joblib"))
    joblib.dump(EMBED_MODEL_NAME, os.path.join(MODELS_DIR, "embedder_name.joblib"))
    print("Saved models to", MODELS_DIR)

if __name__ == "__main__":
    main()