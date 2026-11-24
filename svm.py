
from pathlib import Path

import pandas as pd
import sklearn
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    accuracy_score,
)


# ==================================================
# 1. File paths
# ==================================================

DATA_DIR = Path("data")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

OFFICIAL_TRAIN_PATH = DATA_DIR / "official_train.csv"
VALIDATION_PATH = DATA_DIR / "official_validation.csv"
OPENPOOL_PATH = DATA_DIR / "part_1.csv"


# ==================================================
# 2. Check files
# ==================================================

for path in [
    OFFICIAL_TRAIN_PATH,
    VALIDATION_PATH,
    OPENPOOL_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find file: {path.resolve()}"
        )

print("scikit-learn version:", sklearn.__version__)


# ==================================================
# 3. Load datasets
# ==================================================

print("\nLoading official training data...")
official_train = pd.read_csv(OFFICIAL_TRAIN_PATH)

print("Loading official validation data...")
validation_df = pd.read_csv(VALIDATION_PATH)

print("Loading Open Pool Part 1...")
openpool_df = pd.read_csv(OPENPOOL_PATH)


# ==================================================
# 4. Check required columns
# ==================================================

required_columns = {"text", "model"}

for name, df in [
    ("Official training", official_train),
    ("Official validation", validation_df),
    ("Open Pool", openpool_df),
]:
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} dataset missing columns: {missing}"
        )


# ==================================================
# 5. Clean basic data
# ==================================================

def clean_dataset(df):
    df = df.dropna(subset=["text", "model"]).copy()

    df["text"] = df["text"].astype(str)
    df["model"] = df["model"].astype(str)

    df = df[df["text"].str.strip() != ""]

    return df.reset_index(drop=True)


official_train = clean_dataset(official_train)
validation_df = clean_dataset(validation_df)
openpool_df = clean_dataset(openpool_df)


# ==================================================
# 6. Check validation set
# ==================================================

validation_models = set(validation_df["model"].unique())
official_models = set(official_train["model"].unique())
openpool_models = set(openpool_df["model"].unique())

if len(validation_models) != 18:
    raise ValueError(
        f"Expected 18 validation models, got {len(validation_models)}"
    )

validation_counts = validation_df["model"].value_counts()

if len(validation_df) != 540:
    raise ValueError(
        f"Expected 540 validation rows, got {len(validation_df)}"
    )

if not (validation_counts == 30).all():
    raise ValueError(
        "Validation set must contain exactly 30 rows per model."
    )

if validation_models != official_models:
    raise ValueError(
        "Official training and validation model classes do not match."
    )

unknown_models = openpool_models - validation_models

if unknown_models:
    raise ValueError(
        "Open Pool contains model classes not found in validation:\n"
        f"{sorted(unknown_models)}"
    )


# ==================================================
# 7. Remove exact text overlap with validation set
# ==================================================

# Normalize whitespace only for duplicate checking.
# Original text is preserved for model training.

def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


validation_texts = set(
    normalize_text(validation_df["text"])
)

official_train["_text_key"] = normalize_text(
    official_train["text"]
)

openpool_df["_text_key"] = normalize_text(
    openpool_df["text"]
)

official_before = len(official_train)
openpool_before = len(openpool_df)

official_train = official_train[
    ~official_train["_text_key"].isin(validation_texts)
].copy()

openpool_df = openpool_df[
    ~openpool_df["_text_key"].isin(validation_texts)
].copy()

official_removed = official_before - len(official_train)
openpool_removed = openpool_before - len(openpool_df)

# Remove helper column
official_train = official_train.drop(
    columns=["_text_key"]
).reset_index(drop=True)

openpool_df = openpool_df.drop(
    columns=["_text_key"]
).reset_index(drop=True)


# ==================================================
# 8. Prepare Experiment A and Experiment B
# ==================================================

# Experiment A: Official training data only
train_A = official_train.copy()

# Experiment B: Official + Open Pool Part 1
train_B = pd.concat(
    [official_train, openpool_df],
    ignore_index=True
)

print("\n========== Dataset Summary ==========")

print("Official training rows:", len(official_train))
print("Open Pool rows:", len(openpool_df))
print("Validation rows:", len(validation_df))

print("\nExact validation-text overlaps removed:")
print("Official:", official_removed)
print("Open Pool:", openpool_removed)

print("\nExperiment A training rows:", len(train_A))
print("Experiment B training rows:", len(train_B))

print("Number of models:", len(validation_models))


# ==================================================
# 9. Build TF-IDF + Linear SVM
# ==================================================

def build_model():

    return Pipeline([
        (
            "features",
            FeatureUnion([
                (
                    "word_tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=25000,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char_tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        max_features=35000,
                        sublinear_tf=True,
                    ),
                ),
            ]),
        ),
        (
            "clf",
            LinearSVC(
                C=1.0,
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])


# ==================================================
# 10. Evaluation function
# ==================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    results = {
        "balanced_accuracy": balanced_accuracy_score(
            y, predictions
        ),
        "macro_f1": f1_score(
            y, predictions, average="macro"
        ),
        "accuracy": accuracy_score(
            y, predictions
        ),
    }

    return results, predictions


# ==================================================
# 11. Experiment A: Official only
# ==================================================

print("\n========================================")
print("Experiment A: Official data only")
print("========================================")

model_A = build_model()

print("Training Experiment A...")

model_A.fit(
    train_A["text"],
    train_A["model"],
)

results_A, predictions_A = evaluate_model(
    model_A,
    validation_df["text"],
    validation_df["model"],
)

print("\nExperiment A Results:")
print(f"Balanced Accuracy: {results_A['balanced_accuracy']:.4f}")
print(f"Macro F1:          {results_A['macro_f1']:.4f}")
print(f"Accuracy:          {results_A['accuracy']:.4f}")

joblib.dump(
    model_A,
    RESULT_DIR / "linear_svm_official_only.pkl"
)


# ==================================================
# 12. Experiment B: Official + Open Pool
# ==================================================

print("\n========================================")
print("Experiment B: Official + Open Pool")
print("========================================")

model_B = build_model()

print("Training Experiment B...")

model_B.fit(
    train_B["text"],
    train_B["model"],
)

results_B, predictions_B = evaluate_model(
    model_B,
    validation_df["text"],
    validation_df["model"],
)

print("\nExperiment B Results:")
print(f"Balanced Accuracy: {results_B['balanced_accuracy']:.4f}")
print(f"Macro F1:          {results_B['macro_f1']:.4f}")
print(f"Accuracy:          {results_B['accuracy']:.4f}")

joblib.dump(
    model_B,
    RESULT_DIR / "linear_svm_official_plus_openpool.pkl"
)


# ==================================================
# 13. Compare experiments
# ==================================================

comparison = pd.DataFrame([
    {
        "experiment": "A: Official only",
        "training_rows": len(train_A),
        **results_A,
    },
    {
        "experiment": "B: Official + Open Pool",
        "training_rows": len(train_B),
        **results_B,
    },
])

comparison.to_csv(
    RESULT_DIR / "linear_svm_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n========================================")
print("Final Comparison")
print("========================================")

print(comparison.to_string(index=False))


# ==================================================
# 14. Save validation predictions
# ==================================================

prediction_df = pd.DataFrame({
    "text": validation_df["text"],
    "true_model": validation_df["model"],
    "prediction_A": predictions_A,
    "prediction_B": predictions_B,
})

prediction_df.to_csv(
    RESULT_DIR / "validation_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


# ==================================================
# 15. Save experiment information
# ==================================================

print("\n========================================")
print("Files saved")
print("========================================")

print("Model A: linear_svm_official_only.pkl")
print("Model B: linear_svm_official_plus_openpool.pkl")
print("Comparison: linear_svm_comparison.csv")
print("Predictions: validation_predictions.csv")

print("\nResults folder:", RESULT_DIR.resolve())
print("\nAll experiments completed!")
