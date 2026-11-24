from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


# ==================================================
# 1. File paths
# ==================================================

DATA_DIR = Path("data")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_TRAIN_PATH = DATA_DIR / "official_train.csv"
VALIDATION_PATH = DATA_DIR / "official_validation.csv"
PART1_PATH = DATA_DIR / "part_1.csv"
PART2_PATH = DATA_DIR / "part_2.csv"


# ==================================================
# 2. Check files
# ==================================================

required_files = [
    OFFICIAL_TRAIN_PATH,
    VALIDATION_PATH,
    PART1_PATH,
    PART2_PATH,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find file: {path.resolve()}\n"
            "Please check that all four CSV files are inside the data folder."
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
part1_df = pd.read_csv(PART1_PATH)

print("Loading Open Pool Part 2...")
part2_df = pd.read_csv(PART2_PATH)


# ==================================================
# 4. Check required columns
# ==================================================

required_columns = {"text", "model"}

datasets = [
    ("Official training", official_train),
    ("Official validation", validation_df),
    ("Part 1", part1_df),
    ("Part 2", part2_df),
]

for name, df in datasets:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} dataset is missing required columns: {sorted(missing)}"
        )


# ==================================================
# 5. Clean basic data
# ==================================================

def clean_dataset(df):
    df = df.dropna(subset=["text", "model"]).copy()
    df["text"] = df["text"].astype(str)
    df["model"] = df["model"].astype(str).str.strip()
    df = df[df["text"].str.strip() != ""]
    df = df[df["model"] != ""]
    return df.reset_index(drop=True)


official_train = clean_dataset(official_train)
validation_df = clean_dataset(validation_df)
part1_df = clean_dataset(part1_df)
part2_df = clean_dataset(part2_df)


# ==================================================
# 6. Check validation set and model labels
# ==================================================

class_labels = sorted(validation_df["model"].unique())
validation_models = set(class_labels)

if len(class_labels) != 18:
    raise ValueError(
        f"Expected 18 validation models, got {len(class_labels)}"
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

for name, df in [
    ("Official training", official_train),
    ("Part 1", part1_df),
    ("Part 2", part2_df),
]:
    dataset_models = set(df["model"].unique())
    unknown_models = dataset_models - validation_models
    missing_models = validation_models - dataset_models

    if unknown_models:
        raise ValueError(
            f"{name} contains model labels not found in validation: "
            f"{sorted(unknown_models)}"
        )

    if missing_models:
        raise ValueError(
            f"{name} is missing these validation model labels: "
            f"{sorted(missing_models)}"
        )


# ==================================================
# 7. Remove exact text overlap with validation set
# ==================================================

# Whitespace is normalized only for duplicate checking.
# Original text is preserved for model training.

def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


validation_texts = set(normalize_text(validation_df["text"]))


def remove_validation_overlaps(df, dataset_name):
    df = df.copy()
    df["_text_key"] = normalize_text(df["text"])

    before = len(df)
    df = df[~df["_text_key"].isin(validation_texts)].copy()
    removed = before - len(df)

    df = df.drop(columns=["_text_key"]).reset_index(drop=True)

    print(f"{dataset_name} exact validation-text overlaps removed: {removed}")
    return df, removed


official_train, official_removed = remove_validation_overlaps(
    official_train, "Official"
)
part1_df, part1_removed = remove_validation_overlaps(part1_df, "Part 1")
part2_df, part2_removed = remove_validation_overlaps(part2_df, "Part 2")


# ==================================================
# 8. Prepare the three experiments
# ==================================================

# A: Official only
train_A = official_train.copy()

# B: Official + Part 2
train_B = pd.concat(
    [official_train, part2_df],
    ignore_index=True,
)

# C: Official + Part 1 + Part 2
train_C = pd.concat(
    [official_train, part1_df, part2_df],
    ignore_index=True,
)


# Check that every experiment contains all 18 classes.
for name, df in [
    ("A: Official only", train_A),
    ("B: Official + Part 2", train_B),
    ("C: Official + Part 1 + Part 2", train_C),
]:
    missing_models = validation_models - set(df["model"].unique())
    if missing_models:
        raise ValueError(
            f"{name} is missing model labels: {sorted(missing_models)}"
        )


print("\n========== Dataset Summary ==========")
print("Official training rows:", len(official_train))
print("Part 1 rows:", len(part1_df))
print("Part 2 rows:", len(part2_df))
print("Validation rows:", len(validation_df))
print("Number of models:", len(class_labels))

print("\nExperiment A training rows:", len(train_A))
print("Experiment B training rows:", len(train_B))
print("Experiment C training rows:", len(train_C))


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
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "macro_f1": f1_score(y, predictions, average="macro"),
        "accuracy": accuracy_score(y, predictions),
    }

    return results, predictions


# ==================================================
# 11. Run one experiment and save its outputs
# ==================================================

def run_experiment(experiment_name, train_df, model_filename, prediction_column):
    print("\n" + "=" * 55)
    print(experiment_name)
    print("=" * 55)

    model = build_model()

    print("Training rows:", len(train_df))
    print("Training model...")

    model.fit(train_df["text"], train_df["model"])

    results, predictions = evaluate_model(
        model,
        validation_df["text"],
        validation_df["model"],
    )

    print("\nResults:")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {results['macro_f1']:.4f}")
    print(f"Accuracy:          {results['accuracy']:.4f}")

    # Save trained model
    joblib.dump(model, RESULT_DIR / model_filename)

    # Save confusion matrix. Rows = true labels; columns = predicted labels.
    matrix = confusion_matrix(
        validation_df["model"],
        predictions,
        labels=class_labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=class_labels,
        columns=class_labels,
    )
    matrix_df.index.name = "true_model"
    matrix_df.to_csv(
        RESULT_DIR / f"confusion_matrix_{prediction_column}.csv",
        encoding="utf-8-sig",
    )

    # Save predictions for later error analysis.
    prediction_df = pd.DataFrame({
        "text": validation_df["text"],
        "true_model": validation_df["model"],
        prediction_column: predictions,
    })
    prediction_df.to_csv(
        RESULT_DIR / f"predictions_{prediction_column}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return results, predictions


# ==================================================
# 12. Run A, B, and C
# ==================================================

results_A, predictions_A = run_experiment(
    "Experiment A: Official Only",
    train_A,
    "linear_svm_official_only.pkl",
    "prediction_A",
)

results_B, predictions_B = run_experiment(
    "Experiment B: Official + Part 2",
    train_B,
    "linear_svm_official_plus_part2.pkl",
    "prediction_B",
)

results_C, predictions_C = run_experiment(
    "Experiment C: Official + Part 1 + Part 2",
    train_C,
    "linear_svm_official_plus_part1_part2.pkl",
    "prediction_C",
)


# ==================================================
# 13. Compare all experiments
# ==================================================

comparison = pd.DataFrame([
    {
        "experiment": "A: Official only",
        "training_rows": len(train_A),
        **results_A,
    },
    {
        "experiment": "B: Official + Part 2",
        "training_rows": len(train_B),
        **results_B,
    },
    {
        "experiment": "C: Official + Part 1 + Part 2",
        "training_rows": len(train_C),
        **results_C,
    },
])

comparison.to_csv(
    RESULT_DIR / "linear_svm_comparison_A_B_C.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 55)
print("FINAL COMPARISON")
print("=" * 55)
print(comparison.to_string(index=False))


# ==================================================
# 14. Save combined validation predictions
# ==================================================

all_predictions = pd.DataFrame({
    "text": validation_df["text"],
    "true_model": validation_df["model"],
    "prediction_A_official_only": predictions_A,
    "prediction_B_official_plus_part2": predictions_B,
    "prediction_C_official_plus_part1_part2": predictions_C,
})

all_predictions.to_csv(
    RESULT_DIR / "validation_predictions_A_B_C.csv",
    index=False,
    encoding="utf-8-sig",
)


# ==================================================
# 15. Finish
# ==================================================

print("\n" + "=" * 55)
print("FILES SAVED")
print("=" * 55)
print("Models:")
print("- linear_svm_official_only.pkl")
print("- linear_svm_official_plus_part2.pkl")
print("- linear_svm_official_plus_part1_part2.pkl")

print("\nConfusion matrices:")
print("- confusion_matrix_prediction_A.csv")
print("- confusion_matrix_prediction_B.csv")
print("- confusion_matrix_prediction_C.csv")

print("\nComparison:")
print("- linear_svm_comparison_A_B_C.csv")

print("\nPredictions:")
print("- validation_predictions_A_B_C.csv")

print("\nResults folder:", RESULT_DIR.resolve())
print("\nAll three experiments completed!")
