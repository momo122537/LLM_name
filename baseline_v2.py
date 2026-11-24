
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

# --------------------------------------------------
# 1. Locate the data folder
# Expected structure:
# name-that-llm/
#   baseline.py
#   data/
#     train.csv
#     test.csv (optional)
# --------------------------------------------------

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find {TRAIN_PATH.resolve()}\n"
        "Please put train.csv inside the data folder."
    )

print("scikit-learn version:", sklearn.__version__)
print("Loading training data...")

train_df = pd.read_csv(TRAIN_PATH)

required_columns = {"text", "model", "prompt_id"}
missing = required_columns - set(train_df.columns)

if missing:
    raise ValueError(f"Missing columns in train.csv: {missing}")

print("Train shape:", train_df.shape)
print("Unique models:", train_df["model"].nunique())
print("Unique prompts:", train_df["prompt_id"].nunique())

# --------------------------------------------------
# 2. Build the original baseline pipeline
# --------------------------------------------------

model_pipeline = Pipeline(
    [
        (
            "features",
            FeatureUnion(
                [
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
                ]
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=2.0,
                max_iter=1000,
            ),
        ),
    ]
)

# --------------------------------------------------
# 3. Five-fold GroupKFold cross-validation
# Keep each prompt_id within one fold.
# --------------------------------------------------

print("\nRunning 5-fold GroupKFold CV...")
print("This may take a few minutes.")

cv = GroupKFold(n_splits=5)

cv_scores = cross_val_score(
    model_pipeline,
    train_df["text"].fillna(""),
    train_df["model"],
    groups=train_df["prompt_id"],
    cv=cv,
    scoring="balanced_accuracy",
    n_jobs=-1,
)

print("\nFold scores:", cv_scores)
print(
    "5-Fold Grouped CV Balanced Accuracy: "
    f"{cv_scores.mean():.4f} +/- {cv_scores.std():.4f}"
)

# --------------------------------------------------
# 4. Fit the model on all training data
# --------------------------------------------------

print("\nFitting the final model on all training data...")

model_pipeline.fit(
    train_df["text"].fillna(""),
    train_df["model"],
)

print("Training complete.")

# --------------------------------------------------
# 5. Optional: predict test.csv and save submission
# --------------------------------------------------

if TEST_PATH.exists():
    test_df = pd.read_csv(TEST_PATH)

    if not {"id", "text"}.issubset(test_df.columns):
        print("test.csv does not contain both id and text.")
    else:
        predictions = model_pipeline.predict(
            test_df["text"].fillna("")
        )

        submission = pd.DataFrame(
            {
                "id": test_df["id"],
                "model": predictions,
            }
        )

        output_path = Path("submission.parquet")
        submission.to_parquet(output_path, index=False)

        print(f"\nSaved: {output_path.resolve()}")
        print(submission.head())
else:
    print("\ntest.csv not found; skipping submission generation.")

print("\nAll steps completed.")
