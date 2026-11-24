
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    classification_report
)

# ==================================================
# 1. Settings
# ==================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

OFFICIAL_PATH = "data/train.csv"
PART1_PATH = "open_pool_parts/part_1.csv"
PART2_PATH = "open_pool_parts/part_2.csv"
PART3_PATH = "open_pool_parts/part_3.csv"

print("========== Part 1 + Part 2 + Part 3 Comparison ==========")

# ==================================================
# 2. Load official data
# ==================================================

official = pd.read_csv(OFFICIAL_PATH)

official = official.dropna(
    subset=["text", "model", "prompt_id"]
).copy()

official["text"] = official["text"].astype(str).str.strip()
official = official[official["text"] != ""].copy()

classes = sorted(official["model"].unique())

if len(classes) != 18:
    raise ValueError("官方資料不是完整 18 個模型！")

print("Official rows:", len(official))

# ==================================================
# 3. Fixed official train / validation split
# ==================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    splitter.split(
        official["text"],
        official["model"],
        groups=official["prompt_id"]
    )
)

official_train = official.iloc[train_idx].copy()
official_val = official.iloc[val_idx].copy()

assert set(official_train["prompt_id"]).isdisjoint(
    set(official_val["prompt_id"])
)

if set(official_train["model"].unique()) != set(classes):
    raise ValueError("官方訓練集缺少模型類別")

if set(official_val["model"].unique()) != set(classes):
    raise ValueError("官方驗證集缺少模型類別")

print("Official train:", len(official_train))
print("Official validation:", len(official_val))

# ==================================================
# 4. Load and filter Open Pool
# ==================================================

def load_pool(path, name):

    print(f"\nLoading {name}...")

    df = pd.read_csv(path)

    required = {
        "text",
        "model",
        "prompt_canon",
        "flag_kb_proxy_after_ruling"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} 缺少欄位：{missing}"
        )

    raw_count = len(df)

    df = df[
        df["flag_kb_proxy_after_ruling"] == False
    ].copy()

    df = df.dropna(
        subset=["text", "model", "prompt_canon"]
    ).copy()

    df["text"] = df["text"].astype(str).str.strip()

    df = df[df["text"] != ""].copy()

    df = df[df["model"].isin(classes)].copy()

    print(f"{name} 原始筆數：{raw_count}")
    print(f"{name} 篩選後筆數：{len(df)}")
    print(f"{name} 模型種類：{df['model'].nunique()}")

    missing_classes = sorted(
        set(classes) - set(df["model"].unique())
    )

    if missing_classes:
        print(f"警告：{name} 缺少模型：{missing_classes}")

    return df


part1 = load_pool(PART1_PATH, "Part 1")
part2 = load_pool(PART2_PATH, "Part 2")
part3 = load_pool(PART3_PATH, "Part 3")

# ==================================================
# 5. Remove exact validation answer duplicates
# ==================================================

validation_texts = set(official_val["text"])

official_train = official_train[
    ~official_train["text"].isin(validation_texts)
].copy()


def remove_overlap(df, name):

    before = len(df)

    df = df[
        ~df["text"].isin(validation_texts)
    ].copy()

    print(
        f"{name} 排除重複回答：{before - len(df)} 筆"
    )

    return df


part1 = remove_overlap(part1, "Part 1")
part2 = remove_overlap(part2, "Part 2")
part3 = remove_overlap(part3, "Part 3")

# ==================================================
# 6. Combine Open Pool datasets
# ==================================================

pool_part1_part2 = pd.concat(
    [part1, part2],
    ignore_index=True
)

pool_part1_part2_part3 = pd.concat(
    [part1, part2, part3],
    ignore_index=True
)

print("\n========== DATASET SUMMARY ==========")

print("Part 1 usable:", len(part1))
print("Part 2 usable:", len(part2))
print("Part 3 usable:", len(part3))

print("Part 1 + Part 2:", len(pool_part1_part2))

print(
    "Part 1 + Part 2 + Part 3:",
    len(pool_part1_part2_part3)
)

# ==================================================
# 7. Build TF-IDF features
# ==================================================

def build_features():

    return FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                max_features=25000,
                sublinear_tf=True
            )
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                max_features=35000,
                sublinear_tf=True
            )
        )
    ])

# ==================================================
# 8. Train and evaluate
# ==================================================

def train_and_evaluate(name, train_data):

    print(f"\n========== {name} ==========")

    train_data = train_data.dropna(
        subset=["text", "model"]
    ).copy()

    X_train = train_data["text"].astype(str)
    y_train = train_data["model"]

    X_val = official_val["text"].astype(str)
    y_val = official_val["model"]

    missing_classes = sorted(
        set(classes) - set(y_train.unique())
    )

    if missing_classes:
        raise ValueError(
            f"{name} 訓練資料缺少模型：{missing_classes}"
        )

    print("Training rows:", len(train_data))

    features = build_features()

    X_train_vec = features.fit_transform(X_train)
    X_val_vec = features.transform(X_val)

    model = LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_val_vec)

    # Confusion Matrix
    cm = confusion_matrix(
        y_val,
        predictions,
        labels=classes
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.to_csv(
        f"confusion_matrix_{name}.csv",
        encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(16, 13))

    image = ax.imshow(cm, interpolation="nearest")

    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted Model")
    ax.set_ylabel("Actual Model")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))

    ax.set_xticklabels(
        classes,
        rotation=90,
        fontsize=8
    )

    ax.set_yticklabels(
        classes,
        fontsize=8
    )

    fig.colorbar(image, ax=ax)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7
            )

    fig.tight_layout()

    fig.savefig(
        f"confusion_matrix_{name}.png",
        dpi=200
    )

    plt.close(fig)

    print(f"混淆矩陣已儲存：confusion_matrix_{name}.png")

    # Metrics
    ba = balanced_accuracy_score(y_val, predictions)
    acc = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=classes,
        average="macro",
        zero_division=0
    )

    print("Balanced Accuracy:", round(ba, 4))
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:")

    print(classification_report(
        y_val,
        predictions,
        labels=classes,
        zero_division=0
    ))

    return {
        "experiment": name,
        "train_samples": len(train_data),
        "balanced_accuracy": ba,
        "accuracy": acc,
        "macro_f1": macro_f1
    }

# ==================================================
# 9. Four experiments
# ==================================================

print("\n========== Running Experiments ==========")

official_plus_part3 = pd.concat(
    [official_train, part3],
    ignore_index=True
)

official_plus_part1_part2 = pd.concat(
    [official_train, pool_part1_part2],
    ignore_index=True
)

official_plus_part1_part2_part3 = pd.concat(
    [official_train, pool_part1_part2_part3],
    ignore_index=True
)

experiments = [
    ("A_Official_Only", official_train),

    ("B_Official_Plus_Part3", official_plus_part3),

    ("C_Official_Plus_Part1_Part2",
     official_plus_part1_part2),

    ("D_Official_Plus_Part1_Part2_Part3",
     official_plus_part1_part2_part3)
]

results = []

for name, training_data in experiments:

    results.append(
        train_and_evaluate(name, training_data)
    )

# ==================================================
# 10. Save comparison results
# ==================================================

result_df = pd.DataFrame(results)

print("\n========== FINAL COMPARISON ==========")

print(result_df.to_string(
    index=False,
    formatters={
        "balanced_accuracy": "{:.4f}".format,
        "accuracy": "{:.4f}".format,
        "macro_f1": "{:.4f}".format
    }
))

result_df.to_csv(
    "part123_comparison_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n結果已儲存：part123_comparison_results.csv")
print("Part 1 + Part 2 + Part 3 實驗完成！")
