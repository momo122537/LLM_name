
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    classification_report
)

# =========================================
# 1. 設定
# =========================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

print("========== Fair Comparison ==========")

# =========================================
# 2. 讀取官方資料
# =========================================

print("\n讀取官方資料...")

official = pd.read_csv("data/train.csv")

official = official.dropna(
    subset=["text", "model", "prompt_id"]
).copy()

official["text"] = official["text"].astype(str).str.strip()

official = official[official["text"] != ""].copy()

classes = sorted(official["model"].unique())

print("官方資料筆數：", len(official))
print("模型種類：", len(classes))

if len(classes) != 18:
    raise ValueError("官方資料不是完整 18 個模型！")

# =========================================
# 3. 固定官方訓練與驗證資料
# =========================================

print("\n建立固定官方驗證集...")

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

# 確認 prompt 沒有跨集合
assert set(official_train["prompt_id"]).isdisjoint(
    set(official_val["prompt_id"])
)

# 確認 18 類都存在
if set(official_train["model"].unique()) != set(classes):
    raise ValueError("官方訓練集缺少模型類別！")

if set(official_val["model"].unique()) != set(classes):
    raise ValueError("官方驗證集缺少模型類別！")

print("官方訓練筆數：", len(official_train))
print("固定驗證筆數：", len(official_val))

# =========================================
# 4. 讀取 Open Pool Part 1
# =========================================

print("\n讀取 Open Pool Part 1...")

pool = pd.read_csv("open_pool_parts/part_1.csv")

pool = pool[
    pool["flag_kb_proxy_after_ruling"] == False
].copy()

pool = pool.dropna(
    subset=["text", "model", "prompt_canon"]
)

pool["text"] = pool["text"].astype(str).str.strip()

pool = pool[pool["text"] != ""].copy()

# 只保留官方也有的 18 種模型
pool = pool[pool["model"].isin(classes)].copy()

print("Open Pool 筆數：", len(pool))
print("Open Pool 模型種類：", pool["model"].nunique())

if set(pool["model"].unique()) != set(classes):
    raise ValueError("Open Pool 缺少部分模型類別！")

# =========================================
# 5. 排除完全相同的驗證回答
# =========================================

print("\n檢查完全相同的回答...")

# 這裡只檢查完全相同的回答文字，
# 不代表已經排除所有相同 prompt 或語意相近的回答。

val_texts = set(official_val["text"])

official_train = official_train[
    ~official_train["text"].isin(val_texts)
].copy()

before = len(pool)

pool = pool[
    ~pool["text"].isin(val_texts)
].copy()

removed = before - len(pool)

print("排除與官方驗證回答完全相同的 Open Pool 筆數：", removed)

# =========================================
# 6. 建立共同特徵與分類器
# =========================================

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

def train_and_evaluate(name, train_data):

    print(f"\n========== {name} ==========")

    X_train = train_data["text"].astype(str)
    y_train = train_data["model"]

    X_val = official_val["text"].astype(str)
    y_val = official_val["model"]

    if set(y_train.unique()) != set(classes):
        raise ValueError(f"{name} 訓練資料缺少模型類別")

    print("訓練筆數：", len(train_data))

    # 每組模型都只用自己的訓練資料 fit 特徵
    features = build_features()

    X_train_vec = features.fit_transform(X_train)
    X_val_vec = features.transform(X_val)

    model = LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train_vec, y_train)

    pred = model.predict(X_val_vec)

    ba = balanced_accuracy_score(y_val, pred)
    acc = accuracy_score(y_val, pred)
    macro_f1 = f1_score(
        y_val,
        pred,
        average="macro"
    )

    print("Balanced Accuracy:", round(ba, 4))
    print("Accuracy:", round(acc, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\n各模型分類報告：")

    print(classification_report(
        y_val,
        pred,
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

# =========================================
# 7. 執行三組公平比較
# =========================================

results = []

# A：只有官方訓練資料
results.append(
    train_and_evaluate(
        "A_Official_Only",
        official_train
    )
)

# B：官方訓練資料 + Open Pool
combined_train = pd.concat(
    [official_train, pool],
    ignore_index=True
)

results.append(
    train_and_evaluate(
        "B_Official_Plus_OpenPool",
        combined_train
    )
)

# C：只有 Open Pool
results.append(
    train_and_evaluate(
        "C_OpenPool_Only",
        pool
    )
)

# =========================================
# 8. 比較結果
# =========================================

result_df = pd.DataFrame(results)

print("\n========== FAIR COMPARISON SUMMARY ==========")

print(
    result_df.to_string(
        index=False,
        formatters={
            "balanced_accuracy": "{:.4f}".format,
            "accuracy": "{:.4f}".format,
            "macro_f1": "{:.4f}".format
        }
    )
)

result_df.to_csv(
    "fair_comparison_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n結果已儲存：fair_comparison_results.csv")
print("\n公平比較實驗完成！")
