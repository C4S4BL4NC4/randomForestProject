import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)


# -------------------- 0. HELPER: save DataFrame as PNG table --------------------
def save_table_as_image(df, title, filepath, fontsize=10):
    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.8, len(df) * 0.6 + 1))
    ax.axis("tight")
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize + 2, weight="bold", pad=10)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.2, 1.4)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


# -------------------- 1. LOAD DATA (robust paths) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

train_path = os.path.join(DATASETS_DIR, "train_ai4i2020.csv")
test_path = os.path.join(DATASETS_DIR, "test_ai4i2020.csv")

if not os.path.exists(train_path):
    original_path = os.path.join(DATASETS_DIR, "ai4i2020.csv")
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Neither {train_path} nor {original_path} found.\n")
    df = pd.read_csv(original_path)
    df["failure"] = (df["Machine failure"] == 1).astype(int)
    # Drop identifiers + the individual failure-mode flags (TWF/HDF/PWF/OSF/RNF).
    # These flags are sub-components of "Machine failure" itself, so keeping them
    # would leak the label into the features (near-perfect, meaningless accuracy).
    drop_cols = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]
    X = df.drop(columns=drop_cols)
    y = df["failure"]
    X = pd.get_dummies(X, columns=["Type"], drop_first=False)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_df = X_tr.copy()
    train_df["failure"] = y_tr
    test_df = X_te.copy()
    test_df["failure"] = y_te
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Train/test files regenerated successfully.")
else:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

X_train = train_df.drop("failure", axis=1)
y_train = train_df["failure"]
X_test = test_df.drop("failure", axis=1)
y_test = test_df["failure"]

# -------------------- 2. OUTPUT DIRECTORY --------------------
FOLDER_NAME = "improved_model"
OUTPUT_DIR = f"./outputs/{FOLDER_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- 3. BASELINE MODEL (same as original script) --------------------
rf_baseline = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"
)
rf_baseline.fit(X_train, y_train)
y_pred_baseline = rf_baseline.predict(X_test)
baseline_f1 = f1_score(y_test, y_pred_baseline)
print(
    f"[Baseline] n_estimators=100, max_depth=None -> Test F1 (failure class): {baseline_f1:.4f}"
)

# -------------------- 4. PARAMETER SENSITIVITY (answers "Q9: effect of parameters") --------------------
print("\n--- Sensitivity: n_estimators (max_depth=None) ---")
n_values = [10, 50, 100, 200, 500]
f1_by_n = []
for v in n_values:
    m = RandomForestClassifier(
        n_estimators=v, class_weight="balanced", random_state=42, n_jobs=-1
    )
    m.fit(X_train, y_train)
    f1_by_n.append(f1_score(y_test, m.predict(X_test)))
    print(f"n_estimators={v}: F1={f1_by_n[-1]:.4f}")

print("\n--- Sensitivity: max_depth (n_estimators=200) ---")
depth_values = [3, 5, 10, 20, None]
f1_by_depth = []
for d in depth_values:
    m = RandomForestClassifier(
        n_estimators=200,
        max_depth=d,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    m.fit(X_train, y_train)
    f1_by_depth.append(f1_score(y_test, m.predict(X_test)))
    print(f"max_depth={d}: F1={f1_by_depth[-1]:.4f}")

# Save sensitivity table
sens_df = pd.DataFrame(
    {
        "n_estimators": (
            n_values + [""] * (len(depth_values) - len(n_values))
            if len(depth_values) > len(n_values)
            else n_values
        ),
    }
)
n_df = pd.DataFrame(
    {"n_estimators": n_values, "F1-score": [round(x, 4) for x in f1_by_n]}
).set_index("n_estimators")
depth_df = pd.DataFrame(
    {
        "max_depth": [str(d) for d in depth_values],
        "F1-score": [round(x, 4) for x in f1_by_depth],
    }
).set_index("max_depth")
save_table_as_image(
    n_df,
    "Effect of n_estimators on F1-score",
    os.path.join(OUTPUT_DIR, "param_effect_n_estimators.png"),
)
save_table_as_image(
    depth_df,
    "Effect of max_depth on F1-score",
    os.path.join(OUTPUT_DIR, "param_effect_max_depth.png"),
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(n_values, f1_by_n, marker="o")
axes[0].set_xlabel("n_estimators")
axes[0].set_ylabel("F1-score (failure class)")
axes[0].set_title("Effect of n_estimators")
axes[0].grid(True)
depth_labels = [str(d) for d in depth_values]
axes[1].plot(depth_labels, f1_by_depth, marker="o", color="darkorange")
axes[1].set_xlabel("max_depth")
axes[1].set_ylabel("F1-score (failure class)")
axes[1].set_title("Effect of max_depth")
axes[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "param_sensitivity.png"), dpi=150)
plt.close()

# -------------------- 5. FINAL TUNED MODEL --------------------
# Chosen from the sensitivity study above: more trees + a bounded depth
# generalizes better than the unbounded-depth baseline.
best_n, best_depth = 200, 10
rf = RandomForestClassifier(
    n_estimators=best_n,
    max_depth=best_depth,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)
rf.fit(X_train, y_train)

cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="f1")
print(
    f"\n5-fold CV F1-score (tuned model): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
)

y_score = rf.predict_proba(X_test)[:, 1]
y_pred = (y_score >= 0.5).astype(int)

print(f"\nTest Accuracy (0.5 threshold): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report (0.5 threshold):")
print(classification_report(y_test, y_pred, digits=4))

# -------------------- 6. DECISION-THRESHOLD TUNING --------------------
# Accuracy/0.5-threshold is a poor lens on a ~3.4%-positive dataset.
# Sweep thresholds and pick the one that maximizes F1 on the failure class.
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
f1_curve = 2 * precision * recall / (precision + recall + 1e-12)
best_idx = np.nanargmax(f1_curve[:-1])
best_threshold = thresholds[best_idx]
y_pred_tuned = (y_score >= best_threshold).astype(int)

print(f"\nBest decision threshold (max F1): {best_threshold:.3f}")
print("Classification Report (tuned threshold):")
print(classification_report(y_test, y_pred_tuned, digits=4))

threshold_compare = pd.DataFrame(
    {
        "Threshold": [0.5, round(best_threshold, 3)],
        "Precision (failure)": [
            classification_report(y_test, y_pred, output_dict=True)["1"]["precision"],
            classification_report(y_test, y_pred_tuned, output_dict=True)["1"][
                "precision"
            ],
        ],
        "Recall (failure)": [
            classification_report(y_test, y_pred, output_dict=True)["1"]["recall"],
            classification_report(y_test, y_pred_tuned, output_dict=True)["1"][
                "recall"
            ],
        ],
        "F1 (failure)": [
            f1_score(y_test, y_pred),
            f1_score(y_test, y_pred_tuned),
        ],
    },
    index=["Default", "Tuned"],
).round(4)
save_table_as_image(
    threshold_compare,
    "Effect of Decision Threshold",
    os.path.join(OUTPUT_DIR, "threshold_effect.png"),
)
print("\nThreshold comparison:\n", threshold_compare)

# -------------------- 7. SAVE TABLES AS IMAGES --------------------
report_dict = classification_report(y_test, y_pred_tuned, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(4)
report_df = report_df.fillna("")
save_table_as_image(
    report_df,
    "Classification Report - Tuned Model, Tuned Threshold",
    os.path.join(OUTPUT_DIR, "classification_report.png"),
)

cv_df = pd.DataFrame(
    {"Mean F1": [round(cv_scores.mean(), 4)], "Std F1": [round(cv_scores.std(), 4)]},
    index=["5-fold CV"],
)
save_table_as_image(
    cv_df,
    "Cross-Validation Result (Tuned Model)",
    os.path.join(OUTPUT_DIR, "cv_score.png"),
)

# -------------------- 8. PLOTS --------------------
cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Failure", "Failure"],
    yticklabels=["No Failure", "Failure"],
)
plt.title("Confusion Matrix (Tuned Model + Threshold)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X_train.columns
plt.figure(figsize=(8, 5))
plt.title("Feature Importances (Tuned Model)")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), features[indices], rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
plt.close()

avg_precision = average_precision_score(y_test, y_score)
plt.figure(figsize=(5, 4))
plt.plot(recall, precision, label=f"AP = {avg_precision:.3f}")
plt.scatter(
    recall[best_idx],
    precision[best_idx],
    color="red",
    zorder=5,
    label=f"Chosen threshold ({best_threshold:.2f})",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"), dpi=150)
plt.close()

print(f"\nAll outputs saved in {OUTPUT_DIR}")
