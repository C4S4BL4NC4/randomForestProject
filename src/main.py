import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
)

# -------------------- 0. HELPER: save DataFrame as PNG table --------------------
def save_table_as_image(df, title, filepath, fontsize=10):
    """Save a pandas DataFrame as a nicely formatted PNG table."""
    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.8, len(df) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(title, fontsize=fontsize + 2, weight='bold', pad=10)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.2, 1.4)

    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

# -------------------- 1. LOAD DATA (robust paths) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')

train_path = os.path.join(DATASETS_DIR, 'train_ai4i2020.csv')
test_path  = os.path.join(DATASETS_DIR, 'test_ai4i2020.csv')

if not os.path.exists(train_path):
    original_path = os.path.join(DATASETS_DIR, 'ai4i2020.csv')
    if not os.path.exists(original_path):
        raise FileNotFoundError(
            f"Neither {train_path} nor {original_path} found.\n"
        )
    # Recreate train/test split
    df = pd.read_csv(original_path)
    df['failure'] = (df['Machine failure'] == 1).astype(int)
    drop_cols = ['UDI', 'Product ID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(columns=drop_cols)
    y = df['failure']
    X = pd.get_dummies(X, columns=['Type'], drop_first=False)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_df = X_tr.copy()
    train_df['failure'] = y_tr
    test_df  = X_te.copy()
    test_df['failure'] = y_te
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("Train/test files regenerated successfully.")
else:
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

X_train = train_df.drop('failure', axis=1)
y_train = train_df['failure']
X_test  = test_df.drop('failure', axis=1)
y_test  = test_df['failure']

# -------------------- 2. TRAIN MODEL --------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

# Cross‑validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
print(f"5-fold CV F1-score: {cv_scores.mean():.4f}")

# -------------------- 3. PREDICT & CONSOLE OUTPUT --------------------
y_pred = rf.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------- 4. OUTPUT DIRECTORY --------------------
FOLDER_NAME = "baseline_model"
OUTPUT_DIR = f"./outputs/{FOLDER_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------- 5. SAVE TABLES AS IMAGES --------------------
# 5a. Classification report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(3)
report_df = report_df.fillna('')   # clean up accuracy row
save_table_as_image(
    report_df,
    title='Classification Report – Test Set',
    filepath=os.path.join(OUTPUT_DIR, 'classification_report.png'),
    fontsize=10
)

# 5b. Cross‑validation score
cv_df = pd.DataFrame({
    'Mean F1': [round(cv_scores.mean(), 4)],
    'Std F1':  [round(cv_scores.std(), 4)]
}, index=['5-fold CV'])
save_table_as_image(
    cv_df,
    title='Cross‑Validation Result',
    filepath=os.path.join(OUTPUT_DIR, 'cv_score.png'),
    fontsize=10
)

# -------------------- 6. GENERATE & SAVE PLOTS --------------------
# 6a. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Failure', 'Failure'],
            yticklabels=['No Failure', 'Failure'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

# 6b. Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X_train.columns

plt.figure(figsize=(8, 5))
plt.title('Feature Importances')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), features[indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
plt.close()

# 6c. ROC & AUC
y_score = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'), dpi=150)
plt.close()

# 6d. Precision‑Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_score)
avg_precision = average_precision_score(y_test, y_score)

plt.figure(figsize=(5, 4))
plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curve.png'), dpi=150)
plt.close()

# -------------------- 7. (Optional) PARAMETER SENSITIVITY --------------------
# Uncomment the block below to generate a parameter effect table & plot
"""
param_name = 'n_estimators'
param_values = [10, 50, 100, 200, 500]
f1_list = []

for v in param_values:
    rf_temp = RandomForestClassifier(
        n_estimators=v, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_temp.fit(X_train, y_train)
    y_pred_temp = rf_temp.predict(X_test)
    f1_list.append(f1_score(y_test, y_pred_temp))

# Table
param_df = pd.DataFrame({'n_estimators': param_values, 'F1-score': f1_list})
param_df.set_index('n_estimators', inplace=True)
print("\nEffect of n_estimators on F1-score:")
print(param_df)

save_table_as_image(
    param_df,
    title='Effect of n_estimators on F1-score',
    filepath=os.path.join(OUTPUT_DIR, 'param_effect.png'),
    fontsize=10
)

# Plot
plt.figure()
plt.plot(param_values, f1_list, marker='o')
plt.xlabel('n_estimators')
plt.ylabel('F1-score (Failure class)')
plt.title('Sensitivity to Number of Trees')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'param_effect_plot.png'), dpi=150)
plt.close()
"""

print(f"\nAll outputs saved in {OUTPUT_DIR}")