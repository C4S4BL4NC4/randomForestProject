import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# -------------------- 1. LOAD DATA --------------------
train_df = pd.read_csv('./datasets/train_ai4i2020.csv')
test_df  = pd.read_csv('./datasets/test_ai4i2020.csv')

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

# Cross‑validation (optional, but informative)
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
print(f"5-fold CV F1-score: {cv_scores.mean():.4f}")

# -------------------- 3. PREDICT & CONSOLE OUTPUT --------------------
y_pred = rf.predict(X_test)

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------- 4. GENERATE & SAVE PLOTS --------------------
FOLDER_NAME = "baseline_model"   # change to "tuned_model" etc. as needed
OUTPUT_DIR = f"./outputs/{FOLDER_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4a. Confusion matrix
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

# 4b. Feature importance
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

# 4c. ROC & AUC (needs predicted probabilities)
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

# 4d. Precision‑Recall curve
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

print(f"\nAll plots saved in {OUTPUT_DIR}")