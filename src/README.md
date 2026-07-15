# Random Forest Failure Prediction

This project trains a baseline `RandomForestClassifier` on the AI4I 2020 predictive maintenance dataset. The script reads the prepared train and test CSV files, fits a model, prints evaluation metrics, and saves diagnostic plots for the test set.

## Project Structure

```text
.
├── main.py
├── datasets/
│   ├── ai4i2020.csv
│   ├── train_ai4i2020.csv
│   └── test_ai4i2020.csv
└── outputs/
    └── baseline_model/
```

## Requirements

Install the Python packages used by the script:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## What the Script Does

`main.py`:

1. Loads `datasets/train_ai4i2020.csv` and `datasets/test_ai4i2020.csv`.
2. Separates features from the `failure` target column.
3. Trains a random forest model with `class_weight='balanced'`.
4. Reports 5-fold cross-validation F1, test accuracy, and a classification report.
5. Saves plots to `outputs/baseline_model/`:
   - `confusion_matrix.png`
   - `feature_importance.png`
   - `roc_curve.png`
   - `pr_curve.png`

## Run

From the project root, execute:

```bash
python main.py
```

If you want to change the output folder, edit `FOLDER_NAME` in `main.py`.

## Expected Output

After the script finishes, the console will show model metrics and the generated plots will be written to `outputs/baseline_model/`.

## Notes

- The script expects the `failure` column to be present in both the train and test CSV files.
- The existing `ai4i2020.csv` file can be used as the source dataset for creating alternate train/test splits.
- The current workflow is a baseline implementation and can be extended with hyperparameter tuning, feature engineering, or additional models.
