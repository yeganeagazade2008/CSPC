import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

print("--- STEP A2: CLEAN AND INSPECT ---")
# 1. Load data from CSV
df = pd.read_csv("data/concrete.csv")

# 2. Check for missing values and duplicates
print(f"Missing values:\n{df.isna().sum()}")
print(f"Duplicate rows count: {df.duplicated().sum()}")

# 3. Remove duplicates and reset index
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dataset shape after dropping duplicates: {df.shape}")

# Rebuild X and y from cleaned data
X = df.drop(columns=["strength"])
y = df["strength"]

# 4. Print statistical summary
print("\n--- DATA DESCRIPTION ---")
print(df.describe())

# 5. Plot strength against cement and age
os.makedirs("results", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X["cement"], y, alpha=0.5, color="blue")
axes[0].set_title("Strength vs Cement")
axes[0].set_xlabel("Cement (kg/m3)")
axes[0].set_ylabel("Strength (MPa)")

axes[1].scatter(X["age"], y, alpha=0.5, color="green")
axes[1].set_title("Strength vs Age")
axes[1].set_xlabel("Age (days)")
axes[1].set_ylabel("Strength (MPa)")

plt.tight_layout()
plt.savefig("results/A2_plots.png")
print("Saved plots to 'results/A2_plots.png'")

print("\n--- STEP A3: TRAIN-TEST SPLIT ---")
# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n--- STEP A4: LINEAR REGRESSION ---")
# 1. Fit Linear Regression model
lin_reg = LinearRegression(fit_intercept=True)
lin_reg.fit(X_train, y_train)

# 2. Predictions
y_train_pred = lin_reg.predict(X_train)
y_test_pred = lin_reg.predict(X_test)

# 3. Metrics calculation
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Adjusted R2 calculation
# formula: 1 - [(1 - R2) * (n - 1) / (n - p - 1)]
n = X_train.shape[0]
p = X_train.shape[1]
train_r2_adj = 1 - ((1 - train_r2) * (n - 1) / (n - p - 1))

print(f"Train RMSE: {train_rmse:.4f} MPa")
print(f"Test RMSE: {test_rmse:.4f} MPa")
print(f"Test MAE: {test_mae:.4f} MPa")
print(f"Train R2: {train_r2:.4f}")
print(f"Train Adjusted R2: {train_r2_adj:.4f}")
print(f"Test R2: {test_r2:.4f}")

# Print fitted equation coefficients
print("\nFitted Coefficients (MPa per unit):")
for col, coef in zip(X.columns, lin_reg.coef_):
    print(f"  {col}: {coef:.4f}")
print(f"Intercept (beta_0): {lin_reg.intercept_:.4f}")

# 4. Residuals plot against age (log scale)
residuals = y_test - y_test_pred

plt.figure(figsize=(8, 5))
plt.scatter(X_test["age"], residuals, alpha=0.6, color="purple")
plt.axhline(0, color="red", linestyle="--")
plt.xscale("log")
plt.title("Linear Regression Residuals vs Age (Log Scale)")
plt.xlabel("Age (days, log scale)")
plt.ylabel("Residuals (True - Predicted Strength, MPa)")
plt.tight_layout()
plt.savefig("results/A4_residuals.png")
print("\nSaved residuals plot to 'results/A4_residuals.png'")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

print("\n--- STEP A5: POLYNOMIAL REGRESSION ---")
degrees = [1, 2, 3, 4]
results_poly = []

for d in degrees:
  # Build pipeline: Poly features -> Scaler -> Linear Regression
  poly_model = make_pipeline(
      PolynomialFeatures(degree=d, include_bias=False),
      StandardScaler(),
      LinearRegression(),
  )

  poly_model.fit(X_train, y_train)

  # Number of output features after polynomial expansion
  p_features = poly_model[0].n_output_features_

  # Predictions
  y_tr_pred = poly_model.predict(X_train)
  y_te_pred = poly_model.predict(X_test)

  # Metrics
  tr_rmse = np.sqrt(mean_squared_error(y_train, y_tr_pred))
  te_rmse = np.sqrt(mean_squared_error(y_test, y_te_pred))
  tr_r2 = r2_score(y_train, y_tr_pred)
  te_r2 = r2_score(y_test, y_te_pred)

  # Adjusted R2 for training set
  n_rows = X_train.shape[0]
  tr_r2_adj = 1 - ((1 - tr_r2) * (n_rows - 1) / (n_rows - p_features - 1))

  results_poly.append({
      "Degree": d,
      "Features (p)": p_features,
      "Train RMSE": tr_rmse,
      "Test RMSE": te_rmse,
      "Train R2": tr_r2,
      "Train Adj R2": tr_r2_adj,
      "Test R2": te_r2,
  })

# Print results table
poly_df = pd.DataFrame(results_poly)
print(poly_df.to_string(index=False))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

print("\n--- STEP A6: CLASSIFICATION (KNN & Decision Tree) ---")
# 1. Create binary target: 1 if strength >= 40 MPa, else 0
y_class = (df["strength"] >= 40).astype(int)

# Stratified split to keep class proportions identical
Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

print(f"Class proportions in training set:\n{c_train.value_counts(normalize=True)}")

# 2. KNN Sweep over K values
k_values = [1, 3, 5, 7, 15, 25, 51, 101]
knn_results = []

for K in k_values:
  # Pipeline with Scaler + KNN (since KNN relies on distances)
  knn_pipe = make_pipeline(
      StandardScaler(), KNeighborsClassifier(n_neighbors=K)
  )
  knn_pipe.fit(Xc_train, c_train)

  tr_acc = knn_pipe.score(Xc_train, c_train)
  te_acc = knn_pipe.score(Xc_test, c_test)

  knn_results.append({"K": K, "Train Accuracy": tr_acc, "Test Accuracy": te_acc})

knn_df = pd.DataFrame(knn_results)
print("\nKNN Sweep Results:")
print(knn_df.to_string(index=False))

# Plot KNN accuracies against K (log scale)
plt.figure(figsize=(8, 5))
plt.plot(
    knn_df["K"],
    knn_df["Train Accuracy"],
    marker="o",
    label="Train Accuracy",
    color="blue",
)
plt.plot(
    knn_df["K"],
    knn_df["Test Accuracy"],
    marker="s",
    label="Test Accuracy",
    color="orange",
)
plt.xscale("log")
plt.title("KNN Accuracy vs K (Log Scale)")
plt.xlabel("K Neighbors (log scale)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("results/A6_knn_sweep.png")
print("\nSaved KNN sweep plot to 'results/A6_knn_sweep.png'")
# 3. Decision Tree Sweep over max_depth values
depths = [1, 2, 3, 4, 5, 6, 8, None]
tree_results = []

for depth in depths:
  tree_model = DecisionTreeClassifier(
      criterion="entropy", max_depth=depth, min_samples_leaf=1, random_state=0
  )
  tree_model.fit(Xc_train, c_train)

  tr_acc = tree_model.score(Xc_train, c_train)
  te_acc = tree_model.score(Xc_test, c_test)
  n_leaves = tree_model.get_n_leaves()

  tree_results.append({
      "Max Depth": str(depth),
      "Leaves": n_leaves,
      "Train Accuracy": tr_acc,
      "Test Accuracy": te_acc,
  })

tree_df = pd.DataFrame(tree_results)
print("\nDecision Tree Sweep Results:")
print(tree_df.to_string(index=False))

# Plot Tree accuracies against depth
# Convert None to a string/number label for plotting
plt.figure(figsize=(8, 5))
plt.plot(
    [str(d) for d in depths],
    tree_df["Train Accuracy"],
    marker="o",
    label="Train Accuracy",
    color="blue",
)
plt.plot(
    [str(d) for d in depths],
    tree_df["Test Accuracy"],
    marker="s",
    label="Test Accuracy",
    color="orange",
)
plt.title("Decision Tree Accuracy vs Max Depth")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, ls="--")
plt.tight_layout()
plt.savefig("results/A6_tree_sweep.png")
print("\nSaved Decision Tree sweep plot to 'results/A6_tree_sweep.png'")

# 4. Print top of the chosen tree (e.g., max_depth=3)
chosen_tree = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, min_samples_leaf=1, random_state=0
)
chosen_tree.fit(Xc_train, c_train)
print("\nTop of Decision Tree (Max Depth = 3):")
print(export_text(chosen_tree, feature_names=list(X.columns), max_depth=2))

# 5. Repeat chosen KNN model (K=7) across 5 different splits
print("\n--- KNN STABILITY ACROSS 5 SPLITS (K=7) ---")
from sklearn.base import clone

accuracies = []
for seed in range(5):
  Xtr, Xte, ytr, yte = train_test_split(
      X, y_class, test_size=0.2, random_state=seed, stratify=y_class
  )
  pipe_split = make_pipeline(
      StandardScaler(), KNeighborsClassifier(n_neighbors=7)
  )
  pipe_split.fit(Xtr, ytr)
  acc = pipe_split.score(Xte, yte)
  accuracies.append(acc)
  print(f"Seed {seed} Test Accuracy: {acc:.4f}")

print(
    f"Mean Accuracy: {np.mean(accuracies):.4f} | Spread (Min-Max):"
    f" {np.min(accuracies):.4f} - {np.max(accuracies):.4f}"
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

print("\n--- STEP A7: METRICS THAT MATTER ---")
# Train chosen KNN (K=7) and chosen Tree (max_depth=5 or unlimited) for evaluation
knn_final = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
knn_final.fit(Xc_train, c_train)
knn_preds = knn_final.predict(Xc_test)

tree_final = DecisionTreeClassifier(
    criterion="entropy", max_depth=5, min_samples_leaf=1, random_state=0
)
tree_final.fit(Xc_train, c_train)
tree_preds = tree_final.predict(Xc_test)


# Helper function to compute all metrics
def evaluate_model(y_true, y_pred, model_name):
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
  acc = accuracy_score(y_true, y_pred)
  prec = precision_score(y_true, y_pred)
  rec = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  return {
      "Model": model_name,
      "TP": tp,
      "FP": fp,
      "FN": fn,
      "TN": tn,
      "Accuracy": acc,
      "Precision": prec,
      "Recall": rec,
      "F1": f1,
  }


metrics_list = [
    evaluate_model(c_test, knn_preds, "KNN (K=7)"),
    evaluate_model(c_test, tree_preds, "Decision Tree (Depth=5)"),
]

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df.to_string(index=False))

# Threshold analysis for KNN probabilities
print("\n--- THRESHOLD ANALYSIS (KNN) ---")
knn_Probas = knn_final.predict_proba(Xc_test)[:, 1]

thresholds = [0.3, 0.5, 0.7, 0.9]
thresh_results = []

for t in thresholds:
  t_preds = (knn_Probas >= t).astype(int)
  tn, fp, fn, tp = confusion_matrix(c_test, t_preds).ravel()
  prec = precision_score(c_test, t_preds, zero_division=0)
  rec = recall_score(c_test, t_preds, zero_division=0)
  thresh_results.append({
      "Threshold": t,
      "TP": tp,
      "FP": fp,
      "FN": fn,
      "TN": tn,
      "Precision": prec,
      "Recall": rec,
  })

thresh_df = pd.DataFrame(thresh_results)
print(thresh_df.to_string(index=False))

# Save A metrics table to results/
os.makedirs("results", exist_ok=True)
metrics_df.to_csv("results/A_metrics.csv", index=False)
print(
    "\nSaved metrics summary table to 'results/A_metrics.csv' for Session B"
    " use!"
)
print("\n--- BONUS: DOMAIN KNOWLEDGE FEATURES ---")

# 1. Create copies of original X_train and X_test
X_train_bonus = X_train.copy()
X_test_bonus = X_test.copy()

# 2. Transform age using logarithmic scaling: log(age)
X_train_bonus["age"] = np.log(X_train_bonus["age"])
X_test_bonus["age"] = np.log(X_test_bonus["age"])

# 3. Compute water-to-binder ratio (w/b = water / (cement + slag + ash))
binder_train = (
    X_train_bonus["cement"] + X_train_bonus["slag"] + X_train_bonus["ash"]
)
binder_test = (
    X_test_bonus["cement"] + X_test_bonus["slag"] + X_test_bonus["ash"]
)

X_train_bonus["w_b_ratio"] = X_train_bonus["water"] / binder_train
X_test_bonus["w_b_ratio"] = X_test_bonus["water"] / binder_test

# Drop the original 'water' column
X_train_bonus = X_train_bonus.drop(columns=["water"])
X_test_bonus = X_test_bonus.drop(columns=["water"])

# 4. Fit simple Linear Regression on domain-engineered features
bonus_reg = LinearRegression()
bonus_reg.fit(X_train_bonus, y_train)

y_bonus_pred = bonus_reg.predict(X_test_bonus)
bonus_test_r2 = r2_score(y_test, y_bonus_pred)
bonus_train_r2 = r2_score(y_train, bonus_reg.predict(X_train_bonus))

print(f"Bonus Model Features Count: {X_train_bonus.shape[1]}")
print(f"Bonus Train R2: {bonus_train_r2:.4f}")
print(f"Bonus Test R2: {bonus_test_r2:.4f}")
# --- BONUS: FEATURE IMPORTANCE ANALYSIS (Session A) ---
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd

# 1. Train a Random Forest model to check feature importance for concrete strength
# (Assuming X_train and y_train are already defined in your sessionA script)
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)

# 2. Create a DataFrame for feature importances
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=True)

# 3. Plot and save feature importance
plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance for Concrete Compressive Strength')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/feature_importance.png')
plt.close()

# Save CSV version as well
importance_df.to_csv('results/feature_importance.csv', index=False)
print("\nBonus Session A results (Feature Importance plot and CSV) saved to 'results/'!")