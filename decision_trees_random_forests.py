"""
Task 5: Decision Trees and Random Forests
AI & ML Internship - Elevate Labs

Objective: Learn tree-based models for classification & regression.
Tools: Scikit-learn, Graphviz

Dataset: UCI Heart Disease dataset (heart.csv, 303 rows, 13 clinical features).
Target: 1 = presence of heart disease, 0 = absence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 0: Load dataset")
print("=" * 60)

df = pd.read_csv("heart.csv")
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Target balance:\n{df['target'].value_counts()}")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------------------------------------------------------------
# 1. Train a Decision Tree Classifier and visualize the tree
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 1: Decision Tree (unrestricted depth) + visualization")
print("=" * 60)

dt_full = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_full.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, dt_full.predict(X_train))
test_acc_full = accuracy_score(y_test, dt_full.predict(X_test))
print(f"Full tree depth: {dt_full.get_depth()}, leaves: {dt_full.get_n_leaves()}")
print(f"Train accuracy: {train_acc_full:.4f}")
print(f"Test accuracy:  {test_acc_full:.4f}")
print("-> Large gap between train and test accuracy signals overfitting.")

# Visualize the (unrestricted) tree - top few levels only, for readability
plt.figure(figsize=(20, 10))
plot_tree(
    dt_full,
    feature_names=X.columns,
    class_names=["No disease", "Disease"],
    filled=True,
    rounded=True,
    max_depth=3,  # show top 3 levels only; full tree is too large to read
    fontsize=9,
)
plt.title("Decision Tree (unrestricted, showing top 3 levels only)")
plt.tight_layout()
plt.savefig("decision_tree_full.png")
plt.close()
print("Saved plot -> decision_tree_full.png")

# ---------------------------------------------------------------
# 2. Analyze overfitting and control tree depth
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Overfitting analysis across max_depth")
print("=" * 60)

depths = range(1, 21)
train_accs, test_accs = [], []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
    test_accs.append(accuracy_score(y_test, dt.predict(X_test)))

depth_df = pd.DataFrame({"max_depth": list(depths), "train_acc": train_accs, "test_acc": test_accs})
depth_df.to_csv("depth_analysis.csv", index=False)
print(depth_df.to_string(index=False))

best_depth = depth_df.loc[depth_df["test_acc"].idxmax(), "max_depth"]
print(f"\nBest max_depth by test accuracy: {int(best_depth)}")

plt.figure(figsize=(8, 6))
plt.plot(depths, train_accs, marker="o", label="Train accuracy")
plt.plot(depths, test_accs, marker="o", label="Test accuracy")
plt.axvline(best_depth, color="gray", linestyle="--", alpha=0.7, label=f"Best depth = {int(best_depth)}")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Decision Tree: Overfitting Analysis (Train vs Test Accuracy)")
plt.legend()
plt.tight_layout()
plt.savefig("overfitting_analysis.png")
plt.close()
print("Saved plot -> overfitting_analysis.png")

# Fit the depth-controlled (pruned) tree
dt_pruned = DecisionTreeClassifier(max_depth=int(best_depth), random_state=RANDOM_STATE)
dt_pruned.fit(X_train, y_train)
pruned_test_acc = accuracy_score(y_test, dt_pruned.predict(X_test))
print(f"\nPruned tree (max_depth={int(best_depth)}) test accuracy: {pruned_test_acc:.4f}")

plt.figure(figsize=(20, 10))
plot_tree(
    dt_pruned,
    feature_names=X.columns,
    class_names=["No disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title(f"Pruned Decision Tree (max_depth={int(best_depth)})")
plt.tight_layout()
plt.savefig("decision_tree_pruned.png")
plt.close()
print("Saved plot -> decision_tree_pruned.png")

# ---------------------------------------------------------------
# 3. Train a Random Forest and compare accuracy
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Random Forest vs single tree")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Random Forest train accuracy: {rf_train_acc:.4f}")
print(f"Random Forest test accuracy:  {rf_test_acc:.4f}")

comparison = pd.DataFrame(
    {
        "Model": ["Decision Tree (unrestricted)", f"Decision Tree (max_depth={int(best_depth)})", "Random Forest (200 trees)"],
        "Train Accuracy": [train_acc_full, accuracy_score(y_train, dt_pruned.predict(X_train)), rf_train_acc],
        "Test Accuracy": [test_acc_full, pruned_test_acc, rf_test_acc],
    }
)
comparison.to_csv("model_comparison.csv", index=False)
print(f"\nModel comparison:\n{comparison.to_string(index=False)}")

y_pred_rf = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)
print(f"\nRandom Forest confusion matrix:\n{cm}")
print(f"\nRandom Forest classification report:\n{classification_report(y_test, y_pred_rf, target_names=['No disease', 'Disease'])}")

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No disease", "Disease"])
disp.plot(ax=ax, cmap="Greens", colorbar=False)
ax.set_title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
plt.close()
print("Saved plot -> rf_confusion_matrix.png")

# ---------------------------------------------------------------
# 4. Interpret feature importances
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Feature importances (Random Forest)")
print("=" * 60)

importances = pd.DataFrame(
    {"Feature": X.columns, "Importance": rf.feature_importances_}
).sort_values(by="Importance", ascending=False)
importances.to_csv("feature_importances.csv", index=False)
print(importances.to_string(index=False))

plt.figure(figsize=(8, 6))
plt.barh(importances["Feature"][::-1], importances["Importance"][::-1], color="steelblue")
plt.xlabel("Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()
print("Saved plot -> feature_importances.png")

# ---------------------------------------------------------------
# 5. Evaluate using cross-validation
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: 5-fold cross-validation")
print("=" * 60)

cv_dt = cross_val_score(DecisionTreeClassifier(max_depth=int(best_depth), random_state=RANDOM_STATE), X, y, cv=5)
cv_rf = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), X, y, cv=5)

print(f"Decision Tree CV scores: {np.round(cv_dt, 4)}")
print(f"Decision Tree CV mean +/- std: {cv_dt.mean():.4f} +/- {cv_dt.std():.4f}")
print(f"\nRandom Forest CV scores: {np.round(cv_rf, 4)}")
print(f"Random Forest CV mean +/- std: {cv_rf.mean():.4f} +/- {cv_rf.std():.4f}")

cv_summary = pd.DataFrame(
    {
        "Model": ["Decision Tree (pruned)", "Random Forest"],
        "CV Mean Accuracy": [cv_dt.mean(), cv_rf.mean()],
        "CV Std": [cv_dt.std(), cv_rf.std()],
    }
)
cv_summary.to_csv("cross_validation_summary.csv", index=False)

plt.figure(figsize=(7, 6))
plt.boxplot([cv_dt, cv_rf], tick_labels=["Decision Tree", "Random Forest"])
plt.ylabel("Accuracy (5-fold CV)")
plt.title("Cross-Validation Accuracy: Decision Tree vs Random Forest")
plt.tight_layout()
plt.savefig("cross_validation.png")
plt.close()
print("Saved plot -> cross_validation.png")

print("\nDone.")
