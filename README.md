# Task 5: Decision Trees and Random Forests

**AI & ML Internship — Elevate Labs**
Objective: Learn tree-based models for classification & regression.

## Dataset
**UCI Heart Disease dataset** (`heart.csv`) — 303 patients, 13 clinical features
(age, sex, chest pain type, blood pressure, cholesterol, etc.), binary target:

- `target = 1` → presence of heart disease (165 cases)
- `target = 0` → absence of heart disease (138 cases)

No missing values.

## What was done
1. **Trained a Decision Tree Classifier** (unrestricted depth) and visualized it.
2. **Analyzed overfitting** by sweeping `max_depth` from 1–20 and comparing train vs test accuracy, then picked the best depth and re-visualized the pruned tree.
3. **Trained a Random Forest** (200 trees) and compared its accuracy to both tree versions.
4. **Interpreted feature importances** from the Random Forest.
5. **Evaluated with 5-fold cross-validation**, comparing the tree and the forest.

## Results

### Overfitting: full tree vs pruned tree vs forest

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Decision Tree (unrestricted, depth=8) | 1.000 | 0.705 |
| Decision Tree (max_depth=4, pruned) | 0.905 | 0.787 |
| Random Forest (200 trees) | 1.000 | **0.820** |

The unrestricted tree **memorizes the training set perfectly (100%)** but
generalizes poorly (70.5% on test) — classic overfitting. Limiting depth to 4
closes much of that gap and actually improves test accuracy. The Random
Forest gets the best test accuracy of all despite also fitting the training
set perfectly, because averaging over many decorrelated trees cancels out
individual trees' overfitting (see `overfitting_analysis.png`).

### Random Forest evaluation (test set)
- Confusion matrix: 18 TN, 10 FP, 1 FN, 32 TP
- Precision (disease): 0.76, Recall (disease): 0.97, F1: 0.85
- The model rarely *misses* real disease cases (only 1 false negative) but
  produces more false alarms — reasonable behavior for a screening tool.

### 5-fold cross-validation
| Model | CV Mean Accuracy | CV Std |
|---|---|---|
| Decision Tree (pruned) | 0.765 | 0.064 |
| Random Forest | **0.841** | 0.033 |

The Random Forest is both **more accurate on average and more consistent**
(lower variance) across folds — see `cross_validation.png`.

### Feature importances (Random Forest, top 5)
1. `cp` (chest pain type) — 0.154
2. `thal` (thalassemia result) — 0.119
3. `thalach` (max heart rate achieved) — 0.113
4. `oldpeak` (ST depression) — 0.111
5. `ca` (number of major vessels colored by fluoroscopy) — 0.087

These align with established clinical risk indicators for heart disease,
which is a good sanity check on the model.

## Files in this repo
- `decision_trees_random_forests.py` — full script (load → tree → overfitting sweep → forest → importances → CV)
- `heart.csv` — dataset
- `decision_tree_full.png` — unrestricted tree (top 3 levels shown)
- `decision_tree_pruned.png` — pruned tree (max_depth=4), fully visualized
- `overfitting_analysis.png` — train vs test accuracy across tree depths
- `rf_confusion_matrix.png` — Random Forest confusion matrix
- `feature_importances.png` / `.csv` — Random Forest feature importances
- `cross_validation.png` — CV accuracy boxplot, tree vs forest
- `depth_analysis.csv`, `model_comparison.csv`, `cross_validation_summary.csv` — raw numbers behind the plots

---

## Interview Questions

**1. How does a decision tree work?**
It splits the data recursively: at each node, it picks the feature and
threshold that best separates the classes (using a criterion like Gini
impurity or entropy/information gain), creating child nodes. This repeats
until a stopping condition is met (pure leaf, max depth, minimum samples,
etc.). To predict, a new sample simply follows the path of true/false
answers down the tree from the root to a leaf, and the leaf's majority class
(or mean value, for regression) is the prediction.

**2. What is entropy and information gain?**
**Entropy** measures the impurity/disorder of a set of labels:
`H(S) = -Σ pᵢ log₂(pᵢ)` where `pᵢ` is the proportion of class `i`. It's 0 for
a pure node (all one class) and maximal when classes are evenly mixed.
**Information gain** is the reduction in entropy achieved by a split:
`IG = H(parent) - Σ (weighted average of H(children))`. The tree-building
algorithm picks, at each node, the split that maximizes information gain
(scikit-learn's default, `gini`, is a similar but computationally cheaper
impurity measure that behaves very similarly in practice).

**3. How is random forest better than a single tree?**
A random forest trains many decision trees on **bootstrap samples** of the
data, and at each split only considers a **random subset of features**, then
averages (or majority-votes) their predictions. This de-correlates the trees,
so their individual errors tend to cancel out rather than compound — the
ensemble is much less prone to overfitting than any single deep tree, and
tends to have lower variance and better generalization, as seen here (Random
Forest test accuracy 0.82 vs 0.79 for the best single pruned tree, and much
better cross-validation stability).

**4. What is overfitting and how do you prevent it?**
Overfitting is when a model learns the training data's noise and specific
quirks rather than the underlying pattern, so it performs very well on
training data but poorly on unseen data — exactly what we saw with the
unrestricted tree (100% train accuracy, 70.5% test accuracy). Ways to prevent
it in tree models: limit `max_depth`, set a minimum number of samples per
leaf/split (`min_samples_leaf`, `min_samples_split`), prune the tree after
fitting (cost-complexity pruning), or move to an ensemble method like Random
Forest or Gradient Boosting that averages out individual trees' overfitting.
More generally: get more training data, use cross-validation to pick
hyperparameters (rather than tuning on the test set), and apply
regularization where the model supports it.

**5. What is bagging?**
**Bagging** (Bootstrap AGGregatING) is an ensemble technique that trains
multiple copies of a model on different **bootstrap samples** (random samples
drawn *with replacement* from the training data, each the same size as the
original set) and combines their predictions by averaging (regression) or
majority vote (classification). It reduces variance without increasing bias.
Random Forest is essentially bagging applied to decision trees, with the
extra twist of also randomizing the feature subset considered at each split.

**6. How do you visualize a decision tree?**
With scikit-learn, `sklearn.tree.plot_tree()` draws the tree directly with
matplotlib (used in this project — see `decision_tree_pruned.png`), or
`sklearn.tree.export_graphviz()` can export it to Graphviz's `.dot` format
for rendering with the Graphviz library (`dot -Tpng tree.dot -o tree.png`),
which handles larger trees more cleanly. Each node typically shows the split
condition, the impurity measure, the number of samples, the class
distribution, and the predicted/majority class.

**7. How do you interpret feature importance?**
In scikit-learn's tree-based models, feature importance is computed as the
total reduction in impurity (weighted by the number of samples) that a
feature provides across all the splits where it's used, averaged across all
trees in the forest, then normalized to sum to 1. A higher value means that
feature was more useful, on average, for separating the classes. It's a
relative ranking within this model/dataset — not a causal claim — and can be
biased toward high-cardinality or continuous features, so it's worth
cross-checking with domain knowledge (as we did here: `cp`, `thal`, and
`thalach` topping the list matches known cardiology risk factors).

**8. What are the pros/cons of random forests?**
**Pros:**
- Much less prone to overfitting than a single decision tree; strong
  out-of-the-box accuracy with minimal tuning.
- Handles both numerical and categorical features, missing values (with some
  preprocessing), and non-linear relationships without needing feature
  scaling.
- Provides built-in feature importance estimates.
- Robust to outliers and noisy features due to averaging over many trees.

**Cons:**
- Much less interpretable than a single tree — you can't easily trace "why"
  a prediction was made (though tools like SHAP help).
- Slower to train and predict than a single tree, and the model is larger to
  store (hundreds of trees).
- Can still overfit on very noisy data if `n_estimators` is too low or trees
  are too deep, and doesn't extrapolate well beyond the range of training
  data (a general limitation of tree-based models).
- Feature importance can be misleading with correlated features (importance
  gets split between them).
