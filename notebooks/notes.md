# Feature Selection for Regression — Filter vs. Wrapper vs. Embedded


| Category     | Technique                                            |    Model-dependent? |     Speed    |         Accuracy/Robustness         | When to use                                                                    | Why (what it gives you)                                                                                   |
| ------------ | ---------------------------------------------------- | ------------------: | :----------: | :---------------------------------: | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| **Filter**   | **PCC (Pearson Correlation Coefficient)**            |                   ❌ | ⚡️ Very fast |     👎 Approximate, linear only     | First pass on numeric features with roughly linear relationships to the target | Ranks features by **linear** association with *y*; great for quick pruning.                               |
| **Filter**   | **Spearman ρ (Rank Correlation)**                    |                   ❌ | ⚡️ Very fast |     👍 More robust than Pearson     | Early pass when relationships may be **monotonic but non-linear**              | Uses ranks, so it captures monotonic trends and is less sensitive to outliers.                            |
| **Filter**   | **Kendall τ (Rank Correlation)**                     |                   ❌ |    ⚡️ Fast   |  👍 Robust, stricter than Spearman  | Datasets with many ties / small samples                                        | Counts concordant/discordant pairs; more conservative than Spearman.                                      |
| **Filter**   | **Information Gain / Mutual Information**            |                   ❌ |    ⚡️/⚡️⚡️   | 👍 Captures non-linear dependencies | Early pass when relationships may be **non-linear**                            | Measures how much information a feature provides about *y* (use `mutual_info_regression` for regression). |
| **Wrapper**  | **Forward Selection**                                |                   ✅ |    🐢 Slow   |         ✅ High (model-aware)        | After filter pass, with a **manageable** number of candidates (e.g., ≤ 20)     | Starts empty and **adds** features that actually **improve validation score**.                            |
| **Wrapper**  | **Backward Elimination**                             |                   ✅ |    🐢 Slow   |         ✅ High (model-aware)        | When starting from **all** features and trimming down                          | Starts full and **removes** features that **don’t help** or hurt generalization.                          |
| **Embedded** | **Lasso (L1) / Elastic Net**                         | ✅ (during training) |   ⚡️ Medium  |         ✅ High (regularized)        | Linear/GLM baselines; want **sparse** models                                   | Penalizes weights → pushes weak features to **0** (built-in selection).                                   |
| **Embedded** | **Tree-based Importances (RF / XGBoost / LightGBM)** | ✅ (during training) |   ⚡️ Medium  |                ✅ High               | Non-linear relationships, interactions                                         | Models compute **split/gain importances** while training; handles non-linearity out of the box.           |


---

## Cross Validation (CV)

### **What it is**

A model evaluation technique that checks **how stable and generalizable** a model is — instead of trusting one random train/test split.

---

### **When to use it**

Use CV **after** you split your data into **train / test** sets:

* **Training set** → for model training & cross-validation
* **Test set** → for the **final** unbiased evaluation only

> You never apply CV on the test set — the test set must remain unseen until the very end.

---

### **How it works**

1. Take the **training set only**.
2. Split it into *k* smaller parts (folds), e.g., 5 or 10.
3. Train the model *k* times:

   * Each time, use one fold for validation and the rest for training.
4. Compute the average score (MAE, MSE, R², etc.) across all folds.

This average represents how the model performs on unseen data more reliably than a single validation split.

---

### **Why it matters**

| Benefit                   | Description                                                           |
| ------------------------- | --------------------------------------------------------------------- |
| **Stability check**       | Ensures model performance is consistent across different data splits. |
| **Better generalization** | Reduces the chance of “lucky” or “unlucky” splits.                    |
| **Efficient data use**    | Every sample is used for both training and validation at least once.  |

---

Sure ✅ Here’s a clean, clear English version you can paste directly into your README:

---

### Cross-Validation, Scaling, and Model Setup

During **Cross-Validation (CV)** you must define both:

1. **The model** you want to evaluate (e.g., DecisionTreeRegressor)
2. **The preprocessing steps** (like scaling or encoding)

You combine them into a single **Pipeline** so that scaling and model training happen **inside each fold** — preventing any **data leakage**.

In each CV iteration:

* `fit()` / `fit_transform()` are applied **only on the training fold (K-1 folds)**
* `transform()` is applied **on the validation fold (1 fold)**
* The test set is **never touched** during CV.

This ensures every fold simulates “new, unseen data” and the evaluation is fair.

> Always run CV on the **training set only**,
> and include preprocessing inside the **Pipeline** to avoid data leakage.

---

## ROC Curve & AUC (Area Under the Curve)

### **What it is**

The **ROC Curve (Receiver Operating Characteristic)** is a performance graph used for **classification models** (not regression).
It shows how well a model distinguishes between **positive** and **negative** classes at different threshold values.

---

### **How it works**

For every possible threshold (from 0 → 1), the ROC curve plots:

* **True Positive Rate (TPR)** on the Y-axis → how many positives are correctly detected
* **False Positive Rate (FPR)** on the X-axis → how many negatives are wrongly predicted as positive

The curve shows the trade-off between **sensitivity** and **false alarms**.

---

### **AUC (Area Under the Curve)**

* **AUC** = the **area under the ROC curve** → a single score summarizing overall performance.
* It ranges between **0 and 1**:

  | AUC value | Meaning           |
  | --------- | ----------------- |
  | 1.0       | Perfect model ✅   |
  | 0.9–0.8   | Excellent         |
  | 0.8–0.7   | Good              |
  | 0.7–0.6   | Weak              |
  | 0.5       | Random guessing ❌ |

---

### **Main purpose**

> To measure how well the model **separates classes**,
> regardless of the decision threshold.

It’s especially useful for **imbalanced datasets**, where accuracy alone can be misleading.

---

**ROC/AUC** is only used for **classification** because it relies on true/false outcomes (TPR, FPR).
For **regression**, outputs are continuous values, so we use metrics like **MAE, MSE, or R²** to measure prediction accuracy instead.

---

## Regression Evaluation Metrics

| Metric                                    | What it measures                                                   | Why we use it                                                                | Value meaning                                                        | Notes / Difference                                                          |
| ----------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **MAE (Mean Absolute Error)**             | Average of absolute differences between predicted and true values. | Simple, easy to interpret — shows average error in original units.           | Lower = better. `0` = perfect predictions.                           | Treats all errors equally; less sensitive to outliers.                      |
| **MSE (Mean Squared Error)**              | Mean of squared differences between predicted and true values.     | Penalizes large errors more strongly. Useful when large errors are very bad. | Lower = better. `0` = perfect fit.                                   | Sensitive to outliers (squares errors). Used often as a **loss function**.  |
| **RMSE (Root Mean Squared Error)**        | Square root of MSE.                                                | Easier to interpret since it’s in the same units as the target variable.     | Lower = better. Same scale as target.                                | Shows “typical” error size; still penalizes large errors more.              |
| **MAPE (Mean Absolute Percentage Error)** | Average of absolute percentage errors.                             | Useful to express error as a **percentage** of the true value.               | Lower = better. `0%` = perfect.                                      | Not reliable when true values are near zero.                                |
| **R² (Coefficient of Determination)**     | How much of the target’s variance is explained by the model.       | Shows overall explanatory power of the model.                                | 1 = perfect fit, 0 = no fit, can be negative if worse than baseline. | Intuitive measure of how well predictions follow the actual trend.          |
| **Adjusted R²**                           | R² adjusted for number of predictors (features).                   | Prevents overestimating model quality when adding many features.             | Always ≤ R². Higher = better.                                        | Increases only if a new feature **improves** the model more than by chance. |

---

### Quick summary

* **MAE** → Average error (equal weight to all).
* **MSE / RMSE** → Same idea but penalize large errors more.
* **MAPE** → Expresses error in percentage terms.
* **R² / Adjusted R²** → Explain how much variance the model captures.

---





