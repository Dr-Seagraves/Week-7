"""
Week 7: Logistic Regression Demo
=================================
Teaching binary outcome regression - when to use it, how to interpret it.

Key concepts:
- Logistic vs OLS (when to use which)
- Odds ratios
- Confusion matrices
- Prediction accuracy
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
import warnings

try:
    from sklearn.metrics import roc_curve, roc_auc_score
except ImportError:
    # Fallback: compute ROC curve and AUC without sklearn
    def roc_curve(y_true, y_score):
        thresholds = np.sort(np.unique(y_score))[::-1]
        fpr, tpr = [], []
        n_pos = (y_true == 1).sum()
        n_neg = (y_true == 0).sum()
        for t in np.append(thresholds, 0):
            pred = (y_score >= t).astype(int)
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            tpr.append(tp / n_pos if n_pos else 0)
            fpr.append(fp / n_neg if n_neg else 0)
        return np.array(fpr), np.array(tpr), np.append(thresholds, 0)

    def roc_auc_score(y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        # AUC = area under ROC curve (trapezoidal rule: integral of tpr d(fpr))
        return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
warnings.filterwarnings('ignore')

# Output directory: same folder as this script (works regardless of cwd)
OUTPUT_DIR = Path(__file__).resolve().parent

print("=" * 60)
print("WEEK 7: LOGISTIC REGRESSION")
print("=" * 60)

print("""
  What is this demo?
  -----------------
  We predict a YES/NO outcome (e.g., "Did this person get a job offer?")
  using variables like education, experience, and age.

  Key terms:
  - Binary outcome: the thing we predict (0 = no, 1 = yes)
  - Predictors: the variables we use to predict (Education, Experience, Age)
  - Probability: the model gives us a number between 0 and 1, like 0.82 = 82% chance

  We'll walk through each step and explain what each part means.
""")

# ============================================================================
# STEP 1: Generate Simulated Data
# ============================================================================
print("=" * 60)
print("STEP 1: The Data")
print("=" * 60)

print("\n  We use simulated data so the results are easy to reproduce.")
print("  Imagine we have 500 people with education, experience, and age.")
print("  For each person, we observe: did they get a job offer (yes=1, no=0)?")
print()

np.random.seed(42)
n = 500

# X1: Years of education (10-20)
education = np.random.uniform(10, 20, n)

# X2: Work experience (0-30 years)
experience = np.random.uniform(0, 30, n)

# X3: Age (22-65)
age = np.random.uniform(22, 65, n)

# Generate binary outcome: "Got Job Offer"
# True model: logit(p) = -8 + 0.5*education + 0.1*experience - 0.05*age
linear_combination = -8 + 0.5*education + 0.1*experience - 0.05*age
probability = 1 / (1 + np.exp(-linear_combination))
job_offer = (np.random.random(n) < probability).astype(int)

df = pd.DataFrame({
    'Education': education,
    'Experience': experience,
    'Age': age,
    'JobOffer': job_offer
})

print(f"  Total people: {n}")
print(f"  Outcome: {(job_offer == 1).sum()} got an offer ({(job_offer == 1).mean()*100:.1f}%), "
      f"{(job_offer == 0).sum()} did not ({(job_offer == 0).mean()*100:.1f}%)")
print("\n  Variables:")
print("    Education  = years of schooling")
print("    Experience = years of work experience")
print("    Age        = age in years")
print("    JobOffer   = 1 if they got an offer, 0 otherwise")
print("\n  Sample of the data (first 10 rows):")
print(df.head(10).round(2).to_string(index=False))

# ============================================================================
# STEP 2: Why Not OLS?
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Why Not Use Regular Regression (OLS)?")
print("=" * 60)

# Run OLS anyway (BAD PRACTICE but educational)
X_ols = sm.add_constant(df[['Education', 'Experience', 'Age']])
model_ols = sm.OLS(df['JobOffer'], X_ols).fit()

n_bad = ((model_ols.fittedvalues < 0) | (model_ols.fittedvalues > 1)).sum()
print("\n  If we used OLS on this 0/1 outcome, the model would predict")
print("  values that are NOT valid probabilities:")
print(f"    - Some predictions fall outside 0–1: {n_bad} cases")
print(f"    - Min prediction: {model_ols.fittedvalues.min():.3f}  (negative!)")
print(f"    - Max prediction: {model_ols.fittedvalues.max():.3f}")
print("\n  Takeaway: A probability must be between 0 and 1.")
print("  Logistic regression always gives predictions in that range.")

# ============================================================================
# STEP 3: Logistic Regression
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Fitting the Logistic Model")
print("=" * 60)

X_logit = sm.add_constant(df[['Education', 'Experience', 'Age']])
model_logit = Logit(df['JobOffer'], X_logit).fit(disp=0)

print("\n  The model estimates a coefficient for each predictor.")
print("  How to read:")
print("    Coef   = effect in 'log-odds' (we'll convert to odds ratio next)")
print("    P>|z|  = p-value; * = significant at 5%, ** at 1%, *** at 0.1%")
print()
print("-" * 55)
print(f"{'Variable':<14} {'Coef':>9} {'P>|z|':>8} {'Signif':>8}")
print("-" * 55)
for var in model_logit.params.index:
    coef = model_logit.params[var]
    p = model_logit.pvalues[var]
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"{var:<14} {coef:>9.3f} {p:>7.4f} {stars:>8}")
print("-" * 55)
print(f"  (Pseudo R-squared = {model_logit.prsquared:.3f})")
print("\n  Quick interpretation:")
print("    Positive Coef -> higher value of that variable = higher chance of offer")
print("    Negative Coef -> higher value = lower chance")
print("    We'll convert these to odds ratios next for easier interpretation.")

# ============================================================================
# CONCEPTUAL INTERLUDE: Log-Odds, Sigmoid, and Odds Ratios
# ============================================================================
print("\n" + "=" * 60)
print("HOW DOES LOGISTIC REGRESSION WORK?")
print("=" * 60)

print("""
  The model predicts a PROBABILITY (0 to 1), not the 0/1 outcome directly.
  Example: 0.82 means an 82% predicted chance of a job offer.

  Step A — Probability to odds:
     Odds = p / (1 - p). Examples:
       p = 0.25 -> odds = 0.33 (less likely than not)
       p = 0.50 -> odds = 1.00 (50/50)
       p = 0.75 -> odds = 3.00 (more likely than not)

  Step B — Log of odds:
     log-odds = log(p / (1-p)). This can be any real number, which lets us
     use a linear equation (like OLS) and then convert back to probability.
""")
for p in [0.25, 0.5, 0.75]:
    log_odds = np.log(p / (1 - p))
    print(f"       p = {p:.2f}  ->  log-odds = {log_odds:.2f}")

print("""
  Step C — The model in two parts:
     1. Linear part: z = intercept + coef1*X1 + coef2*X2 + ...
     2. Squeeze z into 0–1: P(Y=1) = 1 / (1 + e^{-z})  [sigmoid curve]

  Step D — Odds ratios for interpretation:
     OR = exp(coefficient). OR > 1 = increases odds; OR < 1 = decreases odds.
     Note: odds are not the same as probability. "Odds up 40%" ≠ "probability up 40%."
""")

beta_edu = model_logit.params['Education']
or_edu = np.exp(beta_edu)
print(f"  Worked example (Education):")
print(f"    Coef = {beta_edu:.3f}  ->  Odds Ratio = exp({beta_edu:.3f}) = {or_edu:.3f}")
print(f"    Meaning: Each extra year of education multiplies the odds by {or_edu:.2f}")
print(f"    (i.e., {(or_edu - 1) * 100:.1f}% higher odds per year).")

# ============================================================================
# STEP 4: Odds Ratios
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Odds Ratios (Easy to Interpret)")
print("=" * 60)

odds_ratios = np.exp(model_logit.params)

print("\n  Odds Ratio = exp(coefficient). We use it because it's easier to interpret.")
print()
print("-" * 50)
print(f"{'Variable':<14} {'Odds Ratio':>10} {'What it means'}")
print("-" * 50)

for var in ['Education', 'Experience', 'Age']:
    or_val = odds_ratios[var]
    if or_val > 1:
        interp = f"{(or_val-1)*100:.0f}% higher odds per unit"
    else:
        interp = f"{abs((or_val-1)*100):.0f}% lower odds per unit"
    print(f"{var:<14} {or_val:>10.3f}   {interp}")

print("-" * 50)
print(f"\n  Education:  +1 year -> {(odds_ratios['Education']-1)*100:.0f}% higher odds of offer")
print(f"  Experience: +1 year -> {(odds_ratios['Experience']-1)*100:.0f}% higher odds")
print(f"  Age:        +1 year -> {abs((odds_ratios['Age']-1)*100):.0f}% lower odds")
print("\n  Remember: these are odds changes, not probability changes.")

# ============================================================================
# STEP 5: Predictions and Confusion Matrix
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Predictions & How Well Did We Do?")
print("=" * 60)

# Get predicted probabilities
df['Pred_Prob'] = model_logit.predict(X_logit)

print("\n  Step 1: The model gives each person a predicted probability.")
print("          Example: 0.82 = 82% chance of an offer.")

print("\n  Step 2: We choose a cutoff (here, 0.5) to turn probabilities into yes/no.")
print("          If predicted prob > 0.5  -> we predict 'Offer'")
print("          If predicted prob ≤ 0.5 -> we predict 'No Offer'")

# Classify at 0.5 threshold
df['Pred_Class'] = (df['Pred_Prob'] > 0.5).astype(int)

# Confusion matrix manually
actual_no = df[df['JobOffer'] == 0]
actual_yes = df[df['JobOffer'] == 1]
TN = (actual_no['Pred_Class'] == 0).sum()
FP = (actual_no['Pred_Class'] == 1).sum()
FN = (actual_yes['Pred_Class'] == 0).sum()
TP = (actual_yes['Pred_Class'] == 1).sum()

accuracy = (TP + TN) / (TP + TN + FP + FN)
sensitivity = TP / (TP + FN)  # True positive rate
specificity = TN / (TN + FP)   # True negative rate
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

print("\n  Confusion matrix (rows = actual, columns = predicted):")
print("                        Predicted")
print("                    No Offer    Offer")
print("-" * 45)
print(f"  Actual No Offer    {TN:>5}       {FP:>5}")
print(f"  Actual Offer       {FN:>5}       {TP:>5}")
print("-" * 45)

print("\n  How to read it:")
print("    - Top-left (TN): Correctly predicted NO offer")
print("    - Top-right (FP): Wrong—we said Offer, they didn't get one")
print("    - Bottom-left (FN): Wrong—we said No Offer, they did get one")
print("    - Bottom-right (TP): Correctly predicted Offer")

print(f"\n  Summary metrics:")
print(f"    Accuracy   = {accuracy*100:.1f}%  (fraction of all predictions correct)")
print(f"    Sensitivity = {sensitivity*100:.1f}%  (of those who got offers, we caught)")
print(f"    Specificity = {specificity*100:.1f}%  (of those who didn't, we correctly said no)")
print(f"    Precision  = {precision*100:.1f}%  (of those we said yes to, how many actually got offers)")

# ============================================================================
# STEP 6: Visualizations
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: Creating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Logistic Regression - Week 7 Demo\nBinary Outcome: Job Offer', 
             fontsize=14, fontweight='bold')

# Plot 1: Sigmoid curve
ax1 = axes[0, 0]
x_range = np.linspace(10, 20, 100)
x_pred = np.column_stack([np.ones(100), x_range, 
                          np.full(100, df['Experience'].mean()),
                          np.full(100, df['Age'].mean())])
y_pred = model_logit.predict(x_pred)
ax1.plot(x_range, y_pred, 'b-', linewidth=2, label='Logistic Curve')
ax1.scatter(df['Education'], df['JobOffer'], alpha=0.3, s=20, c='red', label='Actual Data')
ax1.set_xlabel('Years of Education', fontsize=11)
ax1.set_ylabel('Probability of Job Offer', fontsize=11)
ax1.set_title('1. Logistic Sigmoid Curve\n(Probability vs Education)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# Plot 2: Odds ratios
ax2 = axes[0, 1]
vars_plot = ['Education', 'Experience', 'Age']
ors = [odds_ratios[v] for v in vars_plot]
colors = ['green' if o > 1 else 'red' for o in ors]
bars = ax2.barh(vars_plot, ors, color=colors, alpha=0.7)
ax2.axvline(x=1, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Odds Ratio', fontsize=11)
ax2.set_title('2. Odds Ratios\n(Green > 1 = increases odds, Red < 1 = decreases)', fontsize=12)
for i, (bar, or_val) in enumerate(zip(bars, ors)):
    ax2.text(or_val + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{or_val:.3f}', va='center', fontsize=10)

# Plot 3: Confusion matrix heatmap
ax3 = axes[1, 0]
cm = np.array([[TN, FP], [FN, TP]])
im = ax3.imshow(cm, cmap='Blues')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['No Offer', 'Offer'])
ax3.set_yticklabels(['No Offer', 'Offer'])
ax3.set_xlabel('Predicted', fontsize=11)
ax3.set_ylabel('Actual', fontsize=11)
ax3.set_title('3. Confusion Matrix', fontsize=12)
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax3, shrink=0.8)

# Plot 4: Accuracy by threshold
ax4 = axes[1, 1]
thresholds = np.linspace(0.1, 0.9, 50)
accuracies = []
for thresh in thresholds:
    pred = (df['Pred_Prob'] > thresh).astype(int)
    acc = (pred == df['JobOffer']).mean()
    accuracies.append(acc)

ax4.plot(thresholds, accuracies, 'b-', linewidth=2)
ax4.axvline(x=0.5, color='red', linestyle='--', label='Default (0.5)')
best_idx = np.argmax(accuracies)
best_thresh = thresholds[best_idx]
ax4.axvline(x=best_thresh, color='green', linestyle='--', label=f'Best ({best_thresh:.2f})')
ax4.set_xlabel('Classification Threshold', fontsize=11)
ax4.set_ylabel('Accuracy', fontsize=11)
ax4.set_title('4. Accuracy by Threshold\n(Optimal threshold may not be 0.5)', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
out_path = OUTPUT_DIR / 'logistic_regression_plots.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.close()

# Second figure: Log-odds-to-probability (sigmoid) + ROC curve
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle('Logistic Regression Concepts', fontsize=14, fontweight='bold')

# Chart A: Log-odds to probability (sigmoid) - z from -6 to +6
ax_sigmoid = axes2[0]
z_range = np.linspace(-6, 6, 200)
p_from_z = 1 / (1 + np.exp(-z_range))
ax_sigmoid.plot(z_range, p_from_z, 'b-', linewidth=2, label='P(Y=1) = 1/(1+e^{-z})')
ax_sigmoid.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
ax_sigmoid.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax_sigmoid.annotate('log-odds = 0\np = 0.5', xy=(0, 0.5), xytext=(1.5, 0.35),
                   fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
ax_sigmoid.set_xlabel('Linear predictor z (log-odds)', fontsize=11)
ax_sigmoid.set_ylabel('P(Y=1)', fontsize=11)
ax_sigmoid.set_title('Log-Odds to Probability\n(Sigmoid transformation)', fontsize=12)
ax_sigmoid.legend()
ax_sigmoid.grid(True, alpha=0.3)
ax_sigmoid.set_xlim(-6, 6)
ax_sigmoid.set_ylim(0, 1)

# Chart B: ROC curve
fpr, tpr, _ = roc_curve(df['JobOffer'], df['Pred_Prob'])
auc_score = roc_auc_score(df['JobOffer'], df['Pred_Prob'])
ax_roc = axes2[1]
ax_roc.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random guess')
ax_roc.set_xlabel('False Positive Rate', fontsize=11)
ax_roc.set_ylabel('True Positive Rate', fontsize=11)
ax_roc.set_title('ROC Curve\n(Model discrimination ability)', fontsize=12)
ax_roc.legend()
ax_roc.grid(True, alpha=0.3)

plt.tight_layout()
out_path2 = OUTPUT_DIR / 'logistic_regression_concepts.png'
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path2}")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("WHAT TO REMEMBER")
print("=" * 60)
print("""
  1. When to use logistic regression: binary outcome (yes/no). OLS can give
     invalid probabilities; logistic regression always stays between 0 and 1.

  2. Odds ratios: OR = exp(coefficient). OR > 1 means higher X → higher odds;
     OR < 1 means higher X → lower odds. Odds ≠ probability.

  3. Confusion matrix: shows correct vs incorrect predictions. Accuracy =
     (correct) / (total).

  4. Threshold: we use 0.5 by default, but the best cutoff depends on whether
     false positives or false negatives are costlier.

  This run: Education OR = {or_ed:.2f}, Experience OR = {or_ex:.2f}, Age OR = {or_ag:.2f}
  Accuracy = {acc:.1f}%, AUC = {auc:.3f} (1.0 = perfect, 0.5 = random).
""".format(
    or_ed=odds_ratios['Education'],
    or_ex=odds_ratios['Experience'],
    or_ag=odds_ratios['Age'],
    acc=accuracy * 100,
    auc=auc_score,
))

print("=" * 60)
print("LOGISTIC REGRESSION DEMO COMPLETE")
print("=" * 60)
