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

# ============================================================================
# STEP 1: Generate Simulated Data
# ============================================================================
print("\n" + "=" * 60)
print("STEP 1: Generating Simulated Data")
print("=" * 60)
print("We will pretend we are predicting whether a person gets a job offer.")
print("The outcome is binary: 1 = yes, 0 = no.")

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

print(f"Generated {n} observations")
print(f"\nOutcome distribution:")
print(f"  No Job Offer: {(job_offer == 0).sum()} ({(job_offer == 0).mean()*100:.1f}%)")
print(f"  Job Offer:    {(job_offer == 1).sum()} ({(job_offer == 1).mean()*100:.1f}%)")
print("\nPredictors in this demo:")
print("  - Education = years of education")
print("  - Experience = years of work experience")
print("  - Age = age in years")
print("\nFirst 10 rows:")
print(df.head(10).round(2))

# ============================================================================
# STEP 2: Why Not OLS? ( demonstrate the problem)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Why Not OLS for Binary Outcomes?")
print("=" * 60)

# Run OLS anyway (BAD PRACTICE but educational)
X_ols = sm.add_constant(df[['Education', 'Experience', 'Age']])
model_ols = sm.OLS(df['JobOffer'], X_ols).fit()

print("\n>>> PROBLEM: OLS on binary data gives:")
print(f"   - Predicted values outside [0,1]: {((model_ols.fittedvalues < 0) | (model_ols.fittedvalues > 1)).sum()} cases")
print(f"   - Min prediction: {model_ols.fittedvalues.min():.3f}")
print(f"   - Max prediction: {model_ols.fittedvalues.max():.3f}")
print("\n   OLS assumes continuous outcomes with normal errors.")
print("   Binary outcomes violate these assumptions!")
print("   Most importantly: a probability should stay between 0 and 1.")
print("   Logistic regression is built to do exactly that.")

# ============================================================================
# STEP 3: Logistic Regression
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Fitting Logistic Regression")
print("=" * 60)

X_logit = sm.add_constant(df[['Education', 'Experience', 'Age']])
model_logit = Logit(df['JobOffer'], X_logit).fit(disp=0)

print("\nLogistic Regression Results:")
print("-" * 60)
print(f"{'Variable':<15} {'Coef':>10} {'Std Err':>10} {'z':>8} {'P>|z|':>8}")
print("-" * 60)
for var in model_logit.params.index:
    coef = model_logit.params[var]
    se = model_logit.bse[var]
    z = model_logit.tvalues[var]
    p = model_logit.pvalues[var]
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"{var:<15} {coef:>10.4f} {se:>10.4f} {z:>8.2f} {p:>7.4f} {stars}")
print("-" * 60)
print("Pseudo R-squared:", f"{model_logit.prsquared:.4f}")
print("Log-Likelihood:", f"{model_logit.llf:.2f}")
print("AIC:", f"{model_logit.aic:.2f}")
print("\nPlain-English note:")
print("  - Positive coefficient -> higher values of that variable are associated with")
print("    a HIGHER chance of a job offer.")
print("  - Negative coefficient -> higher values are associated with a LOWER chance.")
print("  - But the coefficients are not yet in probability units, so interpretation")
print("    is easier after we convert them to odds ratios.")

# ============================================================================
# CONCEPTUAL INTERLUDE: Log-Odds, Sigmoid, and Odds Ratios
# ============================================================================
print("\n" + "=" * 60)
print("CONCEPTUAL INTERLUDE: How Does Logistic Regression Work?")
print("=" * 60)

print("""
BIG PICTURE:
   Logistic regression does NOT predict the 0/1 outcome directly.
   It predicts a probability, like 0.18 or 0.82.
   Then we can turn that probability into a yes/no prediction if we want.

>>> STEP A: START WITH A PROBABILITY
   Let p = P(JobOffer = 1)
   Example:
   - p = 0.25 means a 25% chance of getting an offer
   - p = 0.50 means a 50% chance
   - p = 0.75 means a 75% chance

>>> STEP B: CONVERT PROBABILITY TO ODDS
   Odds = p / (1 - p)
   Examples:
   - If p = 0.25, odds = 0.25 / 0.75 = 0.33
   - If p = 0.50, odds = 0.50 / 0.50 = 1.00
   - If p = 0.75, odds = 0.75 / 0.25 = 3.00

   Read these as:
   - odds = 0.33 means the event is less likely than not
   - odds = 1.00 means 50/50
   - odds = 3.00 means the event is more likely than not

>>> STEP C: TAKE THE LOG OF THE ODDS
   Log-odds = log(odds) = log(p / (1-p))
   Why do this?
   Because log-odds can take any value from -infinity to +infinity,
   which works nicely with a linear model.

   Example mapping (probability -> log-odds):
""")
for p in [0.25, 0.5, 0.75]:
    log_odds = np.log(p / (1 - p))
    print(f"   p = {p:.2f}  ->  log-odds = {log_odds:.2f}")

print("""
>>> HOW DOES THE MODEL WORK? (Two steps)
   Step 1 - LINEAR PART:
      z = beta_0 + beta_1*X1 + beta_2*X2 + ...
      This part looks like OLS.
      z can be any real number (-infinity to +infinity).

   Step 2 - TURN z INTO A PROBABILITY:
      P(Y=1) = 1 / (1 + exp(-z))
      This is the logistic curve, also called the sigmoid curve.
      It squeezes any z-value into a valid probability between 0 and 1.

   VERY IMPORTANT:
   The coefficients are in log-odds units, not probability units.
   So a 1-unit increase in X changes log-odds by beta.
   That is mathematically correct, but not very intuitive.
   For interpretation, we usually exponentiate the coefficient.

>>> ODDS RATIO FORMULA: OR = exp(beta)
   - OR > 1: one-unit increase in X MULTIPLIES the odds upward
   - OR < 1: one-unit increase in X MULTIPLIES the odds downward
   - OR = 1: no effect

   Quick warning:
   Odds are NOT the same thing as probability.
   Saying "odds go up by 40%" does NOT mean
   "probability goes up by 40 percentage points."
""")
# Worked example using Education coefficient
beta_edu = model_logit.params['Education']
or_edu = np.exp(beta_edu)
print(f"   WORKED EXAMPLE (Education):")
print(f"   beta_education = {beta_edu:.3f}  ->  OR = exp({beta_edu:.3f}) = {or_edu:.3f}")
print(f"   Interpretation: Holding the other variables fixed, each extra year of")
print(f"   education multiplies the odds of a job offer by {or_edu:.2f}.")
print(f"   That means the odds change by {(or_edu - 1) * 100:.1f}% for each extra year.")
print()

# ============================================================================
# STEP 4: Odds Ratios
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Odds Ratios (Easy to Interpret!)")
print("=" * 60)

odds_ratios = np.exp(model_logit.params)

print("\n>>> ODDS RATIOS: exp(coefficient)")
print("-" * 60)
print(f"{'Variable':<15} {'Odds Ratio':>12} {'Interpretation':<30}")
print("-" * 60)

for var in ['Education', 'Experience', 'Age']:
    or_val = odds_ratios[var]
    if or_val > 1:
        interp = f"+{(or_val-1)*100:.1f}% odds per unit"
    else:
        interp = f"{(or_val-1)*100:.1f}% odds per unit"
    print(f"{var:<15} {or_val:>12.4f} {interp:<30}")

print("-" * 60)
print("\n>>> INTERPRETATION EXAMPLES:")
print(f"   Education: Each additional year -> {(odds_ratios['Education']-1)*100:.1f}% HIGHER odds of a job offer")
print(f"   Experience: Each additional year -> {(odds_ratios['Experience']-1)*100:.1f}% HIGHER odds of a job offer")
print(f"   Age: Each additional year -> {abs((odds_ratios['Age']-1)*100):.1f}% LOWER odds of a job offer")
print("\nImportant reminder:")
print("   These are changes in odds, not direct changes in probability.")
print("   The probability effect depends on where you start.")

# ============================================================================
# STEP 5: Predictions and Confusion Matrix
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Predictions & Confusion Matrix")
print("=" * 60)

# Get predicted probabilities
df['Pred_Prob'] = model_logit.predict(X_logit)

print("\nFirst, logistic regression gives each person a predicted probability.")
print("Example: 0.82 means an 82% predicted chance of a job offer.")
print("Only after that do we choose a cutoff, such as 0.50, to turn")
print("probabilities into yes/no predictions.")

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

print("\nConfusion Matrix:")
print("                    Predicted")
print("                 No Offer  |  Offer")
print("-" * 40)
print(f"Actual No Offer:    {TN:>4}   |  {FP:>4}")
print(f"Actual Offer:       {FN:>4}   |  {TP:>4}")
print("-" * 40)

print(f"\nAccuracy:   {accuracy*100:.1f}%")
print(f"Sensitivity (Recall): {sensitivity*100:.1f}% - Of those who got offers, we predicted correctly")
print(f"Specificity: {specificity*100:.1f}% - Of those who didn't, we predicted correctly")
print(f"Precision:  {precision*100:.1f}% - Of those we predicted to get offers, actually got them")
print("\nHow to read the confusion matrix:")
print("  - True Positive (TP): predicted offer, and offer actually happened")
print("  - True Negative (TN): predicted no offer, and no offer actually happened")
print("  - False Positive (FP): predicted offer, but no offer happened")
print("  - False Negative (FN): predicted no offer, but offer actually happened")

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
print("SUMMARY: KEY TAKEAWAYS")
print("=" * 60)
print("""
1. WHEN TO USE LOGISTIC REGRESSION:
   - Binary outcome (yes/no, 0/1)
   - OLS would give predictions outside [0,1]
   - Logistic regression gives valid probabilities between 0 and 1
   
2. ODDS RATIOS:
   - exp(coefficient) = odds ratio
   - >1: increases odds of outcome
   - <1: decreases odds of outcome
   - Odds are not the same as probability
   
3. CONFUSION MATRIX:
   - Shows true/false positives and negatives
   - Accuracy = (TP + TN) / Total
   
4. THRESHOLD MATTERS:
   - Default is 0.5, but optimal varies
   - Depends on cost of false positives vs false negatives
""")

# Parrot-style recap: echo key numbers from this run
print("\n>>> PARROT: What You Just Saw (this run)")
print("-" * 60)
print(f"   Education OR = {odds_ratios['Education']:.3f} -> each year multiplies odds by {odds_ratios['Education']:.2f}")
print(f"   Experience OR = {odds_ratios['Experience']:.3f}, Age OR = {odds_ratios['Age']:.3f}")
print(f"   Confusion: {TP + TN} correct, {FP + FN} errors | Accuracy = {accuracy*100:.1f}%")
print(f"   AUC = {auc_score:.3f} (1.0 = perfect, 0.5 = random)")
print("-" * 60)
print("   Main idea: logistic regression predicts probabilities for 0/1 outcomes,")
print("   and odds ratios help us describe how predictors are associated with those probabilities.")

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION DEMO COMPLETE!")
print("=" * 60)
