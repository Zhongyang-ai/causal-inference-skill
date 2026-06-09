---
name: aipw-causal-inference
description: Use when the user needs augmented inverse probability weighting (AIPW) or doubly robust (DR) estimation for causal inference from observational data. AIPW combines an outcome regression model with propensity score weighting so the estimator is consistent if EITHER the propensity model OR the outcome model is correctly specified. Covers ATE and ATT estimation, the efficient influence function and its use for variance and confidence intervals, cross-fitting / sample splitting (TMLE-adjacent and double machine learning, DML, ideas) to control overfitting when using flexible ML learners, robustness to model misspecification, overlap/positivity checks, and clear reporting for treatment effect studies in Python, R, or SQL-backed analytics workflows.
---

# Augmented IPW (Doubly Robust)

Use this skill for observational treatment effect questions where the user wants an estimate that is more robust to model misspecification than either plain outcome regression or plain inverse probability weighting, by combining both models into a single doubly robust estimator.

## Workflow

1. Define the estimand before code:
   - Treatment, outcome, unit of analysis, time window, and eligible population.
   - Target estimand: ATE for the whole population, or ATT for the effect on treated units.
   - Confounders must be pre-treatment variables only; exclude mediators, colliders, and outcome-derived features.

2. Fit the outcome regression models:
   - Estimate `mu_1(x) = E[Y | T = 1, X = x]` and `mu_0(x) = E[Y | T = 0, X = x]`.
   - Fit two separate learners on the treated and control subsamples, or one learner with treatment as a feature, then predict both potential outcomes for every unit.
   - Any learner is allowed: linear/logistic regression, regularized regression, gradient boosting, random forest. Flexible learners require cross-fitting (step 5).

3. Fit the propensity model:
   - Estimate `e(x) = P(T = 1 | X = x)` with logistic regression or another calibrated classifier.
   - Inspect overlap/common support; positivity requires `e(x)` bounded away from 0 and 1.
   - Trim or truncate extreme propensities, since the weights `1/e` and `1/(1-e)` blow up the variance near the boundaries.

4. Combine into the AIPW estimator:
   - Per unit, form the doubly robust score: the outcome-regression difference `mu_1(x) - mu_0(x)` plus the inverse-probability-weighted residual correction (the augmentation term) for the observed arm.
   - The augmentation term corrects the IPW estimate using the outcome model and corrects the outcome model using the propensity weights; this is what delivers double robustness.
   - Average the scores over all units for ATE. For ATT, the score is not simply the ATE score reweighted by ATT IPW weights — it has its own doubly robust form (average over the treated of `Y - mu_0(x)` minus an IPW correction that uses only the control augmentation, scaled by the treated share). If you need ATT, derive its specific DR score rather than reusing the ATE estimator with different weights.

5. Use cross-fitting / sample splitting (strongly recommended):
   - Split the data into K folds. For each fold, fit both nuisance models (`mu` and `e`) on the other K-1 folds and predict on the held-out fold.
   - This decouples nuisance estimation from the effect estimate, removes overfitting bias, and is what gives AIPW its good theoretical guarantees with ML learners (the double machine learning / DML and TMLE-adjacent rationale).
   - Without cross-fitting, flexible learners overfit the data they will be evaluated on and bias the estimate.

6. Estimate the effect and its uncertainty:
   - Report the point estimate as the mean of the doubly robust scores.
   - Estimate the variance from the efficient influence function: the sample variance of the per-unit scores (centered at the estimate) divided by n. This is asymptotically valid and reaches the semiparametric efficiency bound when both models are consistent.
   - Report a 95% confidence interval; bootstrap is an alternative but must respect the cross-fitting structure.

7. Diagnose both models and overlap:
   - Check propensity overlap (distribution of `e(x)` by arm, share of extreme scores, effective sample size).
   - Check outcome model fit (residuals, calibration, cross-validated error) for each arm.
   - Compare AIPW against plain IPW and plain outcome regression; large disagreement signals misspecification or weak overlap in at least one component.

## Reporting Checklist

- Estimand, inclusion criteria, treatment timing, and outcome window.
- Outcome model and propensity model specifications, with the pre-treatment confounder list.
- Cross-fitting scheme: number of folds, learners, and how nuisances were predicted out-of-fold.
- Overlap/positivity diagnostics and any trimming or truncation of propensities.
- AIPW point estimate with influence-function-based standard error and 95% CI.
- Side-by-side comparison with plain IPW and plain outcome regression.
- Misspecification and unmeasured-confounding caveats.

## Guardrails

- Double robustness is not invincibility: if BOTH the propensity model and the outcome model are wrong, AIPW is still biased.
- Consistency requires that at least one of the two models is correctly specified; "doubly robust" means two chances, not a guarantee.
- Extreme propensity scores still inflate variance even when the point estimate stays unbiased; trimming and overlap checks remain essential.
- Flexible ML learners need cross-fitting; without sample splitting the theoretical guarantees and valid inference do not hold.
- Correlated or common-source misspecification (e.g., both models omit the same confounder or share the same wrong functional form) breaks the "at least one is correct" assumption.
- AIPW adjusts only for observed confounders; it does not solve unmeasured confounding. If overlap is weak, state that the causal question is poorly supported by the data.
