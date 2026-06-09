---
name: ipw-causal-inference
description: Use when the user needs inverse probability weighting (IPW) or inverse probability of treatment weighting (IPTW) for causal inference from observational data, including propensity-score weighting, stabilized weights, ATE or ATT estimation via weighting, an introduction to marginal structural models, positivity/overlap diagnostics, weight trimming and truncation, effective sample size, Horvitz-Thompson and Hajek estimators, and covariate balance after weighting using weighted standardized mean differences. Applies to Python, R, Stata, or SQL-backed analytics workflows for observational treatment effect studies.
---

# Inverse Probability Weighting (IPW)

Use this skill for observational treatment effect questions where the user wants to reweight units by the inverse of their treatment probability to build a pseudo-population in which treatment is independent of measured confounders, then estimate ATE or ATT on that pseudo-population.

## Workflow

1. Define the estimand before code:
   - Treatment, outcome, unit of analysis, time window, and eligible population.
   - Target estimand: ATE for the whole population, or ATT for the effect on treated units.
   - Confounders must be pre-treatment variables only; never condition on mediators, colliders, or outcome-derived features.

2. Estimate the propensity score:
   - Estimate `e(x) = P(T = 1 | X)` with logistic regression, regularized logistic regression, gradient boosting, or another calibrated classifier.
   - Calibration matters more than raw discrimination; the weights depend on the predicted probabilities, not on classification accuracy.
   - Inspect the propensity distribution by treatment arm before constructing any weights.

3. Construct the weights:
   - For ATE, weight treated units by `1 / e(x)` and control units by `1 / (1 - e(x))`.
   - For ATT, weight treated units by `1` and control units by `e(x) / (1 - e(x))`, reweighting controls to look like the treated population.
   - Prefer stabilized weights: multiply the ATE numerator by the marginal treatment probability, so treated weight is `P(T=1) / e(x)` and control weight is `P(T=0) / (1 - e(x))`. Stabilization shrinks weight variance without changing the target estimand.

4. Check positivity and extreme weights:
   - Verify positivity/overlap: every unit should have `0 < e(x) < 1` with adequate support in both arms; near-deterministic propensities break IPW.
   - Inspect the weight distribution: report the maximum weight, the ratio of max to mean, and the share of total weight carried by the top few units.
   - Trim units in the tails of the propensity distribution, or truncate (Winsorize) weights at percentile cutoffs such as the 1st/99th, when a few units would otherwise dominate. Document any trimming/truncation because it changes the estimand slightly.

5. Estimate the treatment effect:
   - Use the Hajek (self-normalizing) estimator, i.e. weighted means normalized by the sum of weights in each arm, which is more stable than the raw Horvitz-Thompson sum.
   - Equivalently, fit a weighted outcome regression of the outcome on treatment (the marginal structural model in its simplest form).
   - Report uncertainty with the bootstrap (refitting the propensity model inside each replicate) or a robust/sandwich variance; naive unweighted standard errors understate uncertainty.

6. Validate balance and stability:
   - Report weighted standardized mean differences (weighted SMD) for every covariate; aim for absolute weighted SMD below `0.1`.
   - Report the effective sample size (ESS) per arm; a large drop versus the nominal count signals weight concentration and unstable estimates.
   - Try alternative propensity specifications, stabilized versus unstabilized weights, and several trimming thresholds; report how the estimate moves.

## Reporting Checklist

- Estimand (ATE or ATT), inclusion criteria, treatment timing, and outcome window.
- Propensity model and covariates, with justification that they are pre-treatment confounders.
- Positivity/overlap diagnostics and the propensity distribution by arm.
- Weight type (ATE vs ATT, stabilized vs unstabilized), weight summary statistics, and the maximum weight.
- Any trimming or truncation rule applied and how many units it affected.
- Weighted SMD balance table before and after weighting.
- Effective sample size per arm.
- Treatment effect estimate with bootstrap or robust confidence interval.
- Sensitivity analyses and unmeasured-confounding caveats.

## Guardrails

- Extreme weights inflate variance: a handful of near-zero or near-one propensities can dominate the estimate and produce wide, unstable intervals.
- Positivity is the binding assumption: if some covariate region has no treated (or no control) units, the effect there is not identified and weighting will extrapolate silently.
- A misspecified propensity model biases pure IPW with no internal warning; consider doubly robust estimation (AIPW) when feasible, which stays consistent if either the propensity or the outcome model is correct.
- Always stabilize weights and inspect ESS; unstabilized weights routinely have far larger variance for the same point estimate.
- Good predictive propensity scores are not the goal; covariate balance after weighting is.
- IPW adjusts only for observed confounders and does not address unmeasured confounding.
