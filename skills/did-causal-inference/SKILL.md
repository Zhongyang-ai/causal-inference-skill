---
name: did-causal-inference
description: Use when the user needs difference-in-differences (DID) for causal inference, including DID treatment effect estimation, the parallel trends assumption, two-period/two-group (2x2) designs, policy or intervention/program evaluation, natural experiments, event studies for dynamic effects, two-way fixed effects (TWFE) regressions, pre-trend checks, staggered adoption caveats (Goodman-Bacon, Callaway-Sant'Anna, Sun-Abraham), ATT estimation, and clear reporting for panel data or repeated cross-section treatment-effect studies in Python, R, Stata, or SQL-backed analytics workflows.
---

# Difference-in-Differences (DID)

Use this skill for treatment effect questions where the user observes outcomes before and after an intervention for a treated group and a comparison group, and wants to isolate the causal effect by differencing out time trends and fixed group differences.

## Workflow

1. Define the estimand before code:
   - Treatment (the policy/intervention), outcome, unit of analysis, and the timing of the intervention.
   - Identify the treated group (exposed to the intervention) and the control group (not exposed but observed over the same periods).
   - Identify the pre period and post period relative to the intervention date.
   - Target estimand: typically ATT (effect of treatment on the treated). State the eligible population and time window.
   - Data structure: panel (same units observed over time) or repeated cross-section (different units, same groups/periods).

2. Check the parallel trends assumption:
   - Parallel trends requires that, absent treatment, treated and control groups would have evolved with the same outcome trend.
   - It is not directly testable; support it with pre-treatment evidence.
   - With multiple pre periods, plot and test pre-trends: treated and control should track each other before the intervention.
   - With only two periods, rely on domain reasoning, placebo groups/periods, and similarity of baseline trends.

3. Estimate the effect:
   - 2x2 DID by hand: `DID = (Ybar_treated_post - Ybar_treated_pre) - (Ybar_control_post - Ybar_control_pre)`.
   - Regression form: `Y = b0 + b1*Treat + b2*Post + b3*(Treat x Post) + e`, where `b3` is the DID estimate (ATT under parallel trends).
   - Multi-period / many-unit designs: two-way fixed effects (TWFE) with unit and time fixed effects, treatment indicator switching on post-adoption.
   - Add pre-treatment covariates only if they are not affected by treatment and improve the comparability of trends.

4. Get the uncertainty right:
   - Cluster standard errors at the level of treatment assignment (commonly the unit or group), not the observation, to handle serial correlation.
   - With few clusters, use cluster-robust corrections, wild cluster bootstrap, or report the limitation.
   - Report the DID estimate with a confidence interval, and a practical effect size, not just a p-value.

5. Run an event study for dynamic effects:
   - Interact treatment with leads and lags of event time, omitting one pre period as the baseline.
   - Lead (pre-treatment) coefficients near zero support parallel trends; lag coefficients trace the dynamic treatment path.
   - Use the event study to detect anticipation effects and effect ramp-up/decay.

6. Stress test the conclusion:
   - Placebo tests: fake treatment dates or untreated groups should yield null effects.
   - Vary the control group and the estimation window.
   - Under staggered adoption, check whether naive TWFE is biased and use a robust estimator (Callaway-Sant'Anna, Sun-Abraham, de Chaisemartin-D'Haultfoeuille) or Goodman-Bacon decomposition.
   - State residual risks from confounding shocks that hit one group only.

## Reporting Checklist

- Estimand (usually ATT), treated vs control definition, intervention timing, pre/post windows.
- Data structure (panel or repeated cross-section) and unit of analysis.
- Parallel trends evidence: pre-trend plot/test or the reasoning when untestable.
- DID estimate from the 2x2 means and/or the interaction regression, shown to agree.
- Standard errors with the clustering level stated, plus confidence interval.
- Event-study results for dynamic effects and anticipation.
- Placebo/robustness checks and staggered-adoption handling.
- Caveats: untestable parallel trends, composition change, spillovers.

## Guardrails

- Parallel trends cannot be proven, only supported; passing a pre-trend test is necessary but not sufficient.
- Under staggered adoption (units treated at different times) with heterogeneous effects, naive TWFE can be biased and even wrong-signed because already-treated units act as controls; use a modern estimator.
- Composition change (the units making up a group differ across periods, especially in repeated cross-sections) can masquerade as a treatment effect.
- DID assumes no spillovers: the control group must not be affected by the treatment (SUTVA). Contaminated controls bias the estimate.
- A level difference between groups is not an effect; only the differential change after treatment is.
- DID controls for time-invariant unobserved confounders but not for time-varying confounders or shocks that hit only one group.
- Do not interpret the ATT as an ATE unless the design and homogeneity assumptions support it.
