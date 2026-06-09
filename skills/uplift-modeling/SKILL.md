---
name: uplift-modeling
description: Use when the user needs uplift modeling to estimate individual or conditional treatment effects (CATE/ITE) and heterogeneous treatment effects, i.e. who-to-target / treatment targeting questions rather than a single average effect. Covers segmenting a population into persuadables, sure things, lost causes, and sleeping dogs; meta-learners (S-learner, T-learner, X-learner) and uplift/causal trees; ranking-based evaluation with the Qini curve, uplift curve, and AUUC/Qini coefficient instead of ordinary classification or regression metrics; and marketing campaign targeting, retention, and budget allocation. Best suited to settings with a treatment flag and an outcome, ideally from a randomized A/B test or quasi-random/de-confounded assignment, where the goal is individual-level heterogeneity. Applies to Python, R, or SQL-backed analytics workflows.
---

# Uplift Modeling (Heterogeneous Treatment Effects)

Use this skill when the business question is "who should we treat / target?" rather than "what is the average effect?". Uplift modeling estimates the conditional average treatment effect `tau(x) = E[Y(1) - Y(0) | X = x]` for each individual or subgroup, so a fixed budget can be spent on the units whose outcome the treatment actually changes.

## Workflow

1. Frame the target as an individual/subgroup effect, not an average:
   - Confirm the goal is targeting or prioritization (rank units by how much the treatment moves their outcome), not a single ATE/ATT number.
   - State the treatment, the outcome, the unit of analysis, the time window, and the eligible population.
   - Decide the targeting decision the estimate will drive (whom to send the offer to, what fraction of the population to treat, expected incremental gain).

2. Check the data and identification before modeling:
   - You need a treatment indicator `T` and an outcome `Y` for both treated and control units, plus pre-treatment covariates `X`.
   - The cleanest source is a randomized experiment / A/B test, where treatment is independent of potential outcomes by design and raw treated-minus-control differences are unbiased.
   - With observational data, treatment is confounded: de-confound first (propensity-score weighting / IPW, matching, or a doubly robust transformation such as AIPW pseudo-outcomes) before fitting CATE, otherwise the estimated uplift is biased.
   - Verify positivity/overlap: every covariate region must contain both treated and control units, or uplift there is extrapolated.

3. Choose a meta-learner (or uplift tree):
   - S-learner: one model trained on `X` plus the treatment `T` as a feature; predict with `T=1` and `T=0` and take the difference. Simple, but can underfit small treatment effects and bias them toward zero if the learner ignores `T`.
   - T-learner: two separate models, one fit on treated units and one on control units; uplift is the difference of their predictions. Flexible, but variance is high when one arm is small and the two models' errors do not cancel.
   - X-learner: builds on the T-learner by imputing individual treatment effects and combining the two arm-specific CATE models with a propensity weight; preferred when treatment and control groups are imbalanced.
   - Uplift/causal trees and forests split directly on the difference in response between arms (maximizing divergence in uplift) rather than on outcome purity; useful for interpretable segments.

4. Train and predict CATE per unit:
   - Fit the chosen learner on a training split and predict uplift on a held-out split; never evaluate ranking on the training data.
   - Keep base learners regularized; CATE is a noisy difference of two predictions and overfits easily, especially on small or imbalanced samples.

5. Evaluate ranking ability, not pointwise accuracy:
   - Rank units by predicted uplift (descending) and build the uplift curve and the Qini curve, which track cumulative incremental response (treated minus rescaled control) as more of the ranked population is treated.
   - Summarize with the AUUC (area under the uplift curve) or the Qini coefficient, and compare against the random-targeting diagonal; a model that only ranks at chance gains nothing over treating everyone at random.
   - Sanity-check monotonicity: bin units by predicted uplift and confirm the realized treated-minus-control response (or known true CATE in simulation) rises across bins.

6. Translate uplift into a targeting policy:
   - Map predicted uplift to the four segments: persuadables (positive uplift, treat them), sure things (convert regardless, treatment wasted), lost causes (never convert, treatment wasted), and sleeping dogs (negative uplift, treatment backfires, do not treat).
   - Choose a treatment cutoff (top-k by uplift, or uplift above a cost-justified threshold) and report the expected incremental outcome and the share of the population treated.
   - Explicitly exclude or suppress negative-uplift sleeping dogs; treating them destroys value.

## Reporting Checklist

- The targeting question and the decision the uplift estimate will drive.
- Data source and assignment mechanism: randomized A/B test, quasi-random, or observational with the de-confounding method applied.
- Covariates used, confirmed to be pre-treatment, and positivity/overlap status.
- Meta-learner (S/T/X) or tree choice and base learner, with the train/test split.
- Qini curve / uplift curve plus AUUC or Qini coefficient versus the random baseline.
- Monotonicity check: realized response (or true CATE) by predicted-uplift bin.
- Segment breakdown (persuadables / sure things / lost causes / sleeping dogs) and the recommended targeting cutoff with expected incremental gain.
- Sample size, class balance per arm, and variance/stability caveats.

## Guardrails

- Do not evaluate uplift with ordinary classification or regression metrics (accuracy, AUC, RMSE on `Y`): a model can predict the outcome well yet rank treatment effect poorly. Use Qini/uplift curves and AUUC, which score the ranking of `tau(x)`.
- The individual treatment effect is never observed for any unit (only one potential outcome is seen), so CATE cannot be validated pointwise; rely on aggregate ranking metrics and, in simulation, on the known true effect.
- With observational data, fitting CATE on raw treated-vs-control differences is biased by confounding; de-confound first (IPW/matching/AIPW) and treat uplift as a layer on top of an unbiased baseline.
- Interpret all four quadrants: a positive average effect can hide sleeping dogs whom the treatment harms; targeting them, or treating sure things and lost causes, wastes budget or backfires.
- Uplift is a small, noisy difference of two predictions; small samples, imbalanced arms, and rare outcomes inflate variance. Regularize, validate on held-out data, and report uncertainty rather than chasing in-sample lift.
- A randomized assignment is the gold standard; if treatment was selected by a rule or by humans, say so and de-confound, do not present biased uplift as causal targeting.
