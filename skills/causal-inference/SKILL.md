---
name: causal-inference
description: >-
  Use as the entry point for any causal inference or treatment-effect question
  when the right method is not yet decided. This router inspects the user's data
  shape (cross-sectional vs panel, randomized vs observational, sample size) and
  business goal (average effect vs who-to-target), then recommends and dispatches
  to the appropriate sub-skill — PSM (psm-causal-inference), IPW
  (ipw-causal-inference), AIPW/doubly-robust (aipw-causal-inference), DID
  (did-causal-inference), or Uplift modeling (uplift-modeling). Triggers include
  评估某个动作/投放/政策的因果效果, incrementality, lift, treatment effect, ATE,
  ATT, CATE, 该对谁投放 (who-to-target), A/B test analysis with confounding,
  observational treatment-effect studies, policy evaluation, and questions like
  用哪种因果推断方法 / which causal method should I use. Prefer this skill first
  whenever the user uploads data and asks for a causal effect without naming a
  specific method.
---

# Causal Inference Method Router

This is the **entry-point skill** for causal inference. When a user wants to estimate a causal / treatment effect but has not committed to a specific method, use this skill to (1) diagnose their data and goal, (2) recommend the best-fit method, and (3) hand off to the matching sub-skill. The five sub-skills are:

| Sub-skill | Method | Estimand | Core data requirement |
| --- | --- | --- | --- |
| `psm-causal-inference` | Propensity Score Matching | ATT (also ATE) | Cross-sectional observational data, measured confounders |
| `ipw-causal-inference` | Inverse Probability Weighting | ATE / ATT | Cross-sectional observational data, good overlap |
| `aipw-causal-inference` | Augmented IPW / Doubly Robust | ATE / ATT | Observational data; want robustness to model misspecification |
| `did-causal-inference` | Difference-in-Differences | ATT | Panel / repeated cross-section, pre & post periods, a control group |
| `uplift-modeling` | Uplift Modeling (CATE) | Individual/subgroup effect (CATE) | Treatment + outcome, ideally randomized; goal is targeting |

## Workflow

1. **Clarify the causal question before anything else.** Establish, in plain language:
   - Treatment / intervention `T` (the action whose effect we want).
   - Outcome `Y` and the unit of analysis (user, order, store, region…).
   - Eligible population and the time window.
   - **The decision the answer will inform** — this determines the estimand.

2. **Determine the estimand — this is the first fork.**
   - "What is the *average* effect of the treatment?" → ATE / ATT → consider PSM, IPW, AIPW, DID.
   - "*Who* should I treat / who responds most?" → individual/subgroup effect (CATE) → **Uplift modeling**.
   - If the user wants both, estimate the average effect first, then layer Uplift for targeting.

3. **Inspect the data shape — the second fork.** Ask for or infer:
   - **Assignment mechanism**: randomized/quasi-random (A/B test) vs observational (self-selected).
   - **Time structure**: a single snapshot (cross-section) vs before-and-after observations for the same/comparable units (panel or repeated cross-section).
   - **Is there a clean control group and a pre-period?** (decisive for DID).
   - Sample size, number of confounders, and outcome type (binary/continuous).

4. **Apply the recommendation logic** (see the decision guide below) and state *why* the chosen method fits — naming the assumption it relies on and the one it cannot rescue.

5. **Hand off to the chosen sub-skill** for the detailed workflow, diagnostics, and a runnable `example.py`. Each sub-skill ships an `EXPLANATION.md` (Chinese, with English terms) explaining its principle, assumptions, and pitfalls.

6. **When two methods are viable, recommend a primary plus a robustness check.** Triangulating across methods (e.g. PSM vs IPW, or IPW vs AIPW) and showing the estimate is stable is far more convincing than a single number.

## Recommendation Logic (decision guide)

Walk these checks in order; the first match is the primary recommendation.

1. **Goal is targeting / heterogeneous effect ("对谁投放", who are the persuadables, individual ROI)?**
   → **Uplift modeling**. Works best on randomized/A-B data; on observational data, de-confound first (PSM/IPW/AIPW) before modeling CATE.

2. **Do you have before-and-after data for a treated group AND a comparable control group (panel or repeated cross-section), e.g. a policy/feature went live on a date?**
   → **DID**. Strongest reason to choose it: it differences out *time-invariant unobserved* confounders, which the propensity-based methods cannot touch. Requires the **parallel trends** assumption — check pre-trends.

3. **Only a single cross-sectional snapshot, observational (non-random) treatment, and you want an average effect?** Then it is a propensity-score family problem — pick among the three by sub-goal:
   - **Want an interpretable matched cohort / ATT, or need to show stakeholders "comparable treated vs untreated units"?** → **PSM**. Discards non-overlapping units; intuitive but loses sample.
   - **Want ATE over the whole population using all data, and overlap is adequate?** → **IPW**. Uses every unit via weights; watch for extreme weights / positivity.
   - **Want robustness to getting a model wrong, or you want to use ML models for nuisance functions?** → **AIPW (doubly robust)**. Consistent if *either* the propensity model *or* the outcome model is correct; use cross-fitting. This is the safest default among the three when you can build both models.

4. **Randomized experiment (clean A/B test) and you only need the average effect?** A simple difference in means is already unbiased — causal machinery is optional. Use IPW/AIPW only to correct imperfect randomization or attrition, or move to **Uplift** if the question is targeting.

### Quick reference table

| Situation | Primary | Robustness check |
| --- | --- | --- |
| Policy/feature launched on a date, have pre/post + control | DID | Event-study / placebo pre-trend |
| Observational snapshot, want ATT, need interpretability | PSM | IPW on same data |
| Observational snapshot, want population ATE | IPW | AIPW |
| Observational snapshot, unsure which model is right | AIPW | PSM or IPW |
| "Who should we target?" / budget-constrained outreach | Uplift | De-confound (IPW/AIPW) first if observational |
| Clean randomized A/B, want average effect | Difference in means | IPW for imperfect randomization |

## Reporting Checklist

- The causal question: treatment, outcome, unit, population, time window, and the decision it informs.
- The estimand (ATE / ATT / CATE) and why it matches the decision.
- Data diagnosis: assignment mechanism, time structure, control group, sample size, overlap.
- The recommended method **and** the assumption it leans on plus the bias it cannot fix.
- At least one robustness/triangulation result when feasible.
- A clear statement of residual risk (unmeasured confounding, weak overlap, untestable parallel trends).

## Guardrails

- **No method fixes a bad design.** If treatment is entangled with the outcome's cause and there is no overlap, no pre-period, and no randomization, say the causal question is not answerable from this data.
- **Match the estimand to the decision, not to convenience.** A population ATE answers a different question than "who should I target."
- **DID's parallel trends and the propensity family's unconfoundedness are different escape hatches.** DID can absorb time-invariant unobserved confounders but needs panel data; PSM/IPW/AIPW only adjust for *observed* confounders but work on a single snapshot.
- **Predictive accuracy of a propensity or outcome model is not the goal** — covariate balance (PSM/IPW), double robustness (AIPW), or ranking quality / Qini (Uplift) are.
- **On observational data, never build Uplift CATE without first addressing confounding** — heterogeneous effects estimated on biased data are biased per subgroup.
- Always hand off to the specific sub-skill for the detailed, method-correct workflow rather than improvising the estimation here.
