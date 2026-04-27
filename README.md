# PSM Causal Inference Skill

A Codex skill for propensity score methods in causal inference, focused on practical analytics workflows such as marketing incrementality, treatment-effect estimation, covariate balance checks, and business-facing reporting.

## What It Covers

- Propensity score matching (PSM)
- Inverse probability weighting (IPW)
- ATT / ATE framing
- Common support and overlap checks
- Caliper and nearest-neighbor matching
- Standardized mean difference (SMD) balance diagnostics
- Sensitivity analysis and reporting guardrails

## Installation

Copy the skill folder into your Codex skills directory:

```bash
mkdir -p ~/.codex/skills
cp -R skills/psm-causal-inference ~/.codex/skills/psm-causal-inference
```

Restart Codex after installing.

## Example Prompts

```text
Use the psm-causal-inference skill to evaluate whether Google remarketing increased first-deposit conversion.
```

```text
用 psm-causal-inference skill，帮我做一个倾向得分匹配分析，比较被投放和未投放客户的转化差异。
```

```text
Run a PSM analysis with SMD balance diagnostics and estimate ATT for this treatment.
```

## Skill Layout

```text
skills/
└── psm-causal-inference/
    └── SKILL.md
```

