# Causal Inference Skills

一套面向实战分析的**因果推断技能集**，覆盖营销增量、处理效应估计、政策评估、人群定向等场景。包含一个**自动选方法的路由入口**和五个独立可调用的方法子技能，每个方法都配有中文讲解文档和可直接运行的 Python 示例。

> 本仓库由原 `psm-causal-inference-skill` 扩展而来，现已升级为多方法的 `causal-inference-skill`。原 PSM 技能完整保留。

## 技能总览

| 技能 | 方法 | 估计目标 | 适用数据 | 一句话定位 |
| --- | --- | --- | --- | --- |
| **`causal-inference`** | 🧭 方法路由入口 | — | — | 根据你的数据和业务目标，自动推荐用哪种方法 |
| `psm-causal-inference` | PSM 倾向得分匹配 | ATT / ATE | 横截面观测数据 + 可观测混淆 | 给处理组找"双胞胎"对照后比差异 |
| `ipw-causal-inference` | IPW 逆概率加权 | ATE / ATT | 横截面观测数据 + 重叠良好 | 用倾向得分倒数加权构造伪总体 |
| `aipw-causal-inference` | AIPW 双稳健 | ATE / ATT | 观测数据 + 想抗模型误设 | IPW + 结果回归，二者有一个对就行 |
| `did-causal-inference` | DID 双重差分 | ATT | 面板/重复截面，有前后期 + 对照组 | 处理组前后差 − 对照组前后差 |
| `uplift-modeling` | Uplift 增益建模 | 个体/子群 CATE | 处理+结果，最好随机实验 | 估个体异质效应，回答"该对谁投放" |

## 如何选方法（决策速查）

```
你要的是"平均效应"还是"对谁有效"？
├─ 对谁有效 / 该投放谁 ──────────────────────────→ Uplift Modeling
│                                                  (观测数据先用 PSM/IPW/AIPW 去混淆)
└─ 平均效应(ATE/ATT)
   │
   有没有"前后两期 + 对照组"(面板)？
   ├─ 有 ──────────────────────────────────────→ DID（能吸收不随时间变的不可观测混淆）
   └─ 没有（横截面观测快照）
      ├─ 要可解释的匹配队列 / ATT ──────────────→ PSM
      ├─ 要全样本估总体 ATE，重叠良好 ──────────→ IPW
      └─ 不确定哪个模型对 / 想更稳健 ───────────→ AIPW（双稳健，推荐默认）
```

不想自己判断？**直接调用 `causal-inference` 路由技能**，描述你的数据和目标，它会帮你诊断并推荐。

## 安装

每个子技能都是自包含、可独立调用的。把需要的技能文件夹拷进你的 skills 目录即可。

### Codex

```bash
mkdir -p ~/.codex/skills
# 安装全部
cp -R skills/* ~/.codex/skills/
# 或只装路由入口 + 某个方法
cp -R skills/causal-inference        ~/.codex/skills/
cp -R skills/psm-causal-inference    ~/.codex/skills/
```

### Claude Code / Claude Agent SDK

```bash
mkdir -p ~/.claude/skills
cp -R skills/* ~/.claude/skills/
```

安装后重启你的 agent 客户端。

## 每个方法包含什么

```text
skills/<method>/
├── SKILL.md         # 工作流指导（英文，触发场景 + Workflow + Checklist + Guardrails）
├── EXPLANATION.md   # 原理与适用场景讲解（中文 + 英文术语）
└── example.py       # 可直接运行的示例（含合成数据，验证估计值≈真实值）
```

`example.py` 仅依赖 `numpy / pandas / scikit-learn / scipy`，无需 statsmodels 等额外库，开箱即跑：

```bash
python3 skills/psm-causal-inference/example.py
python3 skills/did-causal-inference/example.py
python3 skills/ipw-causal-inference/example.py
python3 skills/aipw-causal-inference/example.py
python3 skills/uplift-modeling/example.py
```

每个示例都会打印"真实效应 vs 估计效应"的对比，用于验证方法实现的正确性（如 AIPW 示例会演示在模型误设时的双稳健性，Uplift 示例会验证高增益人群的真实 CATE 确实更高）。

## 调用示例

```text
# 让路由器帮我选方法
用 causal-inference skill：我有一份用户横截面数据，想评估"是否推送过优惠券"对复购的因果影响，该用什么方法？

# 直接指定方法
用 did-causal-inference skill，分析某功能上线前后处理组与对照组的留存差异，并检验平行趋势。

用 uplift-modeling skill，基于这次 A/B 实验数据，找出最该投放的人群。

用 aipw-causal-inference skill，用双稳健估计这个投放的 ATE，并做 cross-fitting。
```

## 仓库结构

```text
.
├── README.md
└── skills/
    ├── causal-inference/        # 🧭 路由入口（SKILL.md + EXPLANATION.md）
    ├── psm-causal-inference/
    ├── ipw-causal-inference/
    ├── aipw-causal-inference/
    ├── did-causal-inference/
    └── uplift-modeling/
```

## 方法间的关系

- **PSM / IPW / AIPW** 都基于倾向得分、校正**可观测**混淆，适合横截面观测数据：PSM 匹配丢样本、IPW 全样本加权估 ATE、AIPW 融合两者且双稳健。
- **DID** 走另一条路，靠前后两期差分吸收**不随时间变化的不可观测**混淆，但需要面板数据和平行趋势假设。
- **Uplift** 估的是**个体/子群异质效应(CATE)**而非平均效应，用于"对谁投放"；在观测数据上应先用前几种方法去混淆。

详见各方法的 `EXPLANATION.md`。
