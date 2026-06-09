"""
IPW(逆概率加权 / Inverse Probability Weighting)端到端可运行示例。

本脚本在固定随机种子下生成合成观测数据,真实 ATE 已知,然后:
  1. 用 LogisticRegression 估计倾向得分 e(x) = P(T=1 | X);
  2. 构造 ATE 权重(1/e 与 1/(1-e))以及 stabilized weights(稳定化权重);
  3. 做 positivity 检查 + 极端权重诊断,并按分位(1%/99%)对权重做 truncation;
  4. 计算有效样本量 ESS;
  5. 用 Hajek(自归一化)加权均值估计 ATE,并给 bootstrap 置信区间;
  6. 检查加权后协变量平衡(weighted SMD);
  7. 打印真实 ATE vs 估计 ATE 对比。

依赖约束:仅使用 numpy / pandas / scikit-learn / scipy(无 statsmodels)。
Python 3.9.6 下可直接运行:python3 example.py
"""

import numpy as np
import pandas as pd
from scipy.special import expit  # logistic sigmoid,用于生成倾向
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------------------------- #
# 1. 生成合成观测数据(已知真实 ATE)
# --------------------------------------------------------------------------- #
def generate_data(n=4000, seed=42):
    """生成混淆型观测数据。

    - X1, X2, X3 为协变量(混淆变量,同时影响 T 和 Y);
    - T 的分配概率依赖 X(非随机,故存在混淆);
    - Y 同时依赖 X 和 T,处理的真实因果效应为常数 TRUE_ATE。
    返回 DataFrame 和真实 ATE。
    """
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.binomial(1, 0.5, n).astype(float)

    # 倾向得分:T 依赖 X(确保 0<e(x)<1,保持 positivity)
    # X1、X2 同时正向推高 T 和 Y -> 制造明显的正向混淆偏差
    logit_t = 0.8 * x1 + 0.7 * x2 + 0.5 * x3
    prop_true = expit(logit_t)
    treatment = rng.binomial(1, prop_true).astype(float)

    # 真实处理效应(常数,可加性):这就是我们要还原的目标
    true_ate = 3.0

    # 结果模型:Y 依赖 X(混淆)与 T(因果效应)
    noise = rng.normal(0.0, 1.0, n)
    outcome = 2.0 + 1.5 * x1 + 1.2 * x2 - 0.5 * x3 + true_ate * treatment + noise

    df = pd.DataFrame(
        {"X1": x1, "X2": x2, "X3": x3, "T": treatment, "Y": outcome}
    )
    return df, true_ate


# --------------------------------------------------------------------------- #
# 2. 估计倾向得分
# --------------------------------------------------------------------------- #
def estimate_propensity(df, covariates):
    """用逻辑回归估计倾向得分 e(x) = P(T=1 | X),返回每个样本的概率。"""
    x_mat = df[covariates].values
    t_vec = df["T"].values
    model = LogisticRegression(max_iter=1000)
    model.fit(x_mat, t_vec)
    propensity = model.predict_proba(x_mat)[:, 1]
    return propensity


# --------------------------------------------------------------------------- #
# 3. 构造权重(ATE 权重 + stabilized weights)
# --------------------------------------------------------------------------- #
def compute_ate_weights(treatment, propensity, stabilized=True):
    """构造 ATE 的 IPW 权重。

    - 非稳定化:treated -> 1/e, control -> 1/(1-e);
    - 稳定化(stabilized):treated -> P(T=1)/e, control -> P(T=0)/(1-e),
      用边际处理概率做分子,不改变估计量但显著降低权重方差。
    """
    p_treat = treatment.mean()  # 边际处理概率 P(T=1)
    num = np.where(treatment == 1, p_treat, 1.0 - p_treat) if stabilized else 1.0
    den = np.where(treatment == 1, propensity, 1.0 - propensity)
    return num / den


# --------------------------------------------------------------------------- #
# 4. positivity 检查 + 极端权重诊断 + 权重 truncation
# --------------------------------------------------------------------------- #
def positivity_report(propensity):
    """打印倾向得分分布,检查 positivity(是否接近 0 或 1)。"""
    print("---- Positivity / Overlap 检查 ----")
    print(f"倾向得分范围: [{propensity.min():.4f}, {propensity.max():.4f}]")
    print(
        f"接近 0 (<0.01) 的样本数: {(propensity < 0.01).sum()}; "
        f"接近 1 (>0.99) 的样本数: {(propensity > 0.99).sum()}"
    )


def weight_diagnostics(weights, label):
    """打印权重分布诊断:均值、最大值、最大/均值比、尾部权重占比。"""
    w = np.asarray(weights, dtype=float)
    top5_share = np.sort(w)[-5:].sum() / w.sum()
    print(f"---- 权重诊断 ({label}) ----")
    print(
        f"min={w.min():.3f}, mean={w.mean():.3f}, "
        f"median={np.median(w):.3f}, max={w.max():.3f}"
    )
    print(f"max/mean 比值 = {w.max() / w.mean():.2f}; 前 5 大权重占总权重比 = {top5_share:.3%}")


def truncate_weights(weights, lower_pct=0.5, upper_pct=99.5):
    """对权重按分位(默认 0.5%/99.5%)做 truncation(Winsorize),抑制极端权重。

    截断分位越激进(如 1%/99%)对极端权重抑制越强,但也会引入更多偏差;
    这里取较温和的 0.5%/99.5%,在控制方差与保留尾部信息间取平衡。
    """
    lo = np.percentile(weights, lower_pct)
    hi = np.percentile(weights, upper_pct)
    return np.clip(weights, lo, hi)


# --------------------------------------------------------------------------- #
# 5. 有效样本量 ESS
# --------------------------------------------------------------------------- #
def effective_sample_size(weights):
    """ESS = (sum w)^2 / sum(w^2),衡量权重均匀程度。权重越不均,ESS 越小。"""
    w = np.asarray(weights, dtype=float)
    return (w.sum() ** 2) / np.sum(w ** 2)


# --------------------------------------------------------------------------- #
# 6. 用 Hajek 自归一化加权均值估计 ATE
# --------------------------------------------------------------------------- #
def estimate_ate_hajek(outcome, treatment, weights):
    """Hajek(自归一化)估计:各组用 加权和 / 权重和 得到加权均值再相减。"""
    treated = treatment == 1
    control = ~treated
    mean_y1 = np.sum(weights[treated] * outcome[treated]) / np.sum(weights[treated])
    mean_y0 = np.sum(weights[control] * outcome[control]) / np.sum(weights[control])
    return mean_y1 - mean_y0


def bootstrap_ate_ci(df, covariates, n_boot=300, seed=123, stabilized=True):
    """bootstrap 置信区间:每个重抽样内部重新拟合倾向模型 + 重新加权 + 估计 ATE。"""
    rng = np.random.default_rng(seed)
    n = len(df)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boot = df.iloc[idx].reset_index(drop=True)
        prop = estimate_propensity(boot, covariates)
        w = compute_ate_weights(boot["T"].values, prop, stabilized=stabilized)
        w = truncate_weights(w)
        ate = estimate_ate_hajek(boot["Y"].values, boot["T"].values, w)
        estimates.append(ate)
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return lo, hi


# --------------------------------------------------------------------------- #
# 7. 加权后协变量平衡:weighted SMD
# --------------------------------------------------------------------------- #
def weighted_mean(values, weights):
    return np.sum(weights * values) / np.sum(weights)


def weighted_var(values, weights):
    mean = weighted_mean(values, weights)
    return np.sum(weights * (values - mean) ** 2) / np.sum(weights)


def weighted_smd(df, covariates, weights):
    """计算每个协变量的加权标准化均值差(weighted SMD),目标 |SMD| < 0.1。"""
    treated = df["T"].values == 1
    control = ~treated
    rows = []
    for cov in covariates:
        v = df[cov].values
        m1 = weighted_mean(v[treated], weights[treated])
        m0 = weighted_mean(v[control], weights[control])
        s1 = weighted_var(v[treated], weights[treated])
        s0 = weighted_var(v[control], weights[control])
        pooled_sd = np.sqrt((s1 + s0) / 2.0)
        smd = (m1 - m0) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({"covariate": cov, "weighted_SMD": smd})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    covariates = ["X1", "X2", "X3"]
    df, true_ate = generate_data()

    # 朴素(未调整)差异:作为存在混淆偏差的对照
    naive_diff = df.loc[df["T"] == 1, "Y"].mean() - df.loc[df["T"] == 0, "Y"].mean()
    print(f"样本量: {len(df)}, 处理组占比: {df['T'].mean():.3f}")
    print(f"朴素(未调整)组间均值差: {naive_diff:.4f}  (含混淆偏差)\n")

    # 1) 估计倾向得分
    propensity = estimate_propensity(df, covariates)
    positivity_report(propensity)
    print()

    # 2) 构造稳定化 ATE 权重并诊断
    weights_raw = compute_ate_weights(df["T"].values, propensity, stabilized=True)
    weight_diagnostics(weights_raw, "stabilized, 未截断")

    # 3) 权重 truncation(0.5%/99.5%)抑制极端权重
    weights = truncate_weights(weights_raw, 0.5, 99.5)
    weight_diagnostics(weights, "stabilized, 截断后")
    print()

    # 4) 有效样本量 ESS
    treated = df["T"].values == 1
    print("---- 有效样本量 ESS ----")
    print(f"处理组: 名义 {treated.sum()} -> ESS {effective_sample_size(weights[treated]):.1f}")
    print(f"对照组: 名义 {(~treated).sum()} -> ESS {effective_sample_size(weights[~treated]):.1f}")
    print()

    # 5) 加权后平衡 weighted SMD
    balance = weighted_smd(df, covariates, weights)
    print("---- 加权后协变量平衡 (weighted SMD, 目标 |SMD|<0.1) ----")
    print(balance.to_string(index=False))
    print()

    # 6) Hajek 估计 ATE + bootstrap CI
    est_ate = estimate_ate_hajek(df["Y"].values, df["T"].values, weights)
    ci_lo, ci_hi = bootstrap_ate_ci(df, covariates, n_boot=300, stabilized=True)

    print("======== ATE 估计结果 ========")
    print(f"真实 ATE        : {true_ate:.4f}")
    print(f"朴素差异         : {naive_diff:.4f}")
    print(f"IPW 估计 ATE(Hajek): {est_ate:.4f}")
    print(f"bootstrap 95% CI : [{ci_lo:.4f}, {ci_hi:.4f}]")
    covered = ci_lo <= true_ate <= ci_hi
    print(f"真实 ATE 是否落入 CI: {'是' if covered else '否'}")


if __name__ == "__main__":
    main()
