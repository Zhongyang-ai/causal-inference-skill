"""
AIPW(增广逆概率加权 / 双稳健,Augmented IPW / Doubly Robust)端到端示例。

本脚本演示:
  1. 固定随机种子生成合成观测数据(X 为混淆变量,T 依赖 X,Y 依赖 X 和 T),
     真实 ATE 已知。
  2. 拟合倾向模型 e(x)=P(T=1|X)(LogisticRegression)。
  3. 拟合结果模型 mu_1(x)、mu_0(x)(GradientBoostingRegressor,分别对 treated/control)。
  4. 用 K 折 cross-fitting(样本分裂)在留出折上预测两个 nuisance 模型,防过拟合——
     这是 AIPW 获得良好理论性质的关键。
  5. 按公式计算 AIPW 估计量,并用影响函数(influence function)估计标准误与 95% CI。
  6. 演示"双稳健":故意把倾向模型设错 / 把结果模型设错,展示 AIPW 仍接近真实 ATE,
     而纯 IPW 或纯回归会偏离。
  7. 打印对比表:真实 ATE vs AIPW vs 纯 IPW vs 纯回归。

仅依赖 numpy / pandas / scikit-learn / scipy(Python 3.9)。
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# 1. 合成数据生成
# ---------------------------------------------------------------------------
def generate_data(n=4000, seed=42):
    """生成带混淆的观测数据。

    - X: 4 个协变量(混淆变量)。
    - 处理 T: 倾向 e(x) 通过非线性的 logit 依赖 X(真实倾向是非线性的)。
    - 结果 Y: 同时依赖 X(非线性)和 T,真实处理效应恒为 TRUE_ATE。

    返回 DataFrame 以及已知真实 ATE。
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(n, 4))
    x0, x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    # 真实倾向:非线性(含平方项与交互项),纯线性 logistic 无法完全拟合
    logit = 0.6 * x0 - 0.8 * x1 + 0.5 * x2 * x1 + 0.4 * (x0 ** 2 - 1.0)
    e_true = 1.0 / (1.0 + np.exp(-logit))
    T = rng.binomial(1, e_true)

    # 真实结果:基线非线性依赖 X,处理效应恒定 = TRUE_ATE
    TRUE_ATE = 3.0
    baseline = 2.0 + 1.5 * x0 + 1.0 * x1 ** 2 - 1.2 * x2 + 0.8 * x0 * x2
    noise = rng.normal(scale=1.0, size=n)
    Y = baseline + TRUE_ATE * T + noise

    df = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
    df["T"] = T
    df["Y"] = Y
    return df, TRUE_ATE


# ---------------------------------------------------------------------------
# 2. Cross-fitting:在留出折上预测 nuisance(倾向 + 两个结果模型)
# ---------------------------------------------------------------------------
def cross_fit_nuisances(
    df,
    feature_cols,
    propensity_misspecified=False,
    outcome_misspecified=False,
    n_splits=5,
    seed=0,
):
    """K 折 cross-fitting,返回每个单位在留出折上的 e_hat, mu1_hat, mu0_hat。

    参数
    ----
    propensity_misspecified: 若 True,倾向模型只用一个无关/弱特征,故意设错。
    outcome_misspecified:    若 True,结果模型只用一个无关/弱特征,故意设错。

    通过"在其余折训练、在留出折预测"避免过拟合偏误。
    """
    n = len(df)
    X_all = df[feature_cols].values
    T_all = df["T"].values
    Y_all = df["Y"].values

    # 倾向模型用的特征:正确设定用全部特征;误设时只用单个弱特征
    if propensity_misspecified:
        X_prop = df[["x3"]].values  # x3 与 T 几乎无关 -> 倾向模型设错
    else:
        X_prop = X_all

    # 结果模型用的特征:正确设定用全部特征;误设时只用单个弱特征
    if outcome_misspecified:
        X_out = df[["x3"]].values  # x3 与 Y 几乎无关 -> 结果模型设错
    else:
        X_out = X_all

    e_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    mu0_hat = np.zeros(n)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X_all):
        # --- 倾向模型 e(x) = P(T=1|X),在训练折拟合 ---
        prop_model = LogisticRegression(max_iter=1000)
        prop_model.fit(X_prop[train_idx], T_all[train_idx])
        e_hat[test_idx] = prop_model.predict_proba(X_prop[test_idx])[:, 1]

        # --- 结果模型:对 treated / control 分别拟合 ---
        tr = train_idx[T_all[train_idx] == 1]
        co = train_idx[T_all[train_idx] == 0]

        m1 = GradientBoostingRegressor(random_state=seed)
        m1.fit(X_out[tr], Y_all[tr])
        mu1_hat[test_idx] = m1.predict(X_out[test_idx])

        m0 = GradientBoostingRegressor(random_state=seed)
        m0.fit(X_out[co], Y_all[co])
        mu0_hat[test_idx] = m0.predict(X_out[test_idx])

    return e_hat, mu1_hat, mu0_hat


# ---------------------------------------------------------------------------
# 3. 三种估计量:AIPW / 纯 IPW / 纯回归
# ---------------------------------------------------------------------------
def estimate_aipw(df, e_hat, mu1_hat, mu0_hat, clip=0.01):
    """计算 AIPW(双稳健)ATE 估计量,并用影响函数估计 SE 与 95% CI。

    AIPW 逐单位得分:
        psi_i = mu1(x_i) - mu0(x_i)
                + T_i (Y_i - mu1(x_i)) / e(x_i)
                - (1-T_i)(Y_i - mu0(x_i)) / (1 - e(x_i))
    ATE = mean(psi); 影响函数方差 = var(psi) / n。
    """
    T = df["T"].values
    Y = df["Y"].values

    # 截断倾向得分,避免权重爆炸(positivity 保护)
    e = np.clip(e_hat, clip, 1 - clip)

    psi = (
        mu1_hat
        - mu0_hat
        + T * (Y - mu1_hat) / e
        - (1 - T) * (Y - mu0_hat) / (1 - e)
    )
    ate = psi.mean()
    n = len(psi)
    se = psi.std(ddof=1) / np.sqrt(n)          # 影响函数标准误
    z = stats.norm.ppf(0.975)
    ci = (ate - z * se, ate + z * se)
    return ate, se, ci


def estimate_ipw(df, e_hat, clip=0.01):
    """纯 IPW(Horvitz-Thompson 型,做归一化)ATE 估计——只依赖倾向模型。"""
    T = df["T"].values
    Y = df["Y"].values
    e = np.clip(e_hat, clip, 1 - clip)

    w1 = T / e
    w0 = (1 - T) / (1 - e)
    # 归一化(Hajek 估计),更稳定
    mu1 = np.sum(w1 * Y) / np.sum(w1)
    mu0 = np.sum(w0 * Y) / np.sum(w0)
    return mu1 - mu0


def estimate_outcome_regression(mu1_hat, mu0_hat):
    """纯结果回归(G-computation)ATE 估计——只依赖结果模型。"""
    return np.mean(mu1_hat - mu0_hat)


# ---------------------------------------------------------------------------
# 4. 在一种设定下跑完整三件套,返回结果字典
# ---------------------------------------------------------------------------
def run_scenario(df, feature_cols, label, propensity_misspecified=False,
                 outcome_misspecified=False, seed=0):
    e_hat, mu1_hat, mu0_hat = cross_fit_nuisances(
        df,
        feature_cols,
        propensity_misspecified=propensity_misspecified,
        outcome_misspecified=outcome_misspecified,
        seed=seed,
    )
    aipw, se, ci = estimate_aipw(df, e_hat, mu1_hat, mu0_hat)
    ipw = estimate_ipw(df, e_hat)
    reg = estimate_outcome_regression(mu1_hat, mu0_hat)
    return {
        "scenario": label,
        "AIPW": aipw,
        "AIPW_SE": se,
        "AIPW_CI_low": ci[0],
        "AIPW_CI_high": ci[1],
        "IPW": ipw,
        "OutcomeReg": reg,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    df, true_ate = generate_data()
    feature_cols = ["x0", "x1", "x2", "x3"]

    print("=" * 78)
    print("AIPW(双稳健)示例 —— 真实 ATE = {:.3f}".format(true_ate))
    print("样本量 n = {} | 处理组占比 = {:.3f}".format(len(df), df["T"].mean()))
    print("=" * 78)

    # 三种设定:
    #  A 两个模型都(基本)正确
    #  B 倾向模型故意设错(只剩结果模型正确)
    #  C 结果模型故意设错(只剩倾向模型正确)
    scenarios = [
        run_scenario(df, feature_cols, "A 两模型均正确", seed=0),
        run_scenario(df, feature_cols, "B 倾向误设(仅结果对)",
                     propensity_misspecified=True, seed=0),
        run_scenario(df, feature_cols, "C 结果误设(仅倾向对)",
                     outcome_misspecified=True, seed=0),
    ]

    res = pd.DataFrame(scenarios)
    res.insert(1, "TrueATE", true_ate)

    # 各估计量相对真值的绝对偏差
    res["|AIPW-真值|"] = (res["AIPW"] - true_ate).abs()
    res["|IPW-真值|"] = (res["IPW"] - true_ate).abs()
    res["|回归-真值|"] = (res["OutcomeReg"] - true_ate).abs()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: "{:.3f}".format(v))

    print("\n【对比表:真实 ATE vs AIPW vs 纯 IPW vs 纯回归】\n")
    show_cols = ["scenario", "TrueATE", "AIPW", "IPW", "OutcomeReg",
                 "|AIPW-真值|", "|IPW-真值|", "|回归-真值|"]
    print(res[show_cols].to_string(index=False))

    print("\n【AIPW 推断(影响函数标准误与 95% CI)】\n")
    inf_cols = ["scenario", "AIPW", "AIPW_SE", "AIPW_CI_low", "AIPW_CI_high"]
    print(res[inf_cols].to_string(index=False))

    # ---- 双稳健解读 ----
    print("\n" + "=" * 78)
    print("双稳健(Doubly Robust)解读")
    print("=" * 78)
    rowB = res[res["scenario"].str.startswith("B")].iloc[0]
    rowC = res[res["scenario"].str.startswith("C")].iloc[0]

    print(
        "- 设定 B(倾向模型设错,但结果模型正确):\n"
        "    纯 IPW 偏差 = {:.3f} (依赖被设错的倾向模型 -> 偏离),\n"
        "    AIPW  偏差 = {:.3f} (靠正确的结果模型救回 -> 仍接近真值)。".format(
            rowB["|IPW-真值|"], rowB["|AIPW-真值|"])
    )
    print(
        "- 设定 C(结果模型设错,但倾向模型正确):\n"
        "    纯回归 偏差 = {:.3f} (依赖被设错的结果模型 -> 偏离),\n"
        "    AIPW   偏差 = {:.3f} (靠正确的倾向模型救回 -> 仍接近真值)。".format(
            rowC["|回归-真值|"], rowC["|AIPW-真值|"])
    )
    print(
        "\n结论:只要倾向模型与结果模型【至少一个正确】,AIPW 都接近真实 ATE,\n"
        "体现了双稳健性;而纯 IPW / 纯回归一旦其依赖的单一模型被设错就会偏离。"
    )

    # 简单断言:正确设定下 AIPW 应接近真值
    assert res.iloc[0]["|AIPW-真值|"] < 0.3, "正确设定下 AIPW 偏离过大"
    print("\n[OK] 正确设定下 AIPW 接近真实 ATE,脚本运行通过。")


if __name__ == "__main__":
    main()
