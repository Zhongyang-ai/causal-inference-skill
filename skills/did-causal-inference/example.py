"""
DID(双重差分,Difference-in-Differences)最小可跑示例。

本脚本:
1. 固定随机种子,生成合成面板数据:多个个体、两期(pre/post)、treated/control 两组,
   outcome 满足平行趋势,并注入一个已知的真实处理效应(treatment 只在 post 期对 treated 组生效)。
2. 用四个组合均值手算 2x2 DID 估计量。
3. 用 numpy 最小二乘手写 OLS,估计交互项回归
   Y = b0 + b1*Treat + b2*Post + b3*(Treat*Post)+ e,
   输出 b3 及其标准误、t 值、95% 置信区间。
4. 验证两种方法的 b3 一致。
5. 打印真实效应 vs 估计效应,并打印 2x2 均值表佐证平行趋势直觉。

依赖:仅 numpy / pandas / scipy(本机 Python 3.9.6,无 statsmodels,故 OLS 全部手写)。
运行:python3 example.py
"""

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------
# 1. 生成合成面板数据
# ----------------------------------------------------------------------
def generate_panel(n_units=400, true_effect=5.0, seed=42):
    """生成两期(pre/post)、treated/control 两组的合成面板数据。

    设计要点(保证识别假设成立):
    - 个体固定效应 alpha_i:制造组间/个体间「不随时间变化」的水平差异
      (DID 通过差分会自动吸收掉,因此不会污染效应估计)。
    - 共同时间趋势 time_trend:pre -> post 两组都加上同样的趋势 → 满足【平行趋势】。
    - 真实处理效应 true_effect:只在 (treated=1 且 post=1) 这一格生效,即 ATT。

    返回:长表(long format),每个个体两行(pre 一行、post 一行)。
    """
    rng = np.random.default_rng(seed)

    # 一半个体为处理组,一半为对照组
    unit_id = np.arange(n_units)
    treat = (unit_id >= n_units // 2).astype(int)  # 后一半为 treated

    # 个体固定效应:处理组基线整体偏高(制造一个事前的固定水平差,考验 DID 能否吸收)
    alpha = rng.normal(loc=10.0 + 3.0 * treat, scale=2.0, size=n_units)

    common_time_trend = 4.0  # pre -> post 两组共同的时间增量(平行趋势的来源)

    rows = []
    for i in range(n_units):
        for post in (0, 1):
            noise = rng.normal(0.0, 1.0)
            # 基础结果 = 个体固定效应 + 共同时间趋势*post + 噪声
            y = alpha[i] + common_time_trend * post + noise
            # 真实处理效应:仅在 treated 组的 post 期叠加
            y += true_effect * (treat[i] * post)
            rows.append({"unit": i, "treat": treat[i], "post": post, "y": y})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. 2x2 DID:四个组合均值
# ----------------------------------------------------------------------
def did_2x2(df):
    """用四个组合的均值手算 2x2 DID 估计量。

    DID = (Ybar_T_post - Ybar_T_pre) - (Ybar_C_post - Ybar_C_pre)
    """
    means = df.groupby(["treat", "post"])["y"].mean()
    yt_pre = means.loc[(1, 0)]   # 处理组 pre
    yt_post = means.loc[(1, 1)]  # 处理组 post
    yc_pre = means.loc[(0, 0)]   # 对照组 pre
    yc_post = means.loc[(0, 1)]  # 对照组 post

    did = (yt_post - yt_pre) - (yc_post - yc_pre)

    # 整理成均值表(行=组,列=期),便于直觉佐证平行趋势
    table = means.unstack("post")
    table.columns = ["pre", "post"]
    table.index = ["control", "treated"]
    table["post-pre (组内前后差)"] = table["post"] - table["pre"]
    return did, table


# ----------------------------------------------------------------------
# 3. numpy 手写 OLS(普通 OLS 标准误)
# ----------------------------------------------------------------------
def ols(X, y):
    """普通最小二乘:返回系数、标准误、t 值、双侧 p 值。

    用 np.linalg.lstsq 解 beta;标准误用经典齐方差 OLS 公式
    Var(beta) = sigma^2 * (X'X)^{-1},sigma^2 = RSS / (n - k)。
    """
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ beta
    rss = float(resid @ resid)
    dof = n - k                       # 自由度
    sigma2 = rss / dof                # 残差方差的无偏估计
    XtX_inv = np.linalg.inv(X.T @ X)
    var_beta = sigma2 * XtX_inv       # 系数协方差矩阵
    se = np.sqrt(np.diag(var_beta))   # 各系数标准误

    t_stat = beta / se
    p_val = 2.0 * stats.t.sf(np.abs(t_stat), df=dof)
    return beta, se, t_stat, p_val, dof


def did_regression(df):
    """估计交互项回归 Y = b0 + b1*Treat + b2*Post + b3*(Treat*Post)+ e。

    返回 b3(DID/ATT 估计)、其标准误、t 值、95% 置信区间。
    """
    treat = df["treat"].to_numpy(dtype=float)
    post = df["post"].to_numpy(dtype=float)
    inter = treat * post
    y = df["y"].to_numpy(dtype=float)

    # 设计矩阵列顺序:[截距, Treat, Post, Treat*Post]
    X = np.column_stack([np.ones_like(treat), treat, post, inter])

    beta, se, t_stat, p_val, dof = ols(X, y)

    idx = 3  # Treat*Post 即 b3
    b3, se3, t3, p3 = beta[idx], se[idx], t_stat[idx], p_val[idx]
    tcrit = stats.t.ppf(0.975, df=dof)  # 95% CI 临界值
    ci_low = b3 - tcrit * se3
    ci_high = b3 + tcrit * se3

    return {
        "beta": beta,
        "b3": b3,
        "se3": se3,
        "t3": t3,
        "p3": p3,
        "ci": (ci_low, ci_high),
        "dof": dof,
    }


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    true_effect = 5.0
    df = generate_panel(n_units=400, true_effect=true_effect, seed=42)

    # --- 2x2 DID ---
    did_means, table = did_2x2(df)

    # --- 回归 DID ---
    reg = did_regression(df)
    b3 = reg["b3"]

    print("=" * 64)
    print("均值表(佐证平行趋势直觉)")
    print("=" * 64)
    print(table.round(3).to_string())
    print()
    print("直觉:control 与 treated 的 [post-pre] 组内前后差,")
    print("两者之差即为 DID;若无处理效应,两组前后差应接近(平行趋势)。")
    print()

    print("=" * 64)
    print("DID 估计结果")
    print("=" * 64)
    print(f"真实处理效应 (true ATT)            : {true_effect:.4f}")
    print(f"2x2 均值法 DID 估计               : {did_means:.4f}")
    print(f"回归交互项 b3 估计 (DID/ATT)      : {b3:.4f}")
    print()
    print(f"b3 标准误 (普通 OLS, 齐方差)      : {reg['se3']:.4f}")
    print(f"b3 t 值                          : {reg['t3']:.4f}")
    print(f"b3 双侧 p 值                     : {reg['p3']:.3e}")
    print(f"b3 95% 置信区间                  : [{reg['ci'][0]:.4f}, {reg['ci'][1]:.4f}]")
    print(f"  (标准误为普通 OLS 齐方差标准误;真实研究中建议在处理分配层级聚类)")
    print()

    # --- 一致性校验 ---
    assert np.isclose(did_means, b3, atol=1e-8), "2x2 与回归的 b3 不一致!"
    print(f"一致性校验通过:|2x2 - 回归 b3| = {abs(did_means - b3):.2e} (< 1e-8)")

    # --- 是否接近真实效应 ---
    err = abs(b3 - true_effect)
    covered = reg["ci"][0] <= true_effect <= reg["ci"][1]
    print(f"估计误差 |b3 - true| = {err:.4f};真实效应是否落入 95% CI: {covered}")
    print()
    if err < 1.0:
        print("结论:估计效应接近真实效应,DID 识别成功。")
    else:
        print("警告:估计与真实效应偏差较大,请检查数据生成或假设。")


if __name__ == "__main__":
    main()
