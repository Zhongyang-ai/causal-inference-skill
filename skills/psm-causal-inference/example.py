"""
PSM(倾向得分匹配)端到端最小示例 / End-to-end minimal PSM example.

本脚本在合成的"观测数据"上演示完整的 PSM 流程:
    1. 生成混淆数据:处理 T 的分配概率依赖协变量 X(制造选择偏倚),
       结果 Y 同时依赖 X 和 T,且我们设定了已知的真实 ATT 以便对照验证。
    2. 用 logistic regression 估计倾向得分 e(X) = P(T=1 | X)。
    3. 做共同支撑 / overlap 检查,打印处理组与对照组的得分重叠区间。
    4. 1:1 最近邻匹配(在 logit(ps) 尺度上),带 caliper = 0.2 * SD(logit(ps))。
    5. 计算匹配前后的 SMD(标准化均差)平衡表并打印。
    6. 估计 ATT(匹配对结果差的均值),并用 bootstrap 给出 95% 置信区间。
    7. 打印"真实 ATT" vs "估计 ATT" 对比。

依赖:仅 numpy / pandas / scikit-learn / scipy(无 statsmodels)。
运行:
    python3 example.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

RANDOM_SEED = 20260609
rng = np.random.default_rng(RANDOM_SEED)


# ----------------------------------------------------------------------------
# 1. 生成合成观测数据
# ----------------------------------------------------------------------------
def generate_data(n=4000):
    """生成带混淆的观测数据。

    设计:
      - 协变量 X1, X2, X3 是处理前特征。
      - 处理 T 的概率依赖 X(倾向高的人更可能被处理 -> 选择偏倚)。
      - 结果 Y 依赖 X 和 T;处理效应是恒定的 true_effect。
        因为处理效应对每个人相同,所以真实 ATT == 真实 ATE == true_effect,
        便于直接和估计值对比。
    返回 (DataFrame, true_att)。
    """
    true_effect = 3.0  # 已知真实(恒定)处理效应,即真实 ATT

    # 三个处理前协变量
    x1 = rng.normal(0.0, 1.0, size=n)          # 连续
    x2 = rng.normal(0.0, 1.0, size=n)          # 连续
    x3 = rng.binomial(1, 0.5, size=n).astype(float)  # 二元

    # 处理分配:logit 依赖 X,制造混淆(X 越大越可能被处理)
    logit_t = -0.3 + 1.0 * x1 + 0.8 * x2 + 0.5 * x3
    p_treat = 1.0 / (1.0 + np.exp(-logit_t))
    t = rng.binomial(1, p_treat).astype(int)

    # 结果:同时依赖 X(混淆)和 T(因果效应)+ 噪声
    noise = rng.normal(0.0, 1.0, size=n)
    y = 1.0 + 2.0 * x1 + 1.5 * x2 + 1.0 * x3 + true_effect * t + noise

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "T": t, "Y": y})
    return df, true_effect


# ----------------------------------------------------------------------------
# 2. 估计倾向得分
# ----------------------------------------------------------------------------
def estimate_propensity(df, covariates):
    """用 logistic regression 估计倾向得分 e(X)=P(T=1|X),返回得分数组。"""
    x_mat = df[covariates].to_numpy()
    y_t = df["T"].to_numpy()
    model = LogisticRegression(max_iter=1000)
    model.fit(x_mat, y_t)
    # predict_proba 返回每行 [P(T=0), P(T=1)],取第 1 列为倾向得分
    ps = model.predict_proba(x_mat)[:, 1]
    return ps


def logit(p, eps=1e-6):
    """logit 变换,做数值裁剪避免 0/1 处溢出。"""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------
# 3. 共同支撑 / overlap 检查
# ----------------------------------------------------------------------------
def check_overlap(df):
    """打印处理组与对照组倾向得分的重叠区间。"""
    ps_treat = df.loc[df["T"] == 1, "ps"]
    ps_ctrl = df.loc[df["T"] == 0, "ps"]
    t_lo, t_hi = ps_treat.min(), ps_treat.max()
    c_lo, c_hi = ps_ctrl.min(), ps_ctrl.max()
    overlap_lo = max(t_lo, c_lo)
    overlap_hi = min(t_hi, c_hi)
    print("=== 共同支撑 / Overlap 检查 ===")
    print(f"处理组 ps 范围 : [{t_lo:.4f}, {t_hi:.4f}]  n={len(ps_treat)}")
    print(f"对照组 ps 范围 : [{c_lo:.4f}, {c_hi:.4f}]  n={len(ps_ctrl)}")
    print(f"共同支撑区间   : [{overlap_lo:.4f}, {overlap_hi:.4f}]")
    if overlap_hi <= overlap_lo:
        print("警告:两组得分几乎无重叠,PSM 不适用!")
    else:
        print("重叠充分,可继续匹配。")
    print()
    return overlap_lo, overlap_hi


# ----------------------------------------------------------------------------
# 4. 1:1 最近邻匹配(带 caliper)
# ----------------------------------------------------------------------------
def match_1to1(df, caliper_factor=0.2):
    """在 logit(ps) 尺度上做 1:1 最近邻匹配(无放回),带 caliper。

    返回:匹配后的 DataFrame(只含成功配对的处理组及其匹配对照),
          每个处理组行带上其匹配到的对照组索引 matched_ctrl_idx。
    """
    df = df.copy()
    df["lps"] = logit(df["ps"].to_numpy())  # logit 尺度的倾向得分

    # caliper = 0.2 * 全样本 logit(ps) 的标准差
    caliper = caliper_factor * np.std(df["lps"].to_numpy(), ddof=1)

    treat = df[df["T"] == 1].copy()
    ctrl = df[df["T"] == 0].copy()

    ctrl_lps = ctrl["lps"].to_numpy().reshape(-1, 1)
    ctrl_orig_idx = ctrl.index.to_numpy()

    # 在对照组上建最近邻索引;查询每个处理组个体的最近若干个对照
    n_neighbors = min(10, len(ctrl))  # 查多个候选以支持无放回贪心匹配
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(ctrl_lps)

    treat_lps = treat["lps"].to_numpy().reshape(-1, 1)
    distances, neighbor_pos = nn.kneighbors(treat_lps)

    used_ctrl = set()        # 已被占用的对照组原始索引(无放回)
    matched_treat_idx = []
    matched_ctrl_idx = []

    treat_orig_idx = treat.index.to_numpy()
    for i in range(len(treat)):
        for j in range(n_neighbors):
            dist = distances[i, j]
            cand_orig = ctrl_orig_idx[neighbor_pos[i, j]]
            if cand_orig in used_ctrl:
                continue          # 该对照已被占用,看下一个候选
            if dist > caliper:
                break             # 候选按距离升序,最近的都超 caliper,放弃此处理个体
            used_ctrl.add(cand_orig)
            matched_treat_idx.append(treat_orig_idx[i])
            matched_ctrl_idx.append(cand_orig)
            break

    n_treat_total = len(treat)
    n_matched = len(matched_treat_idx)
    print("=== 1:1 最近邻匹配(带 caliper)===")
    print(f"caliper = 0.2 * SD(logit(ps)) = {caliper:.4f}")
    print(f"处理组总数 : {n_treat_total}")
    print(f"成功匹配数 : {n_matched}")
    print(f"丢弃处理组 : {n_treat_total - n_matched} "
          f"({100.0 * (n_treat_total - n_matched) / n_treat_total:.1f}%)")
    print()

    return (df, np.array(matched_treat_idx), np.array(matched_ctrl_idx), caliper)


# ----------------------------------------------------------------------------
# 5. SMD 平衡表(匹配前 vs 匹配后)
# ----------------------------------------------------------------------------
def smd(treat_vals, ctrl_vals):
    """标准化均差:均值差 / 两组方差均值的平方根。"""
    m_t = np.mean(treat_vals)
    m_c = np.mean(ctrl_vals)
    v_t = np.var(treat_vals, ddof=1)
    v_c = np.var(ctrl_vals, ddof=1)
    pooled_sd = np.sqrt((v_t + v_c) / 2.0)
    if pooled_sd == 0:
        return 0.0
    return (m_t - m_c) / pooled_sd


def balance_table(df, matched_treat_idx, matched_ctrl_idx, covariates):
    """打印匹配前后各协变量的 SMD。"""
    rows = []
    # 匹配前:全样本处理组 vs 全样本对照组
    pre_treat = df[df["T"] == 1]
    pre_ctrl = df[df["T"] == 0]
    # 匹配后:仅成功配对的处理组与其匹配对照
    post_treat = df.loc[matched_treat_idx]
    post_ctrl = df.loc[matched_ctrl_idx]

    for cov in covariates:
        smd_before = smd(pre_treat[cov].to_numpy(), pre_ctrl[cov].to_numpy())
        smd_after = smd(post_treat[cov].to_numpy(), post_ctrl[cov].to_numpy())
        rows.append({
            "covariate": cov,
            "SMD_before": round(smd_before, 4),
            "SMD_after": round(smd_after, 4),
            "|SMD_after|<0.1": "OK" if abs(smd_after) < 0.1 else "FAIL",
        })

    table = pd.DataFrame(rows)
    print("=== 协变量平衡表(SMD,目标 |SMD_after| < 0.1)===")
    print(table.to_string(index=False))
    print()
    return table


# ----------------------------------------------------------------------------
# 6. 估计 ATT + bootstrap 置信区间
# ----------------------------------------------------------------------------
def estimate_att(df, matched_treat_idx, matched_ctrl_idx, n_boot=1000):
    """ATT = 匹配对结果差的均值;bootstrap 在"匹配对"层面重采样得 95% CI。"""
    y_treat = df.loc[matched_treat_idx, "Y"].to_numpy()
    y_ctrl = df.loc[matched_ctrl_idx, "Y"].to_numpy()
    pair_diff = y_treat - y_ctrl
    att = np.mean(pair_diff)

    n_pairs = len(pair_diff)
    boot_atts = np.empty(n_boot)
    for b in range(n_boot):
        sample_idx = rng.integers(0, n_pairs, size=n_pairs)  # 有放回重采样匹配对
        boot_atts[b] = np.mean(pair_diff[sample_idx])

    ci_low, ci_high = np.percentile(boot_atts, [2.5, 97.5])
    se = np.std(boot_atts, ddof=1)
    return att, se, ci_low, ci_high


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    covariates = ["x1", "x2", "x3"]

    # 1. 数据
    df, true_att = generate_data(n=4000)
    print(f"样本量 n = {len(df)};处理组 = {int(df['T'].sum())};"
          f"对照组 = {int((df['T'] == 0).sum())}")
    print(f"已知真实 ATT(= 真实处理效应)= {true_att:.4f}")
    # 朴素对比(不做任何校正,会被混淆偏倚污染)
    naive = df.loc[df["T"] == 1, "Y"].mean() - df.loc[df["T"] == 0, "Y"].mean()
    print(f"朴素差(未校正,有偏)      = {naive:.4f}")
    print()

    # 2. 倾向得分
    df["ps"] = estimate_propensity(df, covariates)

    # 3. overlap
    check_overlap(df)

    # 4. 匹配
    df, matched_treat_idx, matched_ctrl_idx, _ = match_1to1(df)

    # 5. 平衡表
    balance_table(df, matched_treat_idx, matched_ctrl_idx, covariates)

    # 6. ATT 估计
    att, se, ci_low, ci_high = estimate_att(df, matched_treat_idx, matched_ctrl_idx)

    print("=== ATT 估计结果 ===")
    print(f"估计 ATT             = {att:.4f}")
    print(f"bootstrap 标准误     = {se:.4f}")
    print(f"95% 置信区间         = [{ci_low:.4f}, {ci_high:.4f}]")
    print()
    print("=== 对比 ===")
    print(f"真实 ATT  = {true_att:.4f}")
    print(f"估计 ATT  = {att:.4f}  (偏差 {att - true_att:+.4f})")
    print(f"朴素估计  = {naive:.4f}  (偏差 {naive - true_att:+.4f}，因混淆被高估)")
    covered = ci_low <= true_att <= ci_high
    print(f"真实 ATT 是否落在 95% CI 内: {'是' if covered else '否'}")


if __name__ == "__main__":
    main()
