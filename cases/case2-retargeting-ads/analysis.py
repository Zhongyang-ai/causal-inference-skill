"""
案例 2:券商 Retargeting 广告因果效应评估(观测数据去混淆 + Uplift Targeting)

【业务背景】
某券商线上广告团队开展 retargeting 广告投放,定向向"注册未入金"客户投放广告,
目标是提升"注册→首次入金"的转化率。由于广告系统按竞价、受众匹配和活跃度自动
决定投放对象,无法做 A/B 实验,只能用观测数据评估因果效应。

【核心挑战:活跃度混淆(ad-targeting confounding)】
- 广告系统只能触达近期回访、停留时长高、浏览过入金页的高活跃用户;
- 这些高活跃用户本来就更可能入金(选择偏差);
- 朴素的"被投放 vs 未投放"转化率差会严重高估广告的真实因果效应。

【Treatment / Outcome 定义】
- Treatment T = 是否被投放 retargeting 广告(二元)
- Outcome Y = 是否完成首次入金(二元)
- Confounders X = 近 30 日回访次数、平均停留时长、是否浏览过入金页、注册天数、
                  设备、地域、过往点击 push 记录、账户资产意向分等

【方法学设计】
1. 平均效应估计(ATE):
   - 主用 AIPW(双稳健估计,抗活跃度混淆,即使一个模型误设仍稳健)
   - 对照用 IPW(Hajek 归一化 + truncation + ESS 诊断)
   - 对比朴素估计(未调整)vs 真实 ATE(模拟已知)vs IPW vs AIPW

2. Uplift Modeling(targeting 策略,回答"该向谁投放"):
   - 关键:这是观测数据,必须先去混淆(用 IPW 权重训练或在匹配后样本上建模),
     否则 uplift 有偏(high-activity 用户既更可能被投,也更可能入金,直接建模会
     错把混淆当作异质效应)
   - 用 T-learner / S-learner(GradientBoosting)估每人 CATE
   - 用 Qini 曲线 + AUUC 评估排序能力
   - 按预测 uplift 分桶,验证桶内真实 CATE 单调
   - 识别 persuadables(高 uplift,该投)/ sure things(无增益)/ lost causes(无增益)/
     sleeping dogs(负增益,投了反而流失)

3. Targeting 策略建议:
   - 若只对 Top-X% 高 uplift 用户投放,相比全量投放能多带来多少入金增量

【依赖】
- 仅用 numpy / pandas / scikit-learn / scipy(Python 3.9.6,无 statsmodels)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import trapezoid
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

SEED = 42


# ===========================================================================
# 1. 生成合成观测数据(活跃度混淆 + 异质处理效应)
# ===========================================================================
def generate_retargeting_data(n=12000, seed=SEED):
    """生成券商 retargeting 广告观测数据(非随机分配,存在活跃度混淆)。

    协变量(pre-treatment):
    - visits_30d     : 近 30 日回访次数(活跃度核心指标,强混淆)
    - avg_duration   : 平均停留时长(分钟,强混淆)
    - viewed_deposit : 是否浏览过入金页(0/1,强混淆)
    - days_since_reg : 注册天数
    - device_mobile  : 是否移动端注册(0/1)
    - region_tier    : 地域层级(1=一线,2=二线,3=下沉,弱影响)
    - clicked_push   : 过往是否点击过 push(0/1)
    - intent_score   : 账户资产意向分(0-10,虚拟综合分,衡量入金潜力)

    Treatment T(被投放 retargeting 广告):
    - 倾向得分 e(x) 强依赖活跃度指标(visits/duration/viewed_deposit),
      这些用户更容易被广告系统触达 -> 制造活跃度混淆

    Outcome Y(完成首次入金):
    - 基线入金概率依赖协变量(活跃度高的本来就更可能入金)
    - 真实处理效应 tau(x) 随协变量异质:
      * Persuadables(中高意向 + 看过入金页但还在犹豫):真实 uplift 高(+10~15pp)
      * Sure things(超高意向,已下定决心):无增益(~0pp,treat 不 treat 都会入金)
      * Lost causes(低意向):无增益(~0pp,怎么投也不入)
      * Sleeping dogs(一小撮特殊群体,被打扰反感):负增益(-5pp)

    返回 DataFrame + 真实 ATE。
    """
    rng = np.random.default_rng(seed)

    # --- 协变量:贴合券商场景 ---
    visits_30d = rng.poisson(lam=5, size=n).astype(float)       # 回访次数,右偏分布
    avg_duration = rng.gamma(shape=3, scale=2, size=n)          # 停留时长(分钟),右偏
    viewed_deposit = rng.binomial(1, 0.35, n).astype(float)     # 35% 看过入金页
    days_since_reg = rng.uniform(7, 120, n)                     # 注册 7~120 天
    device_mobile = rng.binomial(1, 0.65, n).astype(float)      # 65% 移动端
    region_tier = rng.choice([1, 2, 3], size=n, p=[0.3, 0.4, 0.3])  # 地域层级
    clicked_push = rng.binomial(1, 0.2, n).astype(float)        # 20% 点过 push
    # 意向分:独立的潜在意向 + 看过入金页的小幅加成。
    # 关键:意向分**不**由活跃度(visits)决定 —— 现实中确实存在"高频闲逛但
    # 没有真实入金意愿"的人。这样才能让"高活跃 + 低意向"的 sleeping dogs 真实存在
    # (旧版把 intent 写成 visits 的函数,导致该人群恒为空)。
    intent_latent = rng.normal(5.0, 2.3, n)                     # 独立潜在意向
    intent_score = np.clip(intent_latent + 1.3 * viewed_deposit, 0, 10)

    # --- Treatment T:被投放概率依赖活跃度(活跃度越高越容易被广告系统触达)---
    # 系数经过中心化与软化,使倾向得分不过度极端、两侧 overlap 更充分
    # (旧版系数过大导致大量用户 e≈1、对照组有效样本量 ESS 偏低)。
    logit_t = (
        -0.4
        + 0.18 * (visits_30d - 5)           # 回访多 -> 更可能被投
        + 0.12 * (avg_duration - 6)         # 停留久 -> 更可能被投
        + 0.9 * viewed_deposit              # 看过入金页 -> 更可能被投
        + 0.12 * (intent_score - 5)         # 意向高 -> 更可能被投
        - 0.004 * days_since_reg            # 注册久了略降低
        + 0.3 * clicked_push                # 点过 push -> 更可能被投
    )
    propensity_true = 1.0 / (1.0 + np.exp(-logit_t))
    T = rng.binomial(1, propensity_true).astype(float)

    # --- 真实异质处理效应 tau(x):不同人群的真实 uplift 差异巨大 ---
    # 用**绝对值**显式赋值(而非叠加),让各人群效应清晰、可被模型学习。
    # tau(x) 完全由可观测特征(intent_score / viewed_deposit / visits)决定。
    tau_base = 0.08  # neutral 人群的基础增益(约 +8pp)

    # (1) Persuadables:中高意向(4 < intent < 7)+ 看过入金页但仍在犹豫
    #     -> 最受广告影响,真实 uplift 最高(+20pp)
    persuadables_mask = (intent_score > 4) & (intent_score < 7) & (viewed_deposit == 1)

    # (2) Sure things:超高意向(intent > 8),已下定决心 -> 无增益(treat 与否都入金)
    sure_things_mask = intent_score > 8

    # (3) Lost causes:极低意向(intent < 1.5) -> 无增益(怎么投也不入)
    lost_causes_mask = intent_score < 1.5

    # (4) Sleeping dogs:高活跃但低意向(visits > 7 且 intent < 3)
    #     —— 高频闲逛、被广告反复打扰而反感,负增益(-6pp)。
    #     现在 intent 独立于 visits,该人群真实存在。
    sleeping_dogs_mask = (visits_30d > 7) & (intent_score < 3)

    # 合成真实 tau(x):绝对赋值,后者覆盖前者(sleeping dogs 优先级最高)
    tau = np.full(n, tau_base)
    tau = np.where(persuadables_mask, 0.20, tau)    # +20pp
    tau = np.where(sure_things_mask, 0.01, tau)     # ~0
    tau = np.where(lost_causes_mask, 0.01, tau)     # ~0
    tau = np.where(sleeping_dogs_mask, -0.06, tau)  # 负增益

    # 标记人群(仅用于验证,不喂给模型)
    segment = np.where(
        sleeping_dogs_mask, "sleeping_dogs",
        np.where(
            persuadables_mask, "persuadables",
            np.where(sure_things_mask, "sure_things",
                     np.where(lost_causes_mask, "lost_causes", "neutral"))
        )
    )

    # --- Outcome Y:首次入金(二元,基线 + 处理效应)---
    # 基线入金概率(未被投放时):依赖活跃度与意向分(这些高活跃/高意向用户
    # 本来就更可能入金)—— 这是混淆的来源。
    # 注意:本人群是"注册未入金"客户,基线转化率应保持在较低、现实的区间
    # (约 15~30%),为处理效应留出空间、避免概率被 clip 到 1 而吃掉真实效应
    # (旧版基线高达 ~85%,导致 baseline+tau 触顶截断、AIPW 系统性低估)。
    logit_y0 = (
        -1.4
        + 0.10 * (visits_30d - 5)
        + 0.05 * (avg_duration - 6)
        + 0.5 * viewed_deposit
        + 0.20 * (intent_score - 5)
        - 0.003 * days_since_reg
        + 0.2 * clicked_push
        - 0.1 * (region_tier - 2)
    )
    baseline_prob = 1.0 / (1.0 + np.exp(-logit_y0))

    # 真实处理效应叠加:P(Y=1|T=1,X) = baseline_prob + T * tau(x)
    treated_prob = np.clip(baseline_prob + T * tau, 0, 1)
    Y = rng.binomial(1, treated_prob).astype(float)

    # 真实总体 ATE = E[tau]
    true_ate = tau.mean()

    df = pd.DataFrame({
        "visits_30d": visits_30d,
        "avg_duration": avg_duration,
        "viewed_deposit": viewed_deposit,
        "days_since_reg": days_since_reg,
        "device_mobile": device_mobile,
        "region_tier": region_tier,
        "clicked_push": clicked_push,
        "intent_score": intent_score,
        "T": T,
        "Y": Y,
        "tau": tau,  # 真实 CATE(仅验证用,建模时不可见)
        "segment": segment,
    })
    return df, true_ate


# ===========================================================================
# 2. 朴素估计(未调整混淆,对比基线)
# ===========================================================================
def naive_estimate(df):
    """朴素估计:被投放组 vs 未投放组的转化率差(含严重活跃度混淆偏差)。"""
    treated_conv = df.loc[df["T"] == 1, "Y"].mean()
    control_conv = df.loc[df["T"] == 0, "Y"].mean()
    return treated_conv - control_conv


# ===========================================================================
# 3. IPW 估计(Hajek + truncation + ESS)
# ===========================================================================
def estimate_propensity(df, feature_cols):
    """逻辑回归估计倾向得分 e(x) = P(T=1|X)。"""
    X = df[feature_cols].values
    T = df["T"].values
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(X, T)
    return model.predict_proba(X)[:, 1]


def compute_ipw_ate(df, e_hat, clip=0.02):
    """IPW(Hajek 归一化)ATE 估计 + ESS 诊断。"""
    T = df["T"].values
    Y = df["Y"].values
    e = np.clip(e_hat, clip, 1 - clip)

    # Hajek 归一化权重
    w1 = T / e
    w0 = (1 - T) / (1 - e)

    mu1 = np.sum(w1 * Y) / np.sum(w1)
    mu0 = np.sum(w0 * Y) / np.sum(w0)
    ate_ipw = mu1 - mu0

    # ESS 诊断
    ess_treated = (w1[T == 1].sum() ** 2) / (w1[T == 1] ** 2).sum()
    ess_control = (w0[T == 0].sum() ** 2) / (w0[T == 0] ** 2).sum()

    return ate_ipw, ess_treated, ess_control


# ===========================================================================
# 4. AIPW(双稳健,主估计量)+ cross-fitting
# ===========================================================================
def cross_fit_aipw(df, feature_cols, n_splits=5, clip=0.02, seed=SEED):
    """K 折 cross-fitting AIPW:倾向模型(LogisticRegression)+ 结果模型
    (GradientBoostingClassifier)。

    返回:ATE, SE, 95% CI。
    """
    n = len(df)
    X_all = df[feature_cols].values
    T_all = df["T"].values
    Y_all = df["Y"].values

    e_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    mu0_hat = np.zeros(n)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X_all):
        # 倾向模型
        prop_model = LogisticRegression(max_iter=1000, random_state=seed)
        prop_model.fit(X_all[train_idx], T_all[train_idx])
        e_hat[test_idx] = prop_model.predict_proba(X_all[test_idx])[:, 1]

        # 结果模型:treated / control 分别拟合
        tr = train_idx[T_all[train_idx] == 1]
        co = train_idx[T_all[train_idx] == 0]

        m1 = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed
        )
        m1.fit(X_all[tr], Y_all[tr])
        mu1_hat[test_idx] = m1.predict_proba(X_all[test_idx])[:, 1]

        m0 = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed + 1
        )
        m0.fit(X_all[co], Y_all[co])
        mu0_hat[test_idx] = m0.predict_proba(X_all[test_idx])[:, 1]

    # AIPW 逐单位得分(影响函数)
    e = np.clip(e_hat, clip, 1 - clip)
    psi = (
        mu1_hat - mu0_hat
        + T_all * (Y_all - mu1_hat) / e
        - (1 - T_all) * (Y_all - mu0_hat) / (1 - e)
    )
    ate = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(0.975)
    ci = (ate - z * se, ate + z * se)
    return ate, se, ci


# ===========================================================================
# 5. Uplift Modeling(去混淆 + T-learner/S-learner + Qini 评估)
# ===========================================================================
def fit_uplift_t_learner(df, feature_cols, use_weights=False, e_hat=None, clip=0.02):
    """T-learner:treated / control 各训一个分类器,预测概率之差为 uplift。

    关键:观测数据必须去混淆。这里用 IPW 权重训练(sample_weight),
    或在匹配后样本上建模(本例用加权)。
    """
    X = df[feature_cols].values
    T = df["T"].values
    Y = df["Y"].values

    # 若去混淆,构造 IPW 权重
    if use_weights and e_hat is not None:
        e = np.clip(e_hat, clip, 1 - clip)
        weights = np.where(T == 1, 1.0 / e, 1.0 / (1 - e))
        # 归一化权重,避免数值过大
        weights = weights / weights.mean()
    else:
        weights = None

    treated_mask = T == 1
    control_mask = T == 0

    # Treated 模型
    m1 = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=SEED
    )
    if weights is not None:
        m1.fit(X[treated_mask], Y[treated_mask],
               sample_weight=weights[treated_mask])
    else:
        m1.fit(X[treated_mask], Y[treated_mask])

    # Control 模型
    m0 = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=SEED + 1
    )
    if weights is not None:
        m0.fit(X[control_mask], Y[control_mask],
               sample_weight=weights[control_mask])
    else:
        m0.fit(X[control_mask], Y[control_mask])

    return m1, m0


def predict_uplift_t_learner(m1, m0, X, feature_cols):
    """T-learner 预测 uplift = P(Y=1|T=1,X) - P(Y=1|T=0,X)。"""
    mu1 = m1.predict_proba(X[feature_cols].values)[:, 1]
    mu0 = m0.predict_proba(X[feature_cols].values)[:, 1]
    return mu1 - mu0


def fit_uplift_s_learner(df, feature_cols, use_weights=False, e_hat=None, clip=0.02):
    """S-learner:单模型,把 T 当作特征,预测 T=1 vs T=0 概率之差为 uplift。"""
    # 用 numpy 数组拟合(列顺序 = feature_cols + ["T"]),预测时保持一致,
    # 避免 sklearn 的 feature-name 警告。
    X = np.column_stack([df[feature_cols].values, df["T"].values])
    Y = df["Y"].values

    if use_weights and e_hat is not None:
        e = np.clip(e_hat, clip, 1 - clip)
        T = df["T"].values
        weights = np.where(T == 1, 1.0 / e, 1.0 / (1 - e))
        weights = weights / weights.mean()
    else:
        weights = None

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=SEED + 2
    )
    if weights is not None:
        model.fit(X, Y, sample_weight=weights)
    else:
        model.fit(X, Y)

    return model


def predict_uplift_s_learner(model, X, feature_cols):
    """S-learner 预测 uplift:代入 T=1 vs T=0 求差。"""
    base = X[feature_cols].values
    X1 = np.column_stack([base, np.ones(len(base))])   # T=1
    X0 = np.column_stack([base, np.zeros(len(base))])  # T=0
    mu1 = model.predict_proba(X1)[:, 1]
    mu0 = model.predict_proba(X0)[:, 1]
    return mu1 - mu0


# ===========================================================================
# 6. Uplift 评估:Qini 曲线 / uplift 曲线 + AUUC
# ===========================================================================
def qini_curve(uplift_pred, T, Y):
    """Qini 曲线:按预测 uplift 降序排序,累积增量 =
    cum_treated_y - cum_control_y * (cum_treated_n / cum_control_n)。
    """
    order = np.argsort(-uplift_pred)
    T_sorted = T[order]
    Y_sorted = Y[order]
    n = len(T_sorted)

    cum_treated_n = np.cumsum(T_sorted)
    cum_control_n = np.cumsum(1 - T_sorted)
    cum_treated_y = np.cumsum(Y_sorted * T_sorted)
    cum_control_y = np.cumsum(Y_sorted * (1 - T_sorted))

    safe_c = np.where(cum_control_n == 0, 1, cum_control_n)
    qini_gain = cum_treated_y - cum_control_y * (cum_treated_n / safe_c)

    cum_n = np.arange(1, n + 1)
    x_frac = cum_n / n
    x_frac = np.concatenate([[0.0], x_frac])
    qini_gain = np.concatenate([[0.0], qini_gain])
    return x_frac, qini_gain


def auuc(x_frac, gain):
    """AUUC(Area Under Uplift Curve):梯形法积分。"""
    return trapezoid(gain, x_frac)


def qini_coefficient(x_frac, qini_gain):
    """Qini 系数:(模型面积 - 随机面积)/ |随机面积|,归一化 lift。"""
    model_area = auuc(x_frac, qini_gain)
    random_area = 0.5 * qini_gain[-1] * 1.0
    if abs(random_area) < 1e-9:
        return np.nan
    return (model_area - random_area) / abs(random_area)


# ===========================================================================
# 7. Uplift 分桶验证:真实 CATE 是否单调
# ===========================================================================
def cate_by_bucket(uplift_pred, true_tau, n_buckets=10):
    """按预测 uplift 分桶,返回每桶的预测均值与真实 CATE 均值。"""
    df_bucket = pd.DataFrame({"pred": uplift_pred, "tau": true_tau})
    df_bucket["bucket"] = pd.qcut(
        df_bucket["pred"].rank(method="first"), n_buckets, labels=False
    )
    grouped = df_bucket.groupby("bucket").agg(
        pred_uplift_mean=("pred", "mean"),
        true_cate_mean=("tau", "mean"),
        n=("tau", "size"),
    )
    return grouped


# ===========================================================================
# 8. Targeting 策略建议:Top-X% uplift 的增量收益
# ===========================================================================
def targeting_analysis(df, uplift_pred):
    """按预测 uplift 排序,计算若只对 Top-X% 投放的增量入金人数。"""
    df_target = df.copy()
    df_target["pred_uplift"] = uplift_pred
    df_target = df_target.sort_values("pred_uplift", ascending=False).reset_index(drop=True)

    rows = []
    for pct in [0.1, 0.2, 0.3, 0.5, 1.0]:
        n_target = int(pct * len(df_target))
        if n_target == 0:
            continue
        top = df_target.iloc[:n_target]
        # 用真实 CATE 计算:若全投放给这些人,增量入金 = sum(tau)
        incremental_deposits = top["tau"].sum()
        avg_cate = top["tau"].mean()
        rows.append({
            "target_pct": f"{pct * 100:.0f}%",
            "n_users": n_target,
            "avg_true_cate": avg_cate,
            "incremental_deposits": incremental_deposits,
        })

    return pd.DataFrame(rows)


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    feature_cols = [
        "visits_30d", "avg_duration", "viewed_deposit", "days_since_reg",
        "device_mobile", "region_tier", "clicked_push", "intent_score"
    ]

    print("=" * 80)
    print("案例 2:券商 Retargeting 广告因果效应评估(观测数据去混淆 + Uplift)")
    print("=" * 80)

    # ---- 1. 生成观测数据 ----
    df, true_ate = generate_retargeting_data()
    print(f"\n【数据概览】")
    print(f"样本量: {len(df)}")
    print(f"被投放组占比: {df['T'].mean():.3f}  (观测数据,非随机分配)")
    print(f"总体入金率: {df['Y'].mean():.3f}")
    print(f"真实总体 ATE = E[tau] = {true_ate:.4f}  ({true_ate * 100:.2f} pp)")

    print(f"\n【各人群(segment)的真实平均 CATE】")
    seg_stats = df.groupby("segment").agg(
        true_cate=("tau", "mean"),
        n=("tau", "size"),
        conv_rate=("Y", "mean"),
    )
    print(seg_stats.to_string(float_format=lambda v: f"{v:.4f}"))

    # ---- 2. 朴素估计(活跃度混淆未调整)----
    naive = naive_estimate(df)
    treated_conv = df.loc[df["T"] == 1, "Y"].mean()
    control_conv = df.loc[df["T"] == 0, "Y"].mean()
    print(f"\n【朴素估计(未调整活跃度混淆,严重高估)】")
    print(f"被投放组转化率: {treated_conv:.4f}")
    print(f"未投放组转化率: {control_conv:.4f}")
    print(f"朴素差异: {naive:.4f}  ({naive * 100:.2f} pp)  <- 含严重混淆偏差")
    print(f"真实 ATE: {true_ate:.4f}  ({true_ate * 100:.2f} pp)")
    print(f"朴素高估倍数: {naive / true_ate:.2f}x")

    # ---- 3. IPW 估计(对照)----
    print(f"\n【IPW 估计(Hajek 归一化 + truncation)】")
    e_hat = estimate_propensity(df, feature_cols)
    ate_ipw, ess_t, ess_c = compute_ipw_ate(df, e_hat, clip=0.02)
    print(f"倾向得分范围: [{e_hat.min():.4f}, {e_hat.max():.4f}]")
    print(f"ESS(treated): {ess_t:.1f} / {df['T'].sum():.0f}  "
          f"ESS(control): {ess_c:.1f} / {(df['T'] == 0).sum():.0f}")
    print(f"IPW ATE: {ate_ipw:.4f}  ({ate_ipw * 100:.2f} pp)")

    # ---- 4. AIPW 估计(主,双稳健)----
    print(f"\n【AIPW 估计(双稳健,主估计量)】")
    ate_aipw, se_aipw, ci_aipw = cross_fit_aipw(df, feature_cols, n_splits=5)
    print(f"AIPW ATE: {ate_aipw:.4f}  ({ate_aipw * 100:.2f} pp)")
    print(f"标准误 SE: {se_aipw:.4f}")
    print(f"95% CI: [{ci_aipw[0]:.4f}, {ci_aipw[1]:.4f}]")
    print(f"真实 ATE 是否落入 CI: {'是' if ci_aipw[0] <= true_ate <= ci_aipw[1] else '否'}")

    # ---- 对比表 ----
    print(f"\n【对比表:朴素 vs IPW vs AIPW vs 真实 ATE】")
    compare = pd.DataFrame([
        {"方法": "朴素(未调整)", "估计值": naive, "vs 真实 ATE 偏差": abs(naive - true_ate)},
        {"方法": "IPW", "估计值": ate_ipw, "vs 真实 ATE 偏差": abs(ate_ipw - true_ate)},
        {"方法": "AIPW(主)", "估计值": ate_aipw, "vs 真实 ATE 偏差": abs(ate_aipw - true_ate)},
        {"方法": "真实 ATE", "估计值": true_ate, "vs 真实 ATE 偏差": 0.0},
    ])
    print(compare.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 5. Uplift Modeling(去混淆 + T-learner/S-learner)----
    print(f"\n" + "=" * 80)
    print('【Uplift Modeling:Targeting 策略(回答"该向谁投放")】')
    print("=" * 80)
    print("关键:观测数据必须先去混淆(用 IPW 权重训练),否则 uplift 有偏。")

    # 训练/测试切分
    rng_split = np.random.default_rng(SEED)
    idx_split = rng_split.permutation(len(df))
    n_train = int(0.7 * len(df))
    train_df = df.iloc[idx_split[:n_train]].reset_index(drop=True)
    test_df = df.iloc[idx_split[n_train:]].reset_index(drop=True)

    # 在训练集上估计倾向(用于加权)
    e_train = estimate_propensity(train_df, feature_cols)

    # T-learner(用 IPW 权重去混淆)
    print(f"\n训练集样本量: {len(train_df)}, 测试集样本量: {len(test_df)}")
    m1_t, m0_t = fit_uplift_t_learner(
        train_df, feature_cols, use_weights=True, e_hat=e_train, clip=0.02
    )
    uplift_t = predict_uplift_t_learner(m1_t, m0_t, test_df, feature_cols)

    # S-learner(用 IPW 权重去混淆)
    m_s = fit_uplift_s_learner(
        train_df, feature_cols, use_weights=True, e_hat=e_train, clip=0.02
    )
    uplift_s = predict_uplift_s_learner(m_s, test_df, feature_cols)

    # ---- 6. Qini 曲线评估 ----
    print(f"\n【Qini 曲线 + AUUC 评估】")
    for name, uplift in [("T-learner", uplift_t), ("S-learner", uplift_s)]:
        xq, gq = qini_curve(uplift, test_df["T"].values, test_df["Y"].values)
        auuc_val = auuc(xq, gq)
        qcoef = qini_coefficient(xq, gq)
        print(f"{name:12s} | AUUC = {auuc_val:8.2f} | Qini 系数 = {qcoef:6.4f}")

    # ---- 7. 分桶验证:真实 CATE 单调性 ----
    print(f"\n【分桶验证(T-learner):按预测 uplift 十分位分桶,验证真实 CATE 单调】")
    buckets = cate_by_bucket(uplift_t, test_df["tau"].values, n_buckets=10)
    print(buckets.to_string(float_format=lambda v: f"{v:.4f}"))

    true_cate_seq = buckets["true_cate_mean"].values
    from scipy.stats import spearmanr
    rho_bucket, _ = spearmanr(buckets.index.values, true_cate_seq)
    print(f"\n桶序 vs 桶内真实 CATE 的 Spearman 相关 = {rho_bucket:.4f}  "
          f"(接近 1 表示排序准确)")
    print(f"最低桶真实 CATE = {true_cate_seq[0]:.4f}  "
          f"-> 最高桶真实 CATE = {true_cate_seq[-1]:.4f}")

    # ---- 8. 识别 persuadables / sleeping dogs ----
    print(f"\n【人群识别:Persuadables / Sure things / Sleeping dogs】")
    test_df_target = test_df.copy()
    test_df_target["pred_uplift"] = uplift_t

    neg_mask = test_df_target["pred_uplift"] < 0
    print(f"预测负 uplift(疑似 sleeping dogs)占比: {neg_mask.mean():.3f}")
    print("预测负 uplift 人群的真实 segment 分布:")
    print(test_df_target.loc[neg_mask, "segment"].value_counts())

    pos_high_mask = test_df_target["pred_uplift"] >= 0.10
    print(f"\n预测高 uplift(≥10pp,疑似 persuadables)占比: {pos_high_mask.mean():.3f}")
    print("预测高 uplift 人群的真实 segment 分布:")
    print(test_df_target.loc[pos_high_mask, "segment"].value_counts())

    # ---- 9. Targeting 策略建议表 ----
    print(f"\n【Targeting 策略建议:若只对 Top-X% 投放的增量收益】")
    targeting_table = targeting_analysis(test_df, uplift_t)
    print(targeting_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"
                                    if isinstance(v, float) else str(v)))

    print(f"\n建议:对预测 uplift 最高的前 30% 用户投放,")
    top30 = test_df_target.nlargest(int(0.3 * len(test_df_target)), "pred_uplift")
    print(f"  - 平均真实 CATE = {top30['tau'].mean():.4f}  "
          f"(显著高于全体真实 ATE {test_df['tau'].mean():.4f})")
    print(f"  - 增量入金人数 ≈ {top30['tau'].sum():.1f}  "
          f"(相比全量投放能节省 70% 预算,且集中在高增益人群)")

    # ---- 断言:核心验证 ----
    print(f"\n" + "=" * 80)
    print("【核心验证】")
    print("=" * 80)
    assert abs(ate_aipw - true_ate) < 0.02, "AIPW 估计偏离真实 ATE 过大"
    assert rho_bucket > 0.85, "高预测 uplift 分桶的真实 CATE 未单调上升"
    print("[OK] AIPW 估计接近真实 ATE,uplift 分桶真实 CATE 单调上升。")
    print("[OK] 脚本运行通过,已生成完整因果推断报告。")


if __name__ == "__main__":
    main()
