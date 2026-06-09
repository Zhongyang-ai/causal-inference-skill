"""
Uplift Modeling(增益建模 / 个体处理效应 CATE)端到端可运行示例。

本脚本只依赖 numpy / pandas / scikit-learn / scipy(无 statsmodels,
也不依赖 causalml / scikit-uplift 等第三方 uplift 库,全部手写),
演示一条完整链路:

  1. 固定随机种子,生成合成的随机实验(A/B test)数据:
     - treatment T 完全随机分配 -> 无混淆,treated/control 直接可比;
     - 协变量 X;
     - 真实 CATE tau(x) 随 X 异质:存在高增益人群、零增益人群,
       以及一个 sleeping dogs 子群(真实 CATE 为负,处理反而有害);
     - outcome Y 由基线响应 + T * tau(x) + 噪声生成。
  2. 手写 T-learner(treated/control 各训一个回归器,预测之差为 uplift)
     与 S-learner(单模型把 T 当特征,预测 T=1 与 T=0 之差)。
  3. 为每个个体预测 uplift(CATE)。
  4. 手写 Qini 曲线 / uplift 曲线,打印 AUUC 与 Qini 系数。
  5. 验证:按预测 uplift 分桶,桶内真实 CATE 均值应随桶单调上升。
  6. 打印 S-learner vs T-learner 对比,以及整体 ATE 校验。

运行:python3 example.py
"""

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

RANDOM_SEED = 42


# ----------------------------------------------------------------------------
# 1. 合成随机实验数据
# ----------------------------------------------------------------------------
def generate_experiment_data(n=20000, seed=RANDOM_SEED):
    """生成一个无混淆的随机实验数据集,真实 CATE 随 X 异质。

    返回 DataFrame,包含:
      - x0..x4  : 协变量
      - segment : 人群标签(便于解读,不喂给模型)
      - T        : 处理标记(完全随机,0/1)
      - tau      : 个体真实 CATE(仅用于验证,建模时不可见)
      - Y        : 观测结果
    """
    rng = np.random.default_rng(seed)

    # 五个协变量
    x0 = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(-1, 1, n)
    x3 = rng.binomial(1, 0.5, n).astype(float)
    x4 = rng.normal(0, 1, n)

    # 基线响应水平(不处理时的结果期望),与 x0/x3 相关
    baseline = 2.0 + 1.5 * x0 + 0.8 * x3 - 0.5 * x4

    # 真实异质 CATE:
    #   - x1 越大,处理增益越高(persuadables 集中在 x1 高的一端);
    #   - x2 > 0.5 的子群为 sleeping dogs:真实 CATE 为负,处理有害;
    #   - 其余人群增益接近 0(sure things / lost causes)。
    tau = 0.4 + 1.8 * np.maximum(x1, 0)        # x1>0 时增益随 x1 上升
    tau = tau - 0.3 * (x1 < -0.5)              # x1 很小的一端增益再低一些
    sleeping_dogs = x2 > 0.5
    tau = np.where(sleeping_dogs, -1.5, tau)   # sleeping dogs:负增益

    # 人群标签(仅解读用)
    segment = np.where(
        sleeping_dogs, "sleeping_dogs",
        np.where(x1 > 0.8, "persuadables",
                 np.where(x1 < -0.5, "lost_causes", "sure_things")),
    )

    # 随机分配处理(A/B test):与 X、tau 独立 -> 无混淆
    T = rng.binomial(1, 0.5, n)

    # 观测结果:基线 + 处理时叠加真实 CATE + 噪声
    noise = rng.normal(0, 1.0, n)
    Y = baseline + T * tau + noise

    df = pd.DataFrame(
        {
            "x0": x0, "x1": x1, "x2": x2, "x3": x3, "x4": x4,
            "segment": segment,
            "T": T,
            "tau": tau,
            "Y": Y,
        }
    )
    return df


# ----------------------------------------------------------------------------
# 2. 元学习器:T-learner 与 S-learner(手写)
# ----------------------------------------------------------------------------
def _make_base_learner(seed=RANDOM_SEED):
    """统一的 base learner 工厂,便于 S/T 公平对比。"""
    return GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )


def fit_t_learner(X_train, T_train, Y_train, feature_cols):
    """T-learner:treated / control 各训一个回归器。

    返回 (model_treated, model_control)。
    uplift = mu1(x) - mu0(x)。
    """
    treated_mask = T_train == 1
    control_mask = T_train == 0

    model_treated = _make_base_learner(seed=RANDOM_SEED)
    model_control = _make_base_learner(seed=RANDOM_SEED + 1)

    model_treated.fit(X_train.loc[treated_mask, feature_cols], Y_train[treated_mask])
    model_control.fit(X_train.loc[control_mask, feature_cols], Y_train[control_mask])
    return model_treated, model_control


def predict_t_learner(models, X, feature_cols):
    """T-learner 预测 uplift。"""
    model_treated, model_control = models
    mu1 = model_treated.predict(X[feature_cols])
    mu0 = model_control.predict(X[feature_cols])
    return mu1 - mu0


def fit_s_learner(X_train, T_train, Y_train, feature_cols):
    """S-learner:单模型,把 T 当作一个普通特征一起拟合。"""
    model = _make_base_learner(seed=RANDOM_SEED + 2)
    X_aug = X_train[feature_cols].copy()
    X_aug["T"] = T_train  # 处理标记作为特征
    model.fit(X_aug, Y_train)
    return model


def predict_s_learner(model, X, feature_cols):
    """S-learner 预测 uplift:分别代入 T=1 与 T=0 求差。"""
    X1 = X[feature_cols].copy()
    X1["T"] = 1
    X0 = X[feature_cols].copy()
    X0["T"] = 0
    return model.predict(X1) - model.predict(X0)


# ----------------------------------------------------------------------------
# 4. 评估:Qini 曲线 / uplift 曲线(手写)
# ----------------------------------------------------------------------------
def uplift_curve(uplift_pred, T, Y):
    """计算 uplift 曲线。

    做法:按预测 uplift 降序排序,沿排序累积,在每个位置计算
        累积增量 = (treated 累积响应均值 - control 累积响应均值) * 当前累积人数
    返回 (x_frac, uplift_gain):
        x_frac      : 已处理(被选中)人群占总体的比例 [0, 1]
        uplift_gain : 对应的累积增量响应
    """
    order = np.argsort(-uplift_pred)  # 降序
    T_sorted = np.asarray(T)[order]
    Y_sorted = np.asarray(Y)[order]
    n = len(T_sorted)

    # 累积的 treated / control 人数与响应
    cum_treated_n = np.cumsum(T_sorted)
    cum_control_n = np.cumsum(1 - T_sorted)
    cum_treated_y = np.cumsum(Y_sorted * T_sorted)
    cum_control_y = np.cumsum(Y_sorted * (1 - T_sorted))

    # 避免除零
    safe_t = np.where(cum_treated_n == 0, 1, cum_treated_n)
    safe_c = np.where(cum_control_n == 0, 1, cum_control_n)

    # uplift 曲线:两组累积响应率之差,再乘以当前累积总人数
    rate_diff = (cum_treated_y / safe_t) - (cum_control_y / safe_c)
    cum_n = np.arange(1, n + 1)
    uplift_gain = rate_diff * cum_n

    x_frac = cum_n / n
    # 起点补 (0, 0)
    x_frac = np.concatenate([[0.0], x_frac])
    uplift_gain = np.concatenate([[0.0], uplift_gain])
    return x_frac, uplift_gain


def qini_curve(uplift_pred, T, Y):
    """计算 Qini 曲线。

    Qini 把 control 累积响应按 treated/control 规模重标定后相减:
        qini = cum_treated_y - cum_control_y * (cum_treated_n / cum_control_n)
    返回 (x_frac, qini_gain)。
    """
    order = np.argsort(-uplift_pred)
    T_sorted = np.asarray(T)[order]
    Y_sorted = np.asarray(Y)[order]
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


def area_under_curve(x_frac, gain):
    """梯形法求曲线下面积(AUUC / Qini 面积的原始值)。"""
    return trapezoid(gain, x_frac)


def qini_coefficient(x_frac, qini_gain):
    """Qini 系数:模型曲线与随机对角线之间的面积,按随机面积归一化。

    随机对角线:从 (0,0) 直线连到终点 (1, qini_gain[-1])。
    """
    model_area = area_under_curve(x_frac, qini_gain)
    # 随机线下的面积(梯形)= 0.5 * 终点高度 * 1
    random_area = 0.5 * qini_gain[-1] * 1.0
    if random_area == 0:
        return np.nan
    return (model_area - random_area) / abs(random_area)


# ----------------------------------------------------------------------------
# 5. 验证:按预测 uplift 分桶,看桶内真实 CATE 是否单调
# ----------------------------------------------------------------------------
def cate_by_bucket(uplift_pred, true_tau, n_buckets=10):
    """按预测 uplift 分桶(十分位),返回每桶的真实 CATE 均值与预测均值。"""
    df = pd.DataFrame({"pred": uplift_pred, "tau": true_tau})
    # 用 rank 再切分,保证桶大小均匀,避免重复边界报错
    df["bucket"] = pd.qcut(df["pred"].rank(method="first"), n_buckets, labels=False)
    grouped = df.groupby("bucket").agg(
        pred_uplift_mean=("pred", "mean"),
        true_cate_mean=("tau", "mean"),
        n=("tau", "size"),
    )
    return grouped


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    np.random.seed(RANDOM_SEED)
    feature_cols = ["x0", "x1", "x2", "x3", "x4"]

    # ---- 数据 ----
    df = generate_experiment_data()
    print("=" * 70)
    print("数据概览")
    print("=" * 70)
    print(f"样本量: {len(df)}")
    print(f"处理组占比: {df['T'].mean():.3f}  (随机分配,应接近 0.5)")
    print(f"真实总体 ATE = E[tau] = {df['tau'].mean():.4f}")
    print("\n各人群(segment)的真实平均 CATE:")
    print(
        df.groupby("segment")["tau"].agg(["mean", "size"]).rename(
            columns={"mean": "true_cate", "size": "n"}
        )
    )

    # ---- 训练/测试切分 ----
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(df))
    n_train = int(0.7 * len(df))
    train_df = df.iloc[idx[:n_train]].reset_index(drop=True)
    test_df = df.iloc[idx[n_train:]].reset_index(drop=True)

    # ---- 朴素估计:实验下 treated 均值 - control 均值,作为 ATE 校验 ----
    naive_ate = (
        train_df.loc[train_df["T"] == 1, "Y"].mean()
        - train_df.loc[train_df["T"] == 0, "Y"].mean()
    )
    print("\n" + "=" * 70)
    print("整体 ATE 校验(随机实验下 treated - control 应接近真实 ATE)")
    print("=" * 70)
    print(f"朴素 ATE (treated 均值 - control 均值) = {naive_ate:.4f}")
    print(f"真实 ATE                              = {df['tau'].mean():.4f}")

    # ---- 训练 T-learner 与 S-learner ----
    t_models = fit_t_learner(train_df, train_df["T"].values, train_df["Y"].values, feature_cols)
    s_model = fit_s_learner(train_df, train_df["T"].values, train_df["Y"].values, feature_cols)

    # ---- 在测试集上预测 uplift(CATE) ----
    uplift_t = predict_t_learner(t_models, test_df, feature_cols)
    uplift_s = predict_s_learner(s_model, test_df, feature_cols)
    true_tau_test = test_df["tau"].values

    # ---- 评估:Qini / uplift 曲线 + AUUC + Qini 系数 ----
    print("\n" + "=" * 70)
    print("排序评估(测试集):Qini 曲线 / uplift 曲线")
    print("=" * 70)
    for name, uplift in [("T-learner", uplift_t), ("S-learner", uplift_s)]:
        xu, gu = uplift_curve(uplift, test_df["T"].values, test_df["Y"].values)
        xq, gq = qini_curve(uplift, test_df["T"].values, test_df["Y"].values)
        auuc = area_under_curve(xu, gu)
        qcoef = qini_coefficient(xq, gq)
        # 预测 uplift 与真实 CATE 的 Spearman 排序相关(模拟下可算,实务不可见)
        rho, _ = spearmanr(uplift, true_tau_test)
        print(
            f"{name:10s} | AUUC = {auuc:10.2f} | Qini系数 = {qcoef:7.4f} "
            f"| Spearman(pred, true tau) = {rho:6.4f}"
        )

    # ---- 验证:按预测 uplift 分桶,真实 CATE 应单调上升 ----
    print("\n" + "=" * 70)
    print("分桶验证(T-learner):按预测 uplift 十分位分桶,看真实 CATE 是否单调")
    print("=" * 70)
    buckets_t = cate_by_bucket(uplift_t, true_tau_test, n_buckets=10)
    print(buckets_t.to_string(float_format=lambda v: f"{v:.4f}"))

    true_cate_seq = buckets_t["true_cate_mean"].values
    diffs = np.diff(true_cate_seq)
    monotonic = np.all(diffs >= -1e-9)
    spearman_bucket, _ = spearmanr(buckets_t.index.values, true_cate_seq)
    print(
        f"\n桶序 vs 桶内真实 CATE 的 Spearman 相关 = {spearman_bucket:.4f} "
        f"(接近 1 表示高预测 uplift 桶的真实 CATE 确实更高)"
    )
    print(f"真实 CATE 是否严格单调不减: {monotonic}")
    print(
        f"最低桶真实 CATE = {true_cate_seq[0]:.4f}  "
        f"-> 最高桶真实 CATE = {true_cate_seq[-1]:.4f}"
    )

    # ---- S vs T learner 对比小结 ----
    print("\n" + "=" * 70)
    print("S-learner vs T-learner 对比小结")
    print("=" * 70)
    mae_t = np.mean(np.abs(uplift_t - true_tau_test))
    mae_s = np.mean(np.abs(uplift_s - true_tau_test))
    print(f"对真实 CATE 的 MAE(仅模拟下可算):T-learner = {mae_t:.4f} | S-learner = {mae_s:.4f}")
    print(f"测试集预测平均 uplift:           T-learner = {uplift_t.mean():.4f} | "
          f"S-learner = {uplift_s.mean():.4f} | 真实 ATE = {true_tau_test.mean():.4f}")

    # ---- 简单 targeting 演示:识别 sleeping dogs(负 uplift)----
    print("\n" + "=" * 70)
    print("Targeting 演示:按 T-learner 预测 uplift 划分人群")
    print("=" * 70)
    test_df = test_df.copy()
    test_df["pred_uplift"] = uplift_t
    neg_mask = test_df["pred_uplift"] < 0
    print(f"预测负 uplift(疑似 sleeping dogs)占比: {neg_mask.mean():.3f}")
    print("预测负 uplift 人群的真实 segment 分布:")
    print(test_df.loc[neg_mask, "segment"].value_counts())
    print(
        f"\n建议:对预测 uplift 最高的前 30% 投放,"
        f"其真实 CATE 均值 = "
        f"{test_df.nlargest(int(0.3*len(test_df)), 'pred_uplift')['tau'].mean():.4f} "
        f"(显著高于全体真实 ATE {true_tau_test.mean():.4f})"
    )

    # ---- 断言:核心验证项必须通过 ----
    assert spearman_bucket > 0.8, "分桶真实 CATE 未随预测 uplift 单调上升"
    assert mae_t < 1.0, "T-learner 对真实 CATE 的误差过大"
    print("\n[OK] 核心验证通过:高预测 uplift 分桶的真实 CATE 确实更高。")


if __name__ == "__main__":
    main()
