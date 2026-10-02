# -*- coding: utf-8 -*-
"""端到端演示：本地独立复现 BCI-decoding track 官方评分流程。

运行：``python demo_scoring.py``

流程
----
1. 生成一份合成测试集（4 类、19 通道、T=400，模拟 Stieger2021 的 4 类设定）
   以及一份同分布的合成训练集；
2. 训练一个"记忆类别模板"的最近质心分类器（TemplateCentroidClassifier），
   用 ``evaluate_model``（复现官方 ``evaluate_result``）评估，其
   balanced accuracy 应远高于随机水平 ``1 / n_classes``；
3. 用官方同款 ``ConstantClassifier``（永远预测类别 0）评估，其
   balanced accuracy 应约等于 ``1 / n_classes``；
4. 用 ``score_predictions`` 对同样的标签直接算分，断言与
   ``evaluate_model`` 路径结果数值完全一致；
5. 用 sklearn 直接对拼接后的 ``y_true / y_pred`` 独立计算两个指标做交叉
   验证并断言一致；
6. 打印对比表格（模型、balanced_accuracy、accuracy）。
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from local_scoring import (
    ConstantClassifier,
    evaluate_model,
    make_synthetic_loader,
    score_predictions,
    to_numpy,
)

# 模拟 Stieger2021 的 4 类设定
N_CLASSES = 4
N_CHANS = 19
N_TIMES = 400
N_TRAIN = 400
N_TEST = 200
BATCH_SIZE = 32
SEED_TRAIN = 0
SEED_TEST = 1


class TemplateCentroidClassifier:
    """最近模板质心分类器："记住"每个类别的 channel-mean 模板。

    fit：在训练集上按类别对窗口的时间平均 ``(B, C, T) -> (B, C)`` 求均值，
    得到每类一个质心 ``(n_classes, C)``。
    predict：对每个窗口的时间平均找欧氏距离最近的质心，返回类别 id。
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.centroids_ = None

    @staticmethod
    def _features(X):
        # (B, C, T) -> (B, C)：channel-mean（官方 Simulated 数据的判别信号）
        return to_numpy(X).mean(axis=-1)

    def fit(self, train_loader):
        feats, targets = [], []
        for X, y, _info in train_loader:
            feats.append(self._features(X))
            targets.append(to_numpy(y))
        feats = np.concatenate(feats)
        targets = np.concatenate(targets)
        self.centroids_ = np.stack(
            [feats[targets == k].mean(axis=0) for k in range(self.n_classes)]
        )
        return self

    def predict(self, X):
        feats = self._features(X)                                # (B, C)
        dists = ((feats[:, None, :] - self.centroids_[None]) ** 2).sum(-1)
        return dists.argmin(axis=1).astype(np.int64)             # (B,)


def collect_labels(model, test_loader):
    """按官方 evaluate_result 的遍历/拼接顺序收集 y_true 与 y_pred。"""
    y_true, y_pred = [], []
    for X, y, _info in test_loader:
        y_pred.append(to_numpy(model.predict(X)))
        y_true.append(to_numpy(y))
    return np.concatenate(y_true), np.concatenate(y_pred)


def main():
    torch.manual_seed(0)

    # 1. 合成数据（官方 Simulated 生成方式）：训练集 + 测试集
    train_loader = make_synthetic_loader(
        N_TRAIN, N_CHANS, N_TIMES, N_CLASSES, BATCH_SIZE, seed=SEED_TRAIN)
    # 官方 Simulated 中 train/test 共用同一组类别模板，此处同样共享
    test_loader = make_synthetic_loader(
        N_TEST, N_CHANS, N_TIMES, N_CLASSES, BATCH_SIZE, seed=SEED_TEST,
        templates=train_loader.dataset.templates)
    print(f"合成数据：4 类 / 19 通道 / T=400，"
          f"训练 {N_TRAIN} 窗、测试 {N_TEST} 窗（batch_size={BATCH_SIZE}）")
    print(f"随机水平（1 / n_classes）= {1 / N_CLASSES:.4f}\n")

    # 2. 质心分类器：evaluate_model（复现官方 evaluate_result）
    centroid = TemplateCentroidClassifier(N_CLASSES).fit(train_loader)
    res_centroid = evaluate_model(centroid, test_loader, N_CLASSES)

    # 3. ConstantClassifier（官方 get_one_result 的 sanity-check 模型）
    constant = ConstantClassifier()
    res_constant = evaluate_model(constant, test_loader, N_CLASSES)

    # 4. score_predictions 路径：对同样的标签直接算分，断言完全一致
    y_true_c, y_pred_c = collect_labels(centroid, test_loader)
    res_centroid_direct = score_predictions(y_true_c, y_pred_c, N_CLASSES)
    y_true_k, y_pred_k = collect_labels(constant, test_loader)
    res_constant_direct = score_predictions(y_true_k, y_pred_k, N_CLASSES)

    assert res_centroid_direct == res_centroid, \
        "score_predictions 与 evaluate_model 结果不一致（centroid）"
    assert res_constant_direct == res_constant, \
        "score_predictions 与 evaluate_model 结果不一致（constant）"
    print("[断言通过] score_predictions 与 evaluate_model 两条路径结果完全一致")

    # 5. sklearn 独立交叉验证：直接对拼接后的 y_true/y_pred 计算指标
    for name, yt, yp, res in [
        ("TemplateCentroid", y_true_c, y_pred_c, res_centroid),
        ("Constant", y_true_k, y_pred_k, res_constant),
    ]:
        ba = balanced_accuracy_score(yt, yp)
        acc = accuracy_score(yt, yp)
        assert res["balanced_accuracy"] == ba, f"{name}: balanced_accuracy 不一致"
        assert res["accuracy"] == acc, f"{name}: accuracy 不一致"
        assert res["n_classes"] == N_CLASSES, f"{name}: n_classes 不一致"
    print("[断言通过] sklearn 独立计算的 balanced_accuracy / accuracy 与评分模块一致")

    # 6. sanity 阈值断言：质心显著高于随机，Constant 约等于随机
    assert res_centroid["balanced_accuracy"] > 0.9, \
        f"质心分类器 balanced accuracy 过低：{res_centroid['balanced_accuracy']}"
    assert abs(res_constant["balanced_accuracy"] - 1 / N_CLASSES) < 0.05, \
        f"ConstantClassifier balanced accuracy 应≈1/n_classes：" \
        f"{res_constant['balanced_accuracy']}"
    print(f"[断言通过] 质心分类器 balanced_accuracy "
          f"({res_centroid['balanced_accuracy']:.4f}) 显著高于随机 "
          f"({1 / N_CLASSES:.4f})；"
          f"ConstantClassifier ({res_constant['balanced_accuracy']:.4f}) "
          f"≈ 1/n_classes\n")

    # 7. 对比表格
    print(f"{'模型':<28}{'balanced_accuracy':>20}{'accuracy':>12}")
    print("-" * 60)
    for name, res in [
        ("TemplateCentroidClassifier", res_centroid),
        ("ConstantClassifier", res_constant),
    ]:
        print(f"{name:<28}{res['balanced_accuracy']:>20.4f}"
              f"{res['accuracy']:>12.4f}")
    print("-" * 60)
    print(f"n_classes = {res_centroid['n_classes']}")
    print("\n全部断言通过，demo 运行成功。")


if __name__ == "__main__":
    main()
