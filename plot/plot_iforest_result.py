import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 Isolation Forest 的 PCA 降维结果
df = pd.read_csv("build/result/iforest_pca_result.csv")

# 类型转换
df['pred'] = df['pred'].astype(int)
df['label'] = df['label'].astype(int)

# 创建可读性更高的分类列
df['ground_truth'] = df['label'].map({0: 'Normal', 1: 'Anomaly'})
df['prediction'] = df['pred'].map({0: 'Normal', 1: 'Predicted Anomaly'})

# 创建 1 行 2 列的子图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ========= 图 1: Ground Truth (正常 / 异常) =========
label_palette = {'Normal': 'green', 'Anomaly': 'red'}

sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='ground_truth',
    style='ground_truth',
    palette=label_palette,
    s=15,
    edgecolor='none',
    alpha=0.7,
    ax=axes[0]
)
axes[0].set_title("Ground Truth (Normal vs Anomaly)")
axes[0].legend(title="Label", bbox_to_anchor=(1.02, 1), loc='upper left')
axes[0].grid(True)

# ========= 图 2: Isolation Forest 判定结果 =========
pred_palette = {'Normal': 'green', 'Predicted Anomaly': 'blue'}

sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='prediction',
    style='prediction',
    palette=pred_palette,
    s=15,
    edgecolor='none',
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title("Predicted: Normal vs Anomaly (Isolation Forest)")
axes[1].legend(title="Prediction", bbox_to_anchor=(1.02, 1), loc='upper left')
axes[1].grid(True)

# 统一标签
for ax in axes:
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

# 保存图像
plt.tight_layout()
plt.savefig("pca_iforest.png", dpi=300)
print("✅ 图像保存为 pca_iforest.png")