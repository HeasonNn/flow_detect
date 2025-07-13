import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("build/result/dbscan_pca_result.csv")

# 类型转换（可选）
df['assignments'] = df['assignments'].astype(int)
df['label'] = df['label'].astype(int)

# 创建子图（1 行 2 列）
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# =========================
# 子图 1：按 assignment 分簇
# =========================
assignments_palette = {
    cid: color for cid, color in zip(
        sorted(df['assignments'].unique()),
        sns.color_palette("hsv", len(df['assignments'].unique()))
    )
}
assignments_palette[-1] = "gray"  # 噪声为灰色

sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='assignments',
    style='assignments',
    palette=assignments_palette,
    s=15,
    edgecolor='none',
    alpha=0.7,
    ax=axes[0]
)

axes[0].set_title("DBSCAN Clustering by Assignment")
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
axes[0].legend(title="Assignment", bbox_to_anchor=(1.02, 1), loc='upper left')
axes[0].grid(True)

# =========================
# 子图 2：按 label（攻击/正常）
# =========================
label_palette = {0: "green", 1: "red"}

sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='label',
    style='label',
    palette=label_palette,
    s=15,
    edgecolor='none',
    alpha=0.7,
    ax=axes[1]
)

axes[1].set_title("Ground Truth Label (0=Normal, 1=Attack)")
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")
axes[1].legend(title="Label", bbox_to_anchor=(1.02, 1), loc='upper left')
axes[1].grid(True)

# 调整布局并保存
plt.tight_layout()
plt.savefig("pca_dbscan.png", dpi=300)
print("✅ 图像保存为 pca_dbscan.png")