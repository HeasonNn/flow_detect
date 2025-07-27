import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_csv("build/result/dbscan_pca_result.csv")

# 类型转换
df['assignments'] = df['assignments'].astype(int)
df['label'] = df['label'].astype(int)
df['is_outlier'] = df['is_outlier'].astype(bool)

# 创建新的分类列
df['ground_truth'] = df['label'].map({0: 'Normal', 1: 'Anomaly'})
df['prediction'] = df['is_outlier'].map({False: 'Normal', True: 'Predicted Anomaly'})

# 创建子图（1行3列）
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# ========= 图 1: DBSCAN 聚类结果 =========
assignments_palette = {
    cid: color for cid, color in zip(
        sorted(df['assignments'].unique()),
        sns.color_palette("hsv", len(df['assignments'].unique()))
    )
}
assignments_palette[-1] = "gray"

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


axes[0].set_title("DBSCAN Clustering (Assignment)")
# axes[0].legend(title="Assignment", bbox_to_anchor=(1.02, 1), loc='upper left')
axes[0].legend_.remove() 
axes[0].grid(True)

# ========= 图 2: Ground Truth (正常 / 异常) =========
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
    ax=axes[1]
)
axes[1].set_title("Ground Truth (Normal vs Anomaly)")
axes[1].legend(title="Label", bbox_to_anchor=(1.02, 1), loc='upper left')
axes[1].grid(True)

# ========= 图 3: 正常 vs 判定为异常 =========
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
    ax=axes[2]
)
axes[2].set_title("Predicted: Normal vs Anomaly")
axes[2].legend(title="Prediction", bbox_to_anchor=(1.02, 1), loc='upper left')
axes[2].grid(True)

# 统一标签
for ax in axes:
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")

# 调整并保存
plt.tight_layout()
plt.savefig("pca_dbscan.png", dpi=300)
print("✅ 图像保存为 pca_dbscan.png")