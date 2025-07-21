import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Seaborn 主题风格
sns.set(style='whitegrid', font_scale=1.1)

# 读取数据
df = pd.read_csv("build/result/iforest_pca_result.csv")

# 类型转换
df['pred'] = df['pred'].astype(int)
df['label'] = df['label'].astype(int)

# 添加可读列
df['ground_truth'] = df['label'].map({0: 'Normal', 1: 'Anomaly'})
df['prediction'] = df['pred'].map({0: 'Normal', 1: 'Predicted Anomaly'})

# 定义调色板（更现代的颜色组合）
label_palette = {'Normal': '#4CAF50', 'Anomaly': '#F44336'}  # 绿色 & 红色
pred_palette = {'Normal': '#4CAF50', 'Predicted Anomaly': '#2196F3'}  # 绿色 & 蓝色

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图 1: Ground Truth
sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='ground_truth',
    style='ground_truth',
    palette=label_palette,
    s=25,
    edgecolor='black',
    linewidth=0.05,
    alpha=0.7,
    ax=axes[0]
)
axes[0].set_title("Ground Truth (Normal vs Anomaly)", fontsize=14)
axes[0].legend(title="Label", bbox_to_anchor=(1.02, 1), loc='upper left')
axes[0].set_xlabel("PCA Component 1")
axes[0].set_ylabel("PCA Component 2")
axes[0].grid(True, linestyle='--', alpha=0.6)

# 图 2: Predicted Results
sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='prediction',
    style='prediction',
    palette=pred_palette,
    s=25,
    edgecolor='black',
    linewidth=0.05,
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title("Predicted by Isolation Forest", fontsize=14)
axes[1].legend(title="Prediction", bbox_to_anchor=(1.02, 1), loc='upper left')
axes[1].set_xlabel("PCA Component 1")
axes[1].set_ylabel("PCA Component 2")
axes[1].grid(True, linestyle='--', alpha=0.6)

# 保存图像
plt.tight_layout()
plt.savefig("pca_iforest.png", dpi=300)
print("✅ 图像保存为 pca_iforest.png")